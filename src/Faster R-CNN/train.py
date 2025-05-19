import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from data import YoloToFasterRCNNDataset
from model import model
import pandas as pd
import os
import pathlib
import sys
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import datetime
import hashlib 

def _make_run_dir():
    ts  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # fingerprint cfg so the same hyper-params always share the suffix
    cfg_hash = hashlib.sha1(yaml.dump(cfg).encode()).hexdigest()[:6]
    run_dir = Path("runs") / f"{ts}_{cfg_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def parse_cli() :
    p = argparse.ArgumentParser(
        description="Train Faster-RCNN on your YOLO-style dataset",
        )
    # required paths
    p.add_argument("--root_dir",        required=True,default=None,
                  help="Dataset root that contains images/ and labels/ folders")
    p.add_argument("--config", default="configs/baseline.yaml",
               help="Path to YAML experiment config")
    p.add_argument("-e", "--epochs", dest="epochs", type=int, default=100)
    p.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=1)
    p.add_argument("--lr", dest="lr", type=float, default=1e-4)
    return p.parse_args()

ARGS = parse_cli()           # <─ parse once; use everywhere

with open(ARGS.config, "r") as f:
    cfg = yaml.safe_load(f)
    EPOCHS         = cfg.get("epochs", ARGS.epochs)
    BATCH_SIZE     = cfg.get("batch_size", ARGS.batch_size)
    LEARNING_RATE  = cfg.get("learning_rate", ARGS.lr)

ROOT_DIR = ARGS.root_dir
print(EPOCHS)
# Validate function
metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5, 0.75, 0.95])

def compute_map(predictions, targets):
    metric.reset()
    formatted_predictions = []
    formatted_targets = []
    for pred, target in zip(predictions, targets):
        if "boxes" in pred and "scores" in pred and "labels" in pred:
            formatted_predictions.append({
                "boxes": pred["boxes"].detach().cpu(),
                "scores": pred["scores"].detach().cpu(),
                "labels": pred["labels"].detach().cpu()
            })
        if "boxes" in target and "labels" in target:
            formatted_targets.append({
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu()
            })
    metric.update(formatted_predictions, formatted_targets)
    results = metric.compute()
    return results["map_50"].item(), results["map"].item()

@torch.no_grad()
def validate():
    model.train()
    total_val_loss = 0
    all_predictions = []
    all_targets = []

    val_cls_loss = 0
    val_box_loss = 0
    val_rpn_obj_loss = 0
    val_rpn_box_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation", leave=False):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):
                total_val_loss += sum(loss for loss in loss_dict.values()).item()
                val_cls_loss += loss_dict["loss_classifier"].item()
                val_box_loss += loss_dict["loss_box_reg"].item()
                val_rpn_obj_loss += loss_dict["loss_objectness"].item()
                val_rpn_box_loss += loss_dict["loss_rpn_box_reg"].item()

            model.eval()
            outputs = model(images)
            all_predictions.extend(outputs)
            all_targets.extend(targets)
            model.train()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_cls_loss = val_cls_loss / len(val_loader)
    avg_box_loss = val_box_loss / len(val_loader)
    avg_rpn_obj_loss = val_rpn_obj_loss / len(val_loader)
    avg_rpn_box_loss = val_rpn_box_loss / len(val_loader)

    val_map50, val_map95 = compute_map(all_predictions, all_targets)
    return avg_val_loss, avg_cls_loss, avg_box_loss, avg_rpn_obj_loss, avg_rpn_box_loss, val_map50, val_map95

# Training Loop

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7  # 6 vehicle classes + background

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
dataset = YoloToFasterRCNNDataset(
    root_dir=ROOT_DIR,
    classes=["Car", "Motorcycle", "Bus", "Taxi", "Medium Vehicle", "Heavy Vehicle"]
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Load from checkpoint if exists

# checkpoint
if cfg.get("checkpoint"):
    checkpoint_path = Path(cfg["checkpoint"])
else:
    run_dir = _make_run_dir()
    checkpoint_path = run_dir / "model_last.pth"

# loss-history CSV
if cfg.get("loss_history"):
    loss_history_path = Path(cfg["loss_history"])
else:
    run_dir = checkpoint_path.parent if "run_dir" in locals() else _make_run_dir()
    loss_history_path = run_dir / "loss_history.csv"

if loss_history_path.is_file():                      # ← guard
    df = pd.read_csv(loss_history_path)
    start_epoch = df["Epoch"].max() + 1
    train_losses  = df["Train Loss"].tolist()
    val_losses    = df["Val Loss"].tolist()
    map50_scores = df["mAP50"].tolist()
    map50_95_scores= df["mAP50-95"].tolist()
    cls_losses    = df["Classification Loss"].tolist()
    box_losses    = df["Box Regression Loss"].tolist()
    rpn_obj_losses = df["RPN Objectness Loss"].tolist()
    rpn_box_losses = df["RPN Box Loss"].tolist()
    print(f"✅ Resuming from epoch {start_epoch}")
else:
    start_epoch = 1
    train_losses, val_losses, map50_scores, map50_95_scores = [], [], [], []
    cls_losses, box_losses, rpn_obj_losses, rpn_box_losses = [], [], [], []
    print("ℹ️ No previous loss history found — starting from scratch")

# be nice to callers: ensure parent dirs exist
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
loss_history_path.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0
    cls_loss = 0
    box_loss = 0
    rpn_obj_loss = 0
    rpn_box_loss = 0

    for images, targets in tqdm(train_loader,
                                desc=f"Epoch {epoch}/{EPOCHS} Training",
                                leave=False): 
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)

        if isinstance(loss_dict, dict):
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            cls_loss += loss_dict["loss_classifier"].item()
            box_loss += loss_dict["loss_box_reg"].item()
            rpn_obj_loss += loss_dict["loss_objectness"].item()
            rpn_box_loss += loss_dict["loss_rpn_box_reg"].item()

    avg_train_loss = total_loss / len(train_loader)
    avg_cls_loss = cls_loss / len(train_loader)
    avg_box_loss = box_loss / len(train_loader)
    avg_rpn_obj_loss = rpn_obj_loss / len(train_loader)
    avg_rpn_box_loss = rpn_box_loss / len(train_loader)

    avg_val_loss, val_cls_loss, val_box_loss, val_rpn_obj_loss, val_rpn_box_loss, val_map50, val_map95 = validate()

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    cls_losses.append(avg_cls_loss)
    box_losses.append(avg_box_loss)
    rpn_obj_losses.append(avg_rpn_obj_loss)
    rpn_box_losses.append(avg_rpn_box_loss)
    map50_scores.append(val_map50)
    map50_95_scores.append(val_map95)

    df = pd.DataFrame({
        "Epoch": list(range(1, len(train_losses) + 1)),
        "Train Loss": train_losses,
        "Val Loss": val_losses,
        "Classification Loss": cls_losses,
        "Box Regression Loss": box_losses,
        "RPN Objectness Loss": rpn_obj_losses,
        "RPN Box Loss": rpn_box_losses,
        "mAP50": map50_scores,
        "mAP50-95": map50_95_scores
    })
    df.to_csv(loss_history_path, index=False)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | mAP50: {val_map50:.3f} | mAP50-95: {val_map95:.3f}")

    torch.save(model.state_dict(), checkpoint_path)
