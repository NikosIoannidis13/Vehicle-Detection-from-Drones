#!/usr/bin/env python
# inference.py
import argparse
from pathlib import Path
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator

# --------------------------------------------------
# 1. ──────────────── CLI ARGUMENTS ────────────────
# --------------------------------------------------
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch inference with a trained Faster-RCNN model"
    )

    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to .pth checkpoint (trained weights)")
    p.add_argument("--input_dir", required=True, type=Path,
                   help="Folder with images (*.jpg | *.png | *.jpeg)")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Folder to save images with detections")
    p.add_argument("--csv_dir",     required=True, type=Path,
                   help="Folder to save per-image CSV files")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Confidence threshold (default: 0.5)")

    return p.parse_args()


ARGS = parse_cli()

# --------------------------------------------------
# 2. ─────────────── SETUP  (unchanged) ────────────
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7  # 6 vehicle classes + background

# anchor sizes / aspect ratios must match training
custom_anchor_sizes  = ((18,), (40,), (110,), (50,), (80,))
custom_aspect_ratios = ((0.5, 1.0, 2.0),) * len(custom_anchor_sizes)
custom_anchor_generator = AnchorGenerator(
    sizes=custom_anchor_sizes,
    aspect_ratios=custom_aspect_ratios
)

# build model skeleton
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = \
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
model.rpn.anchor_generator = custom_anchor_generator

# load weights
model.load_state_dict(torch.load(ARGS.checkpoint, map_location=DEVICE))
model.to(DEVICE).eval()
print("✅ Model loaded and ready for batch inference!")

CLASS_NAMES = ["Background", "Car", "Motorcycle", "Bus",
               "Taxi", "Medium Vehicle", "Heavy Vehicle"]

# --------------------------------------------------
# 3. ────────────────  HELPERS  ────────────────────
# --------------------------------------------------
def convert_bbox_to_gt_format(box):
    x_min, y_min, x_max, y_max = box
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    return cx, cy, x_max - x_min, y_max - y_min

# --------------------------------------------------
# 4. ───────────────   INFERENCE   ────────────────
# --------------------------------------------------
def infer_folder(input_dir: Path, output_dir: Path, csv_dir: Path,
                 confidence_threshold: float = 0.5):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("⚠️ No images found in the folder!")
        return

    for img_file in image_files:
        img_path = input_dir / img_file
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            predictions = model(image_tensor)[0]

        boxes   = predictions["boxes"].cpu().numpy()
        scores  = predictions["scores"].cpu().numpy()
        labels  = predictions["labels"].cpu().numpy()

        keep = scores > confidence_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        results = []
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = map(int, box)
            cx, cy, bw, bh = convert_bbox_to_gt_format(box)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, f"{CLASS_NAMES[label]}: {score:.2f}",
                        (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            results.append([img_file, CLASS_NAMES[label],
                            x_min, y_min, x_max, y_max,
                            cx, cy, bw, bh, score])

        out_img_path = output_dir / img_file
        cv2.imwrite(str(out_img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved image: {out_img_path}")

        csv_path = csv_dir / f"{Path(img_file).stem}.csv"
        pd.DataFrame(results, columns=[
            "Image", "Type", "X_min", "Y_min", "X_max", "Y_max",
            "cx", "cy", "box_w", "box_h", "score"
        ]).to_csv(csv_path, index=False)
        print(f"✅ Saved CSV : {csv_path}")

    print("🎯 Batch inference completed successfully!")

# --------------------------------------------------
# 5. ─────────────────── main() ────────────────────
# --------------------------------------------------
if __name__ == "__main__":
    infer_folder(
        input_dir   = ARGS.input_dir,
        output_dir  = ARGS.output_dir,
        csv_dir     = ARGS.csv_dir,
        confidence_threshold = ARGS.conf
    )
