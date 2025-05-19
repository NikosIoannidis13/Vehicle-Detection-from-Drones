import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class YoloToFasterRCNNDataset(Dataset):
    def __init__(self, root_dir, classes, transforms=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "Frames")
        self.label_dir = os.path.join(root_dir, "Labels_YOLO_upright")
        self.transforms = transforms
        self.classes = classes
        self.imgs = sorted(os.listdir(self.img_dir))
        self.labels = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)  # Normalize and convert


        label_path = os.path.join(self.label_dir, self.labels[idx])
        boxes = []
        labels = []

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])


                x_min = (cx - w / 2) * width
                y_min = (cy - h / 2) * height
                x_max = (cx + w / 2) * width
                y_max = (cy + h / 2) * height

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id + 1)  # Class IDs should be 1-indexed

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": (torch.tensor(boxes)[:, 2] - torch.tensor(boxes)[:, 0]) * (torch.tensor(boxes)[:, 3] - torch.tensor(boxes)[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return img, target
