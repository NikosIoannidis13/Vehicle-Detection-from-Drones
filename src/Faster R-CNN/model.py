import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from data import YoloToFasterRCNNDataset
import pandas as pd
import os


# Fine tune anchor boxes
custom_anchor_sizes = ((18,), (40,), (110,), (50,), (80,))
custom_aspect_ratios = ((0.5, 1.0, 2.0),) * len(custom_anchor_sizes)

# Create an anchor generator with custom settings
anchor_generator = AnchorGenerator(sizes=custom_anchor_sizes, aspect_ratios=custom_aspect_ratios)

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# We replace default RPN with custom anchor generator
model.rpn.anchor_generator = anchor_generator

# Update Class Predictor for Custom Classes
NUM_CLASSES = 7  # 6 vehicle classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)

# Move model to correct device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)



