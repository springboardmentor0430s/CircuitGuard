# ==============================================================
# src/model.py  --  EfficientNet-B4 model definition
# ==============================================================

from builtins import len, print
import torch.nn as nn
import torchvision.models as models
import os

def build_efficientnet_b4(num_classes=None, pretrained=True, data_root="data/processed_rois"):
    """
    Builds an EfficientNet-B4 model for PCB defect classification.
    Automatically detects number of classes if not specified.
    """

    if num_classes is None:
        num_classes = len([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and d.startswith("class_")
        ])
        print(f"[INFO] Auto-detected {num_classes} classes.")

    # Load pretrained EfficientNet-B4 from torchvision
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)

    # Replace final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
