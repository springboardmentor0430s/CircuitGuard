import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

def get_train_transforms(image_size=128):
    """
    Data augmentation transforms for training
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15, 
            p=0.5, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=128):
    """
    Transforms for validation/test (no augmentation)
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])