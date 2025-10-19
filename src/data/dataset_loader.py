import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from .augment import get_train_transforms, get_val_transforms

class PCBDefectDataset(Dataset):
    def __init__(self, dataset_path, is_train=True, image_size=128):
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.image_size = image_size
        self.transforms = get_train_transforms(image_size) if is_train else get_val_transforms(image_size)
        
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        defect_types = [d for d in os.listdir(dataset_path) 
                       if os.path.isdir(os.path.join(dataset_path, d))]
        defect_types.sort()
        
        # Create mapping
        self.class_to_idx = {defect_type: idx for idx, defect_type in enumerate(defect_types)}
        self.idx_to_class = {idx: defect_type for defect_type, idx in self.class_to_idx.items()}
        
        # Load images
        for defect_type in defect_types:
            defect_path = os.path.join(dataset_path, defect_type)
            if not os.path.isdir(defect_path):
                continue
                
            images_in_class = [f for f in os.listdir(defect_path) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in images_in_class:
                self.images.append(os.path.join(defect_path, img_name))
                self.labels.append(self.class_to_idx[defect_type])
        
        print(f"Loaded {len(self.images)} images from {dataset_path}")
        print(f"Classes: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        return image, label
    
    def get_class_distribution(self):
        """Get count of each class"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {self.idx_to_class[idx]: count for idx, count in zip(unique, counts)}