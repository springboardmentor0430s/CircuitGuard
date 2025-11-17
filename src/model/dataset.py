"""
PyTorch Dataset for PCB defect classification
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PCBDefectDataset(Dataset):
    """
    Dataset for PCB defect ROI classification
    """
    
    def __init__(self, root_dir: str, 
                 class_names: List[str],
                 transform: Optional[Callable] = None,
                 grayscale: bool = True):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory containing class folders
            class_names: List of class names (in order)
            transform: Albumentations transform
            grayscale: Whether to load images as grayscale
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        self.grayscale = grayscale
        
        # Create class to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # Load all image paths and labels
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Load all image paths and their labels
        
        Returns:
            List of (image_path, label) tuples
        """
        samples = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all images in this class directory
            image_files = list(Path(class_dir).glob('*.jpg'))
            
            for img_path in image_files:
                samples.append((str(img_path), class_idx))
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        if self.grayscale:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Add channel dimension for grayscale
            image = np.expand_dims(image, axis=-1)
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: normalize and convert to tensor
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """
        Get distribution of samples per class
        
        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {name: 0 for name in self.class_names}
        
        for _, label in self.samples:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        
        return distribution


def get_train_transform(config: dict) -> A.Compose:
    """
    Get training data augmentation transform
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Albumentations transform
    """
    aug_config = config['augmentation']['train']
    
    transforms = [
        A.HorizontalFlip(p=aug_config['horizontal_flip']),
        A.VerticalFlip(p=aug_config['vertical_flip']),
        A.Rotate(limit=aug_config['rotation_limit'], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=aug_config['brightness_limit'],
            contrast_limit=aug_config['contrast_limit'],
            p=0.5
        ),
        A.GaussianBlur(blur_limit=aug_config['blur_limit'], p=0.3),
        A.Normalize(mean=[0.5], std=[0.5]),  # Normalize for grayscale
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def get_val_transform(config: dict) -> A.Compose:
    """
    Get validation/test transform (no augmentation)
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Albumentations transform
    """
    transforms = [
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def create_dataloaders(config: dict) -> Tuple:
    """
    Create train, validation, and test dataloaders
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    roi_dataset_path = config['data']['roi_dataset_path']
    class_names = config['class_names']
    batch_size = config['training']['batch_size']
    
    # Create datasets
    train_dataset = PCBDefectDataset(
        root_dir=os.path.join(roi_dataset_path, 'train'),
        class_names=class_names,
        transform=get_train_transform(config),
        grayscale=True
    )
    
    val_dataset = PCBDefectDataset(
        root_dir=os.path.join(roi_dataset_path, 'val'),
        class_names=class_names,
        transform=get_val_transform(config),
        grayscale=True
    )
    
    test_dataset = PCBDefectDataset(
        root_dir=os.path.join(roi_dataset_path, 'test'),
        class_names=class_names,
        transform=get_val_transform(config),
        grayscale=True
    )
    
    # Print dataset info
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    print(f"\nTrain class distribution:")
    for class_name, count in train_dataset.get_class_distribution().items():
        print(f"  {class_name}: {count}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader