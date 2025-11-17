"""
EfficientNet model for PCB defect classification
"""

import torch
import torch.nn as nn
import timm


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B4 based classifier for PCB defects
    """
    
    def __init__(self, num_classes: int = 6, 
                 pretrained: bool = True,
                 dropout: float = 0.3,
                 in_channels: int = 1):
        """
        Initialize model
        
        Args:
            num_classes: Number of defect classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
            in_channels: Number of input channels (1 for grayscale)
        """
        super(EfficientNetClassifier, self).__init__()
        
        # Load EfficientNet-B4
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            in_chans=in_channels
        )
        
        # Get number of features from backbone
        num_features = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output logits (B, num_classes)
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def create_model(config: dict, device: torch.device) -> nn.Module:
    """
    Create and initialize model
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    model = EfficientNetClassifier(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout'],
        in_channels=1  # Grayscale images
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: EfficientNet-B4")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def count_parameters(model: nn.Module) -> tuple:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable