import torch
import torch.nn as nn
import timm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PCBDefectClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(PCBDefectClassifier, self).__init__()
        
        self.backbone = timm.create_model('efficientnet_b4', 
                                         pretrained=pretrained,
                                         num_classes=0)  
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def get_model(num_classes=6, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = PCBDefectClassifier(num_classes=num_classes)
    return model.to(device)

def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }, filepath)

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    return model, optimizer, epoch, accuracy