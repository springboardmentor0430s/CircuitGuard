"""
Test dataset and model setup
"""

import torch
from src.model.dataset import create_dataloaders
from src.model.model import create_model
from src.utils.file_operations import load_config


def test_dataset():
    """
    Test dataset loading
    """
    print("="*60)
    print("TESTING DATASET")
    print("="*60)
    
    config = load_config()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Test loading a batch
    print("\nTesting batch loading...")
    images, labels = next(iter(train_loader))
    
    print(f"\nBatch info:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Labels: {labels[:10].tolist()}")
    
    # Show class distribution in batch
    print(f"\nClass distribution in batch:")
    for i in range(config['model']['num_classes']):
        count = (labels == i).sum().item()
        print(f"  Class {i} ({config['class_names'][i]}): {count}")
    
    return train_loader, val_loader, test_loader


def test_model():
    """
    Test model creation and forward pass
    """
    print("\n" + "="*60)
    print("TESTING MODEL")
    print("="*60)
    
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    model = create_model(config, device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 128, 128).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nForward pass successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return model


def main():
    """
    Run all tests
    """
    # Test dataset
    train_loader, val_loader, test_loader = test_dataset()
    
    # Test model
    model = test_model()
    
    # Test model with real batch
    print("\n" + "="*60)
    print("TESTING MODEL WITH REAL DATA")
    print("="*60)
    
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    images, labels = next(iter(train_loader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
    
    print(f"\nPredictions for first 10 samples:")
    print(f"  True labels: {labels[:10].tolist()}")
    print(f"  Predictions: {predictions[:10].cpu().tolist()}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nReady to start training!")


if __name__ == "__main__":
    main()