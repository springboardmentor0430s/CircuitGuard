"""
Main training script for PCB defect classification
"""

import torch
import torch.nn as nn
from src.model.dataset import create_dataloaders
from src.model.model import create_model
from src.model.trainer import Trainer
from src.utils.file_operations import load_config


def main():
    """
    Main training function
    """
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dataloaders
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    model = create_model(config, device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    history = trainer.train()
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Final train accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final val accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Best val accuracy: {max(history['val_acc']):.2f}%")
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    
    print("\nModel saved to: models/checkpoints/best_model.pth")
    print("TensorBoard logs: models/logs/")
    print("\nTo view training progress, run:")
    print("  tensorboard --logdir=models/logs")


if __name__ == "__main__":
    main()