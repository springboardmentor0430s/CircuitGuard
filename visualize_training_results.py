"""
Visualize complete training results
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.file_operations import load_config, create_directory


def plot_training_history(history_path: str, output_dir: str):
    """
    Plot training history curves
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[0, 1].axhline(y=97, color='g', linestyle='--', label='Target (97%)', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    
    best_val_acc = max(history['val_acc'])
    best_val_epoch = history['val_acc'].index(best_val_acc) + 1
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    min_val_loss = min(history['val_loss'])
    
    summary_text = f"""
TRAINING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Epochs: {len(epochs)}

Best Validation Accuracy: {best_val_acc:.2f}%
  (Epoch {best_val_epoch})

Final Training Accuracy: {final_train_acc:.2f}%
Final Validation Accuracy: {final_val_acc:.2f}%

Minimum Validation Loss: {min_val_loss:.4f}

Target Achievement: {'✓ YES' if best_val_acc >= 97 else '✗ NO'}
  (Target: ≥ 97%)
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, 
                   fontsize=11, family='monospace', 
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {output_path}")
    plt.show()


def create_comprehensive_report(config: dict, output_dir: str):
    """
    Create comprehensive project report
    """
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE PROJECT REPORT")
    print("="*60)
    
    report = []
    report.append("="*80)
    report.append("CIRCUITGUARD-PCB: DEFECT DETECTION & CLASSIFICATION")
    report.append("Final Project Report")
    report.append("="*80)
    report.append("")
    
    # Project overview
    report.append("PROJECT OVERVIEW:")
    report.append("-"*80)
    report.append("Objective: Automated PCB defect detection and classification using")
    report.append("           image subtraction and deep learning (EfficientNet-B4)")
    report.append("")
    report.append("Approach:")
    report.append("  1. Reference-based defect detection (image subtraction)")
    report.append("  2. Morphological processing and contour extraction")
    report.append("  3. Deep learning classification (EfficientNet-B4)")
    report.append("")
    
    # Dataset statistics
    report.append("DATASET STATISTICS:")
    report.append("-"*80)
    
    # Load dataset stats
    roi_stats_path = os.path.join(config['data']['roi_dataset_path'], 'dataset_statistics.csv')
    if os.path.exists(roi_stats_path):
        df = pd.read_csv(roi_stats_path, index_col=0)
        report.append("\nDefect Distribution Across Splits:")
        report.append(df.to_string())
        report.append(f"\nTotal defects: {df.sum().sum()}")
    
    report.append("")
    
    # Model architecture
    report.append("MODEL ARCHITECTURE:")
    report.append("-"*80)
    report.append(f"Base Model: EfficientNet-B4")
    report.append(f"Input Size: 128x128 (grayscale)")
    report.append(f"Number of Classes: {config['model']['num_classes']}")
    report.append(f"Classes: {', '.join(config['class_names'])}")
    report.append(f"Dropout Rate: {config['model']['dropout']}")
    report.append("")
    
    # Training configuration
    report.append("TRAINING CONFIGURATION:")
    report.append("-"*80)
    report.append(f"Optimizer: {config['optimizer']['type'].upper()}")
    report.append(f"Learning Rate: {config['training']['learning_rate']}")
    report.append(f"Batch Size: {config['training']['batch_size']}")
    report.append(f"Max Epochs: {config['training']['num_epochs']}")
    report.append(f"Early Stopping Patience: {config['training']['early_stopping_patience']}")
    report.append("")
    
    # Data augmentation
    report.append("DATA AUGMENTATION:")
    report.append("-"*80)
    aug = config['augmentation']['train']
    report.append(f"Horizontal Flip: {aug['horizontal_flip']}")
    report.append(f"Vertical Flip: {aug['vertical_flip']}")
    report.append(f"Rotation: ±{aug['rotation_limit']}°")
    report.append(f"Brightness/Contrast: ±{aug['brightness_limit']}")
    report.append(f"Gaussian Blur: up to {aug['blur_limit']}px")
    report.append("")
    
    # Training results
    report.append("TRAINING RESULTS:")
    report.append("-"*80)
    
    history_path = os.path.join(config['paths']['checkpoints'], 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        
        report.append(f"Total Epochs Trained: {len(history['train_loss'])}")
        report.append(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
        report.append(f"Final Training Accuracy: {final_train_acc:.2f}%")
        report.append(f"Final Validation Accuracy: {final_val_acc:.2f}%")
        report.append(f"Target Achievement (≥97%): {'✓ YES' if best_val_acc >= 97 else '✗ NO'}")
    
    report.append("")
    
    # Test results
    report.append("TEST SET EVALUATION:")
    report.append("-"*80)
    
    eval_results_path = os.path.join(config['paths']['results'], 'evaluation_results.json')
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
        
        overall = eval_results['overall']
        report.append(f"Test Accuracy: {overall['accuracy']*100:.2f}%")
        report.append(f"Test Precision: {overall['precision']:.4f}")
        report.append(f"Test Recall: {overall['recall']:.4f}")
        report.append(f"Test F1-Score: {overall['f1_score']:.4f}")
        report.append("")
        report.append("Per-Class Performance:")
        
        for class_name in config['class_names']:
            metrics = eval_results['per_class'][class_name]
            report.append(f"  {class_name}:")
            report.append(f"    Precision: {metrics['precision']:.4f}")
            report.append(f"    Recall: {metrics['recall']:.4f}")
            report.append(f"    F1-Score: {metrics['f1_score']:.4f}")
            report.append(f"    Support: {metrics['support']}")
    
    report.append("")
    
    # Deliverables
    report.append("DELIVERABLES:")
    report.append("-"*80)
    report.append("✓ Trained EfficientNet-B4 model (models/checkpoints/best_model.pth)")
    report.append("✓ Training history and metrics (models/checkpoints/training_history.json)")
    report.append("✓ Confusion matrix visualization")
    report.append("✓ Classification report")
    report.append("✓ Per-class accuracy plots")
    report.append("✓ Prediction pipeline with annotations")
    report.append("✓ TensorBoard logs (models/logs/)")
    report.append("✓ Complete evaluation report")
    report.append("")
    
    # Milestone completion
    report.append("MILESTONE COMPLETION STATUS:")
    report.append("-"*80)
    report.append("✓ Milestone 1: Dataset Preparation & Image Processing")
    report.append("  - Dataset inspection and organization")
    report.append("  - Image alignment (ORB feature matching)")
    report.append("  - Image subtraction and difference maps")
    report.append("  - Otsu thresholding")
    report.append("  - Morphological operations")
    report.append("  - Contour extraction and ROI detection")
    report.append("")
    report.append("✓ Milestone 2: Model Training & Evaluation")
    report.append("  - EfficientNet-B4 implementation")
    report.append("  - Data augmentation pipeline")
    report.append("  - Model training with Adam optimizer")
    report.append("  - Validation and early stopping")
    report.append("  - Test set evaluation")
    report.append("  - Confusion matrix and metrics")
    report.append("  - Prediction pipeline implementation")
    report.append("")
    
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    # Save report
    report_path = os.path.join(output_dir, 'FINAL_PROJECT_REPORT.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\nComprehensive report saved to: {report_path}")
    
    # Print to console
    print("\n" + '\n'.join(report))


def main():
    """
    Generate all final visualizations and reports
    """
    print("="*60)
    print("GENERATING FINAL PROJECT DOCUMENTATION")
    print("="*60)
    
    config = load_config()
    output_dir = config['paths']['results']
    
    # Plot training history
    history_path = os.path.join(config['paths']['checkpoints'], 'training_history.json')
    if os.path.exists(history_path):
        print("\n1. Creating training history visualization...")
        plot_training_history(history_path, output_dir)
    
    # Create comprehensive report
    print("\n2. Creating comprehensive project report...")
    create_comprehensive_report(config, output_dir)
    
    print("\n" + "="*60)
    print("DOCUMENTATION GENERATION COMPLETE")
    print("="*60)
    print(f"\nAll files saved to: {output_dir}/")


if __name__ == "__main__":
    main()