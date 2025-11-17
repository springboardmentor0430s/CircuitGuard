"""
Model evaluation and metrics
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
from pathlib import Path
import json

from src.model.model import PCBDefectClassifier
from src.model.dataset import PCBDefectDataset
from src.utils.file_operations import load_config


def load_best_model(checkpoint_path: str, config: dict, device: str = 'cpu'):
    """
    Load the best trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = PCBDefectClassifier(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate model on dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        class_names: List of class names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return results


def save_evaluation_results(results, output_dir):
    """
    Save evaluation results to files
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics = {
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score'])
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(results['classification_report'])
    
    # Save confusion matrix as numpy array
    np.save(
        os.path.join(output_dir, 'confusion_matrix.npy'),
        results['confusion_matrix']
    )
    
    print(f"Results saved to {output_dir}")


def plot_confusion_matrix_lazy(cm, class_names, output_path):
    """
    Plot confusion matrix with lazy matplotlib import
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Confusion matrix plot saved to {output_path}")
    except ImportError:
        print("Matplotlib not available - skipping confusion matrix plot")


def main():
    """Main evaluation function"""
    
    # Load config
    config = load_config()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load best model
    checkpoint_path = os.path.join(
        config['training']['checkpoint_dir'],
        'best_model.pth'
    )
    
    print("Loading best model...")
    model = load_best_model(checkpoint_path, config, device)
    
    # Load test dataset
    test_dataset = PCBDefectDataset(
        root_dir=os.path.join(config['data']['processed_path'], 'test'),
        transform=None
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    print("\nEvaluating model on test set...")
    results = evaluate_model(
        model,
        test_loader,
        device,
        config['data']['class_names']
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save results
    output_dir = 'outputs/evaluation_results'
    save_evaluation_results(results, output_dir)
    
    # Plot confusion matrix (with lazy import)
    plot_confusion_matrix_lazy(
        results['confusion_matrix'],
        config['data']['class_names'],
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    print("\n" + "="*50)
    print("Evaluation complete!")
    print("="*50)


if __name__ == "__main__":
    main()