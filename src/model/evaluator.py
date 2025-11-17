"""
Model evaluation and testing
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support
)
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.file_operations import create_directory


class ModelEvaluator:
    """
    Evaluates trained model on test set
    """
    
    def __init__(self, model: nn.Module, 
                 test_loader,
                 class_names: List[str],
                 device: torch.device,
                 output_dir: str = "outputs/training_results"):
        """
        Initialize evaluator
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            class_names: List of class names
            device: Device to run on
            output_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.output_dir = output_dir
        create_directory(output_dir)
        
        self.model.eval()
    
    def evaluate(self) -> Dict:
        """
        Evaluate model on test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL ON TEST SET")
        print("="*60)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(all_labels, all_predictions, average=None)
        
        # Print results
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {precision_per_class[i]:.4f}")
            print(f"    Recall:    {recall_per_class[i]:.4f}")
            print(f"    F1-Score:  {f1_per_class[i]:.4f}")
            print(f"    Support:   {support_per_class[i]}")
        
        # Create results dictionary
        results = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'per_class': {
                self.class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                }
                for i in range(len(self.class_names))
            },
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probabilities.tolist()
        }
        
        # Save results
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {results_path}")
        
        return results
    
    def plot_confusion_matrix(self, results: Dict, save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save plot
        """
        # Lazy import visualization libraries
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        predictions = np.array(results['predictions'])
        labels = np.array(results['labels'])
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot absolute confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        # Plot normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax2, cbar_kws={'label': 'Percentage'})
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_classification_report(self, results: Dict, save_path: str = None):
        """
        Plot classification report as heatmap
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save plot
        """
        # Lazy import visualization libraries
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        per_class = results['per_class']
        
        # Create DataFrame
        data = []
        for class_name in self.class_names:
            metrics = per_class[class_name]
            data.append([
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score']
            ])
        
        df = pd.DataFrame(data, 
                         columns=['Precision', 'Recall', 'F1-Score'],
                         index=self.class_names)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Class', fontsize=12, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Classification report saved to: {save_path}")
        
        plt.show()
    
    def plot_per_class_accuracy(self, results: Dict, save_path: str = None):
        """
        Plot per-class accuracy bar chart
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save plot
        """
        # Lazy import visualization libraries
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        predictions = np.array(results['predictions'])
        labels = np.array(results['labels'])
        
        # Calculate per-class accuracy
        accuracies = []
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == labels[class_mask]).mean()
                accuracies.append(class_acc * 100)
            else:
                accuracies.append(0)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(self.class_names, accuracies, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Classification Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # Add horizontal line for overall accuracy
        overall_acc = results['overall']['accuracy'] * 100
        ax.axhline(y=overall_acc, color='red', linestyle='--', 
                  label=f'Overall: {overall_acc:.1f}%', linewidth=2)
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class accuracy plot saved to: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results: Dict):
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*60)
        print("GENERATING EVALUATION REPORT")
        print("="*60)
        
        # Plot confusion matrix
        cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(results, save_path=cm_path)
        
        # Plot classification report
        report_path = os.path.join(self.output_dir, 'classification_report.png')
        self.plot_classification_report(results, save_path=report_path)
        
        # Plot per-class accuracy
        acc_path = os.path.join(self.output_dir, 'per_class_accuracy.png')
        self.plot_per_class_accuracy(results, save_path=acc_path)
        
        # Generate text report
        report_text = self._generate_text_report(results)
        report_txt_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        with open(report_txt_path, 'w') as f:
            f.write(report_text)
        print(f"\nText report saved to: {report_txt_path}")
        
        print("\n" + "="*60)
        print("EVALUATION REPORT COMPLETE")
        print("="*60)
        print(f"\nAll results saved to: {self.output_dir}/")
    
    def _generate_text_report(self, results: Dict) -> str:
        """
        Generate text evaluation report
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Report text
        """
        report = []
        report.append("="*60)
        report.append("PCB DEFECT CLASSIFICATION - EVALUATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE:")
        report.append("-"*60)
        overall = results['overall']
        report.append(f"Accuracy:  {overall['accuracy']*100:.2f}%")
        report.append(f"Precision: {overall['precision']:.4f}")
        report.append(f"Recall:    {overall['recall']:.4f}")
        report.append(f"F1-Score:  {overall['f1_score']:.4f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS PERFORMANCE:")
        report.append("-"*60)
        for class_name in self.class_names:
            metrics = results['per_class'][class_name]
            report.append(f"\n{class_name}:")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall:    {metrics['recall']:.4f}")
            report.append(f"  F1-Score:  {metrics['f1_score']:.4f}")
            report.append(f"  Support:   {metrics['support']}")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)


def load_best_model(config: dict, device: torch.device) -> nn.Module:
    """
    Load best trained model
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    from src.model.model import create_model
    
    # Create model
    model = create_model(config, device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config['paths']['checkpoints'], 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded best model from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best val accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    return model