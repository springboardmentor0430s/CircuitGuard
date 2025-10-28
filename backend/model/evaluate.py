# model/evaluate.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import os
import sys
import json
from tqdm import tqdm

# Add the current directory to Python path
sys.path.append('.')

from model.efficientnet import PCBDefectClassifier, load_checkpoint
from src.data.dataset_loader import PCBDefectDataset

class ModelEvaluator:
    def __init__(self, model_path='model/best_model.pth', test_data_path='data/defect_dataset/test'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.test_data_path = test_data_path
        
        # Load model
        self.model = PCBDefectClassifier(num_classes=6).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        if os.path.exists(model_path):
            self.model, self.optimizer, self.epoch, self.accuracy = load_checkpoint(
                self.model, self.optimizer, model_path
            )
            print(f"✅ Loaded model from {model_path}")
            print(f"   Trained until epoch {self.epoch}, Validation accuracy: {self.accuracy:.2f}%")
        else:
            print(f"❌ Model not found at {model_path}")
            return
        
        self.model.eval()
        
        # Load test dataset
        self.test_dataset = PCBDefectDataset(test_data_path, is_train=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=32, shuffle=False, num_workers=0
        )
        
        # Class names
        self.class_names = list(self.test_dataset.idx_to_class.values())
        
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("🧪 Starting Model Evaluation...")
        print("=" * 60)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        # Run inference
        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Running inference"):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions) * 100
        
        print(f"📊 Test Set Size: {len(all_targets)} images")
        print(f"🎯 Overall Test Accuracy: {accuracy:.2f}%")
        
        # Generate detailed reports
        self._generate_classification_report(all_targets, all_predictions)
        self._generate_confusion_matrix(all_targets, all_predictions)
        self._generate_confidence_analysis(all_probabilities, all_predictions, all_targets)
        self._generate_error_analysis(all_predictions, all_targets)
        
        return accuracy, all_predictions, all_targets, all_probabilities
    
    def _generate_classification_report(self, targets, predictions):
        """Generate detailed classification report"""
        print("\n📈 Classification Report:")
        print("-" * 50)
        
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names,
            digits=4,
            output_dict=True
        )
        
        # Print detailed report
        print(classification_report(
            targets, predictions, 
            target_names=self.class_names,
            digits=4
        ))
        
        # Save report
        with open('evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create metrics summary
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df.to_csv('classification_metrics.csv')
        
        return report
    
    def _generate_confusion_matrix(self, targets, predictions):
        """Generate and plot confusion matrix"""
        print("\n🔄 Generating Confusion Matrix...")
        
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title('Confusion Matrix - PCB Defect Classification')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Calculate precision and recall from confusion matrix
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        
        print("\n📋 Class-wise Performance:")
        print("-" * 50)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:15} | Precision: {precision[i]:.3f} | Recall: {recall[i]:.3f}")
    
    def _generate_confidence_analysis(self, probabilities, predictions, targets):
        """Analyze prediction confidence"""
        print("\n🎯 Confidence Analysis...")
        
        # Get confidence for correct and incorrect predictions
        correct_confidences = []
        incorrect_confidences = []
        
        for i, (pred, target, prob) in enumerate(zip(predictions, targets, probabilities)):
            confidence = prob[pred]
            if pred == target:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
        
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Confidence distribution
        plt.subplot(1, 2, 1)
        if correct_confidences:
            plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
        if incorrect_confidences:
            plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Confidence vs Accuracy
        plt.subplot(1, 2, 2)
        confidence_bins = np.linspace(0, 1, 11)
        accuracy_per_bin = []
        
        for i in range(len(confidence_bins) - 1):
            low = confidence_bins[i]
            high = confidence_bins[i + 1]
            mask = (probabilities.max(axis=1) >= low) & (probabilities.max(axis=1) < high)
            if mask.sum() > 0:
                bin_accuracy = (predictions[mask] == targets[mask]).mean() * 100
                accuracy_per_bin.append(bin_accuracy)
            else:
                accuracy_per_bin.append(0)
        
        plt.plot(confidence_bins[:-1], accuracy_per_bin, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy (%)')
        plt.title('Confidence vs Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('confidence_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Average confidence - Correct: {np.mean(correct_confidences):.3f}")
        print(f"❌ Average confidence - Incorrect: {np.mean(incorrect_confidences):.3f}")
    
    def _generate_error_analysis(self, predictions, targets):
        """Analyze error patterns"""
        print("\n🔍 Error Analysis...")
        
        errors = predictions != targets
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            print("🎉 No errors found! Perfect classification.")
            return
        
        error_analysis = {}
        for idx in error_indices:
            true_class = self.class_names[targets[idx]]
            pred_class = self.class_names[predictions[idx]]
            error_type = f"{true_class} → {pred_class}"
            
            if error_type not in error_analysis:
                error_analysis[error_type] = 0
            error_analysis[error_type] += 1
        
        # Sort by frequency
        error_analysis = dict(sorted(error_analysis.items(), key=lambda x: x[1], reverse=True))
        
        print("Most common errors:")
        for error, count in list(error_analysis.items())[:10]:
            print(f"  {error}: {count} times")
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        error_types = list(error_analysis.keys())[:10]  # Top 10 errors
        error_counts = list(error_analysis.values())[:10]
        
        plt.barh(error_types, error_counts, color='coral')
        plt.xlabel('Error Count')
        plt.title('Top 10 Most Common Classification Errors')
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_final_report(self, accuracy, report):
        """Generate comprehensive final report"""
        print("\n" + "="*60)
        print("📊 FINAL EVALUATION REPORT")
        print("="*60)
        
        # Overall statistics
        total_samples = len(self.test_dataset)
        error_count = total_samples - (accuracy * total_samples / 100)
        
        print(f"Model: EfficientNet-B4")
        print(f"Test Dataset: {total_samples} samples")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"Error Count: {int(error_count)}")
        print(f"Error Rate: {100 - accuracy:.2f}%")
        
        # Milestone achievement
        if accuracy >= 97:
            print("🎯 MILESTONE 2 TARGET ACHIEVED: ≥97% accuracy")
        elif accuracy >= 95:
            print("✅ GOOD PERFORMANCE: ≥95% accuracy")
        elif accuracy >= 90:
            print("⚠️  ACCEPTABLE PERFORMANCE: ≥90% accuracy")
        else:
            print("❌ NEEDS IMPROVEMENT: <90% accuracy")
        
        # Class-wise performance
        print("\nClass-wise Accuracy:")
        print("-" * 30)
        for class_name in self.class_names:
            class_idx = self.test_dataset.class_to_idx[class_name]
            class_mask = np.array(self.test_dataset.labels) == class_idx
            if class_mask.sum() > 0:
                # Convert numpy types to Python native types for JSON serialization
                class_acc = float((np.array(self.test_dataset.labels)[class_mask] == 
                            np.array(self.test_dataset.labels)[class_mask]).mean() * 100)
                print(f"{class_name:20}: {class_acc:.2f}%")
        
        print("="*60)
        
        # Save final report with proper type conversion
        final_report = {
            'model': 'EfficientNet-B4',
            'test_samples': int(total_samples),  # Convert to native Python int
            'overall_accuracy': float(accuracy),  # Convert to native Python float
            'error_rate': float(100 - accuracy),
            'milestone_achieved': accuracy >= 97,
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'class_distribution': {k: int(v) for k, v in self.test_dataset.get_class_distribution().items()},  # Convert to native types
            'detailed_metrics': report
        }
        
        with open('final_evaluation_report.json', 'w') as f:
            json.dump(final_report, f, indent=2, default=str)  # Added default=str for any remaining non-serializable objects
        
        print("💾 Reports saved:")
        print("   - evaluation_report.json")
        print("   - classification_metrics.csv") 
        print("   - final_evaluation_report.json")
        print("   - confusion_matrix.png")
        print("   - confidence_analysis.png")
        print("   - error_analysis.png")

def main():
    evaluator = ModelEvaluator()
    accuracy, predictions, targets, probabilities = evaluator.evaluate_model()
    
    # Load the classification report for final report
    with open('evaluation_report.json', 'r') as f:
        report = json.load(f)
    
    evaluator.generate_final_report(accuracy, report)

if __name__ == "__main__":
    main()