"""
Evaluate trained model on test set
"""

import torch
from src.model.dataset import create_dataloaders
from src.model.evaluator import ModelEvaluator, load_best_model
from src.utils.file_operations import load_config


def main():
    """
    Main evaluation function
    """
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load dataloaders
    print("\nLoading test dataset...")
    _, _, test_loader = create_dataloaders(config)
    
    # Load best model
    print("\nLoading best trained model...")
    model = load_best_model(config, device)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=config['class_names'],
        device=device,
        output_dir=config['paths']['results']
    )
    
    # Evaluate
    results = evaluator.evaluate()
    
    # Generate report
    evaluator.generate_evaluation_report(results)
    
    # Final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    accuracy = results['overall']['accuracy']
    
    if accuracy >= 0.97:
        print(f"✓ Target achieved! Accuracy: {accuracy*100:.2f}% (≥ 97%)")
    else:
        print(f"✗ Target not met. Accuracy: {accuracy*100:.2f}% (target: ≥ 97%)")
    
    print(f"\nPrecision: {results['overall']['precision']:.4f}")
    print(f"Recall:    {results['overall']['recall']:.4f}")
    print(f"F1-Score:  {results['overall']['f1_score']:.4f}")


if __name__ == "__main__":
    main()