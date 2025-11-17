import os
import json
from pathlib import Path
from src.utils.file_operations import load_config


def print_summary():
    """
    Print final project summary
    """
    config = load_config()
    
    print("\n" + "="*80)
    print(" "*20 + "CIRCUITGUARD-PCB PROJECT SUMMARY")
    print("="*80)
    
    # Check what's been completed
    checkpoints_dir = config['paths']['checkpoints']
    results_dir = config['paths']['results']
    
    best_model_exists = os.path.exists(os.path.join(checkpoints_dir, 'best_model.pth'))
    history_exists = os.path.exists(os.path.join(checkpoints_dir, 'training_history.json'))
    eval_exists = os.path.exists(os.path.join(results_dir, 'evaluation_results.json'))
    
    print("\n📋 PROJECT DELIVERABLES:")
    print("-"*80)
    print(f"{'✓' if best_model_exists else '✗'} Trained Model (EfficientNet-B4)")
    print(f"{'✓' if history_exists else '✗'} Training History & Metrics")
    print(f"{'✓' if eval_exists else '✗'} Evaluation Results")
    print("✓ Complete Pipeline Implementation")
    print("✓ Comprehensive Documentation")
    
    # Performance summary
    if eval_exists:
        with open(os.path.join(results_dir, 'evaluation_results.json'), 'r') as f:
            eval_results = json.load(f)
        
        print("\n🎯 MODEL PERFORMANCE:")
        print("-"*80)
        accuracy = eval_results['overall']['accuracy'] * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Precision: {eval_results['overall']['precision']:.4f}")
        print(f"Recall: {eval_results['overall']['recall']:.4f}")
        print(f"F1-Score: {eval_results['overall']['f1_score']:.4f}")
        
        if accuracy >= 97:
            print("\n🎉 TARGET ACHIEVED! Accuracy ≥ 97%")
        else:
            print(f"\n⚠️  Target not met (achieved: {accuracy:.2f}%, target: ≥97%)")
    
    # File locations
    print("\n📂 KEY FILES & DIRECTORIES:")
    print("-"*80)
    print(f"Trained Model: {checkpoints_dir}/best_model.pth")
    print(f"Results: {results_dir}/")
    print(f"TensorBoard Logs: {config['paths']['logs']}/")
    print(f"Configuration: configs/config.yaml")
    
    # Next steps
    print("\n🚀 USAGE:")
    print("-"*80)
    print("1. View training curves:")
    print("   tensorboard --logdir=models/logs")
    print("\n2. Run predictions:")
    print("   python predict_defects.py")
    print("\n3. Generate final report:")
    print("   python visualize_training_results.py")
    
    print("\n" + "="*80)
    print(" "*25 + "PROJECT COMPLETE! 🎊")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_summary()