"""
Run predictions on test images
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

from src.model.inference import DefectPredictor, compare_with_ground_truth
from src.model.evaluator import load_best_model
from src.utils.file_operations import load_config, create_directory


def predict_single_pair(predictor: DefectPredictor,
                        template_path: str,
                        test_path: str,
                        label_path: str = None,
                        output_dir: str = None):
    """
    Predict on a single image pair
    """
    image_id = Path(template_path).stem.replace('_temp', '')
    
    print(f"\nProcessing: {image_id}")
    
    # Run prediction
    result = predictor.predict(template_path, test_path)
    
    if not result['success']:
        print(f"  ✗ Failed: {result['error']}")
        return None
    
    print(f"  ✓ Detected {result['num_defects']} defects")
    
    # Print classifications
    for defect in result['classifications']:
        print(f"    - {defect['predicted_label']}: {defect['confidence']:.2f}")
    
    # Compare with ground truth if available
    comparison = None
    if label_path and os.path.exists(label_path):
        config = load_config()
        comparison = compare_with_ground_truth(
            result['classifications'], 
            label_path,
            config
        )
        
        if comparison['has_ground_truth']:
            print(f"  Ground truth comparison:")
            print(f"    Detection rate: {comparison['detection_rate']*100:.1f}%")
            print(f"    Classification accuracy: {comparison['classification_accuracy']*100:.1f}%")
    
    # Create visualization
    if output_dir:
        visualize_prediction(result, image_id, output_dir)
    
    return {
        'image_id': image_id,
        'num_defects': result['num_defects'],
        'classifications': result['classifications'],
        'comparison': comparison
    }


def visualize_prediction(result: dict, image_id: str, output_dir: str):
    """
    Create visualization of prediction
    """
    predictor = result.get('predictor')
    
    # Create annotated image
    annotated = predictor.annotate_image(
        result['test'], 
        result['classifications']
    ) if 'predictor' in result else None
    
    # If predictor not in result, create annotation manually
    if annotated is None:
        test_img = result['test']
        if len(test_img.shape) == 2:
            annotated = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        else:
            annotated = test_img.copy()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                 (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for defect in result['classifications']:
            x, y, w, h = defect['bbox']
            pred_class = defect['predicted_class']
            color = colors[pred_class % len(colors)]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            text = f"{defect['predicted_label']}: {defect['confidence']:.2f}"
            cv2.putText(annotated, text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # Template
    plt.subplot(2, 3, 1)
    plt.imshow(result['template'], cmap='gray')
    plt.title('Template', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Test image
    plt.subplot(2, 3, 2)
    plt.imshow(result['test'], cmap='gray')
    plt.title('Test Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Aligned
    plt.subplot(2, 3, 3)
    plt.imshow(result['aligned'], cmap='gray')
    plt.title(f"Aligned\n(Matches: {result['align_info']['num_matches']})", 
             fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Difference map
    plt.subplot(2, 3, 4)
    plt.imshow(result['diff_map'], cmap='hot')
    plt.title('Difference Map', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Mask
    plt.subplot(2, 3, 5)
    plt.imshow(result['mask'], cmap='gray')
    plt.title(f'Defect Mask\n({result["num_defects"]} defects)', 
             fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Annotated result
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title('Classified Defects', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.suptitle(f'Defect Detection & Classification: {image_id}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{image_id}_prediction.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """
    Run predictions on test set
    """
    print("="*60)
    print("DEFECT DETECTION & CLASSIFICATION PREDICTION")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading trained model...")
    model = load_best_model(config, device)
    
    # Create predictor
    predictor = DefectPredictor(model, config, device)
    
    # Setup paths
    splits_path = config['data']['splits_path']
    test_templates = os.path.join(splits_path, 'test', 'templates')
    test_images = os.path.join(splits_path, 'test', 'test_images')
    test_labels = os.path.join(splits_path, 'test', 'labels')
    
    output_dir = os.path.join(config['paths']['results'], 'predictions')
    create_directory(output_dir)
    
    # Get test images (first 10 for demo)
    template_files = sorted(list(Path(test_templates).glob('*_temp.jpg')))[:10]
    
    print(f"\nProcessing {len(template_files)} test images...")
    
    # Run predictions
    all_results = []
    
    for template_file in template_files:
        image_id = template_file.stem.replace('_temp', '')
        test_file = os.path.join(test_images, f'{image_id}_test.jpg')
        label_file = os.path.join(test_labels, f'{image_id}.txt')
        
        if os.path.exists(test_file):
            result = predict_single_pair(
                predictor, 
                str(template_file),
                test_file,
                label_file,
                output_dir
            )
            
            if result:
                all_results.append(result)
    
    # Summary statistics
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    total_defects = sum(r['num_defects'] for r in all_results)
    print(f"\nTotal images processed: {len(all_results)}")
    print(f"Total defects detected: {total_defects}")
    print(f"Average defects per image: {total_defects/len(all_results):.2f}")
    
    # Ground truth comparison
    comparisons = [r['comparison'] for r in all_results 
                  if r['comparison'] and r['comparison']['has_ground_truth']]
    
    if comparisons:
        avg_detection = np.mean([c['detection_rate'] for c in comparisons])
        avg_classification = np.mean([c['classification_accuracy'] for c in comparisons])
        
        print(f"\nGround Truth Comparison:")
        print(f"  Average detection rate: {avg_detection*100:.1f}%")
        print(f"  Average classification accuracy: {avg_classification*100:.1f}%")
    
    # Save results
    results_path = os.path.join(output_dir, 'prediction_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"Visualizations: {output_dir}/*.png")


if __name__ == "__main__":
    main()