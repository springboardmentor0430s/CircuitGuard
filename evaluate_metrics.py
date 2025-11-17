"""
Evaluate alignment and detection quality using metrics
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_preparation.preprocessing import ImagePreprocessor
from src.data_preparation.image_alignment import ImageAligner
from src.data_preparation.image_subtraction import ImageSubtractor
from src.defect_detection.thresholding import DefectThresholder
from src.defect_detection.morphological_ops import MorphologicalProcessor
from src.utils.metrics import AlignmentMetrics, DefectDetectionMetrics, ImageQualityMetrics
from src.utils.file_operations import load_config


def evaluate_single_pair(template_path, test_path):
    """
    Evaluate metrics for a single image pair
    """
    # Load config
    config = load_config()
    
    # Initialize components
    preprocessor = ImagePreprocessor(grayscale=True, normalize=False)
    aligner = ImageAligner(method='orb')
    subtractor = ImageSubtractor()
    thresholder = DefectThresholder()
    
    # Process
    template = preprocessor.load_image(template_path)
    test_img = preprocessor.load_image(test_path)
    
    # Evaluate original image quality
    template_quality = ImageQualityMetrics.evaluate_quality(template)
    test_quality = ImageQualityMetrics.evaluate_quality(test_img)
    
    # Align and evaluate alignment
    try:
        aligned, align_info = aligner.align_image(template, test_img)
        alignment_metrics = AlignmentMetrics.evaluate_alignment(template, aligned)
    except Exception as e:
        print(f"Alignment failed: {e}")
        return None
    
    # Compute difference
    diff_map = subtractor.compute_difference_map(template, aligned)
    
    # Threshold
    binary_mask, _ = thresholder.threshold_difference_map(diff_map)
    
    return {
        'image_id': Path(template_path).stem,
        'template_quality': template_quality,
        'test_quality': test_quality,
        'alignment': alignment_metrics,
        'align_info': align_info
    }


def main():
    """
    Evaluate metrics on sample dataset
    """
    print("="*60)
    print("EVALUATING ALIGNMENT AND DETECTION METRICS")
    print("="*60)
    
    # Get sample images
    config = load_config()
    splits_path = config['data']['splits_path']
    
    train_templates = os.path.join(splits_path, 'train', 'templates')
    train_tests = os.path.join(splits_path, 'train', 'test_images')
    
    # Evaluate 10 samples
    template_files = sorted(os.listdir(train_templates))[:10]
    
    results = []
    for template_file in template_files:
        image_id = template_file.replace('_temp.jpg', '')
        test_file = f"{image_id}_test.jpg"
        
        template_path = os.path.join(train_templates, template_file)
        test_path = os.path.join(train_tests, test_file)
        
        if os.path.exists(template_path) and os.path.exists(test_path):
            print(f"\nEvaluating: {image_id}")
            result = evaluate_single_pair(template_path, test_path)
            if result:
                results.append(result)
                
                # Print metrics
                print(f"\n  Template Quality:")
                for k, v in result['template_quality'].items():
                    print(f"    {k}: {v:.4f}")
                
                print(f"\n  Alignment Quality:")
                for k, v in result['alignment'].items():
                    print(f"    {k}: {v:.4f}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if results:
        avg_mse = np.mean([r['alignment']['mse'] for r in results])
        avg_psnr = np.mean([r['alignment']['psnr'] for r in results])
        avg_ssim = np.mean([r['alignment']['ssim'] for r in results])
        avg_ncc = np.mean([r['alignment']['ncc'] for r in results])
        
        print(f"\nAverage Alignment Metrics (n={len(results)}):")
        print(f"  MSE:  {avg_mse:.4f}")
        print(f"  PSNR: {avg_psnr:.2f} dB")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  NCC:  {avg_ncc:.4f}")
        
        print(f"\nAlignment Success:")
        avg_matches = np.mean([r['align_info']['num_matches'] for r in results])
        avg_inliers = np.mean([r['align_info']['num_inliers'] for r in results])
        print(f"  Avg matches:  {avg_matches:.1f}")
        print(f"  Avg inliers:  {avg_inliers:.1f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()