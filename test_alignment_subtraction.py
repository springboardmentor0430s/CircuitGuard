"""
Test script for image alignment and subtraction
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_preparation.preprocessing import ImagePreprocessor
from src.data_preparation.image_alignment import ImageAligner
from src.data_preparation.image_subtraction import ImageSubtractor
from src.defect_detection.thresholding import DefectThresholder
from src.utils.file_operations import load_config, create_directory


def test_single_pair(template_path, test_path, output_dir):
    """
    Test alignment and subtraction on a single image pair
    """
    print(f"\nProcessing:")
    print(f"  Template: {os.path.basename(template_path)}")
    print(f"  Test: {os.path.basename(test_path)}")
    
    # Initialize components
    preprocessor = ImagePreprocessor(grayscale=True, normalize=False)
    aligner = ImageAligner(method='orb', max_features=5000)
    subtractor = ImageSubtractor(method='absolute')
    thresholder = DefectThresholder(method='otsu')
    
    # Load images
    print("  Loading images...")
    template = preprocessor.load_image(template_path)
    test_img = preprocessor.load_image(test_path)
    
    print(f"  Template shape: {template.shape}")
    print(f"  Test shape: {test_img.shape}")
    
    # Align images
    print("  Aligning images...")
    try:
        aligned, align_info = aligner.align_image(template, test_img)
        print(f"  Matches: {align_info['num_matches']}, Inliers: {align_info['num_inliers']}")
    except Exception as e:
        print(f"  Error during alignment: {e}")
        return None
    
    # Compute difference map
    print("  Computing difference map...")
    diff_map = subtractor.compute_difference_map(template, aligned, preprocess=True)
    
    # Apply thresholding
    print("  Applying Otsu thresholding...")
    binary_mask, threshold_info = thresholder.threshold_difference_map(diff_map)
    print(f"  Threshold value: {threshold_info.get('threshold_value', 'N/A')}")
    
    # Clean mask
    clean_mask = thresholder.post_process_mask(binary_mask, min_area=50)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(template, cmap='gray')
    axes[0, 0].set_title('Template (Reference)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(test_img, cmap='gray')
    axes[0, 1].set_title('Test Image (Original)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(aligned, cmap='gray')
    axes[0, 2].set_title('Test Image (Aligned)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(diff_map, cmap='hot')
    axes[1, 0].set_title('Difference Map')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(binary_mask, cmap='gray')
    axes[1, 1].set_title(f"Binary Mask (Otsu: {threshold_info.get('threshold_value', 'N/A')})")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(clean_mask, cmap='gray')
    axes[1, 2].set_title('Cleaned Mask')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{Path(template_path).stem}_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved result to: {output_path}")
    plt.close()
    
    return {
        'template': template,
        'test': test_img,
        'aligned': aligned,
        'diff_map': diff_map,
        'binary_mask': binary_mask,
        'clean_mask': clean_mask
    }


def main():
    """
    Test on sample image pairs
    """
    print("=" * 60)
    print("TESTING ALIGNMENT AND SUBTRACTION")
    print("=" * 60)
    
    # Setup paths
    config = load_config()
    splits_path = config['data']['splits_path']
    output_dir = "outputs/visualizations/alignment_test"
    create_directory(output_dir)
    
    # Get sample pairs from train split
    train_templates = os.path.join(splits_path, 'train', 'templates')
    train_tests = os.path.join(splits_path, 'train', 'test_images')
    
    # Get first 3 pairs
    template_files = sorted(os.listdir(train_templates))[:3]
    
    for template_file in template_files:
        image_id = template_file.replace('_temp.jpg', '')
        test_file = f"{image_id}_test.jpg"
        
        template_path = os.path.join(train_templates, template_file)
        test_path = os.path.join(train_tests, test_file)
        
        if os.path.exists(template_path) and os.path.exists(test_path):
            result = test_single_pair(template_path, test_path, output_dir)
        else:
            print(f"\nSkipping {image_id}: files not found")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()