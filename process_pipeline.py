"""
Complete processing pipeline for a single image pair
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
from src.defect_detection.morphological_ops import MorphologicalProcessor
from src.defect_detection.contour_extraction import ContourExtractor
from src.utils.file_operations import load_config, create_directory


def process_image_pair(template_path, test_path, output_dir, save_intermediate=True):
    """
    Complete processing pipeline for one image pair
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(template_path).stem}")
    print(f"{'='*60}")
    
    # Initialize all components
    config = load_config()
    preprocessor = ImagePreprocessor(grayscale=True, normalize=False)
    aligner = ImageAligner(
        method=config['alignment']['method'],
        max_features=config['alignment']['max_features']
    )
    subtractor = ImageSubtractor(
        method=config['subtraction']['method'],
        blur_kernel=tuple(config['subtraction']['blur_kernel'])
    )
    thresholder = DefectThresholder(method=config['thresholding']['method'])
    morph_processor = MorphologicalProcessor(
        erosion_kernel=tuple(config['morphology']['erosion_kernel']),
        dilation_kernel=tuple(config['morphology']['dilation_kernel'])
    )
    contour_extractor = ContourExtractor(
        min_area=config['contours']['min_area'],
        max_area=config['contours']['max_area']
    )
    
    # Step 1: Load images
    print("Step 1: Loading images...")
    template = preprocessor.load_image(template_path)
    test_img = preprocessor.load_image(test_path)
    
    # Step 2: Align images
    print("Step 2: Aligning images...")
    try:
        aligned, align_info = aligner.align_image(template, test_img)
        print(f"  Matches: {align_info['num_matches']}, Inliers: {align_info['num_inliers']}")
    except Exception as e:
        print(f"  Alignment failed: {e}")
        return None
    
    # Step 3: Compute difference
    print("Step 3: Computing difference map...")
    diff_map = subtractor.compute_difference_map(template, aligned)
    
    # Step 4: Threshold
    print("Step 4: Applying thresholding...")
    binary_mask, threshold_info = thresholder.threshold_difference_map(diff_map)
    print(f"  Threshold: {threshold_info.get('threshold_value', 'adaptive')}")
    
    # Step 5: Refine mask
    print("Step 5: Refining mask with morphological operations...")
    refined_mask = morph_processor.refine_mask(binary_mask, min_area=config['contours']['min_area'])
    
    # Step 6: Extract contours
    print("Step 6: Extracting contours...")
    contours = contour_extractor.extract_contours(refined_mask)
    properties = contour_extractor.get_contour_properties(contours)
    print(f"  Found {len(contours)} defects")
    
    # Step 7: Extract ROIs
    print("Step 7: Extracting defect ROIs...")
    rois = contour_extractor.extract_rois(test_img, contours, padding=10)
    
    # Step 8: Annotate image
    annotated = contour_extractor.annotate_defects(test_img, properties)
    
    # Save intermediate results if requested
    if save_intermediate and output_dir:
        image_id = Path(template_path).stem.replace('_temp', '')
        
        # Save aligned pair
        cv2.imwrite(os.path.join(output_dir, 'aligned_pairs', f"{image_id}_aligned.jpg"), aligned)
        
        # Save difference map
        cv2.imwrite(os.path.join(output_dir, 'difference_maps', f"{image_id}_diff.jpg"), diff_map)
        
        # Save thresholded mask
        cv2.imwrite(os.path.join(output_dir, 'thresholded', f"{image_id}_thresh.jpg"), refined_mask)
        
        # Save annotated image
        cv2.imwrite(os.path.join(output_dir, 'annotated', f"{image_id}_annotated.jpg"), annotated)
        
        # Save ROIs
        roi_dir = os.path.join(output_dir, 'defect_rois', image_id)
        create_directory(roi_dir)
        for i, roi in enumerate(rois):
            cv2.imwrite(os.path.join(roi_dir, f"roi_{i:03d}.jpg"), roi)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Original images
    plt.subplot(3, 4, 1)
    plt.imshow(template, cmap='gray')
    plt.title('Template (Reference)', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(test_img, cmap='gray')
    plt.title('Test Image', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(aligned, cmap='gray')
    plt.title(f'Aligned\n(Matches: {align_info["num_matches"]})', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(diff_map, cmap='hot')
    plt.title('Difference Map', fontsize=10)
    plt.axis('off')
    
    # Row 2: Thresholding and morphology
    plt.subplot(3, 4, 5)
    plt.imshow(binary_mask, cmap='gray')
    plt.title(f'Binary Mask\n(Threshold: {threshold_info.get("threshold_value", "N/A")})', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(refined_mask, cmap='gray')
    plt.title('Refined Mask', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    contour_img = contour_extractor.draw_contours_and_boxes(test_img, contours)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Contours\n({len(contours)} defects)', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title('Annotated Result', fontsize=10)
    plt.axis('off')
    
    # Row 3: Sample ROIs
    for i in range(min(4, len(rois))):
        plt.subplot(3, 4, 9 + i)
        plt.imshow(rois[i], cmap='gray')
        plt.title(f'ROI {i}\nArea: {int(properties[i]["area"])}', fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    if output_dir:
        vis_path = os.path.join(output_dir, 'visualizations', f"{Path(template_path).stem}_pipeline.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization: {vis_path}")
    
    plt.close()
    
    return {
        'properties': properties,
        'num_defects': len(contours),
        'align_info': align_info
    }


def main():
    """
    Run pipeline on sample images
    """
    print("="*60)
    print("COMPLETE DEFECT DETECTION PIPELINE")
    print("="*60)
    
    # Setup
    config = load_config()
    splits_path = config['data']['splits_path']
    output_dir = "outputs/pipeline_results"
    
    # Create output directories
    for subdir in ['aligned_pairs', 'difference_maps', 'thresholded', 'annotated', 'defect_rois', 'visualizations']:
        create_directory(os.path.join(output_dir, subdir))
    
    # Process 5 sample pairs
    train_templates = os.path.join(splits_path, 'train', 'templates')
    train_tests = os.path.join(splits_path, 'train', 'test_images')
    
    template_files = sorted(os.listdir(train_templates))[:5]
    
    results = []
    for template_file in template_files:
        image_id = template_file.replace('_temp.jpg', '')
        test_file = f"{image_id}_test.jpg"
        
        template_path = os.path.join(train_templates, template_file)
        test_path = os.path.join(train_tests, test_file)
        
        if os.path.exists(template_path) and os.path.exists(test_path):
            result = process_image_pair(template_path, test_path, output_dir)
            if result:
                results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nProcessed {len(results)} image pairs")
    print(f"Total defects detected: {sum(r['num_defects'] for r in results)}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()