import cv2
import os
import argparse
import glob
import numpy as np
from src.preprocessing.alignment import align_images, simple_alignment
from src.preprocessing.binary_defect_detection import xor_defect_detection, highlight_xor_defects
from src.preprocessing.contour_detection import detect_contours, draw_contours_and_boxes
from src.utils.visualization import display_results

def load_specific_pair(template_dir, test_dir, image_name):
    """Load matched images"""
    image_name = os.path.splitext(image_name)[0]
    template_patterns = [
        os.path.join(template_dir, f"{image_name}_temp.*"),
        os.path.join(template_dir, f"*{image_name}*_temp.*")
    ]
    template_path = None
    for pattern in template_patterns:
        matches = glob.glob(pattern)
        if matches:
            template_path = matches[0]
            break
    test_patterns = [
        os.path.join(test_dir, f"{image_name}_test.*"),
        os.path.join(test_dir, f"*{image_name}*_test.*")
    ]
    test_path = None
    for pattern in test_patterns:
        matches = glob.glob(pattern)
        if matches:
            test_path = matches[0]
            break
    if template_path and test_path:
        print(f"Found pair:")
        print(f"  Template: {os.path.basename(template_path)}")
        print(f"  Test: {os.path.basename(test_path)}")
        return [(template_path, test_path, os.path.basename(template_path))]
    else:
        if not template_path:
            print(f"Template file not found for: {image_name}")
        if not test_path:
            print(f"Test file not found for: {image_name}")
        return []

def load_all_pairs(template_dir, test_dir):
    """Batch load pairs"""
    image_pairs = []
    template_files = glob.glob(os.path.join(template_dir, "*_temp.jpg"))
    template_files.extend(glob.glob(os.path.join(template_dir, "*_temp.png")))
    template_files.extend(glob.glob(os.path.join(template_dir, "*_temp.jpeg")))
    print(f"Found {len(template_files)} template files")
    for template_path in template_files:
        template_filename = os.path.basename(template_path)
        base_name = template_filename.replace('_temp.', '_test.')
        test_path = os.path.join(test_dir, base_name)
        if not os.path.exists(test_path):
            test_path = test_path.replace('.jpg', '.png')
        if not os.path.exists(test_path):
            test_path = test_path.replace('.png', '.jpeg')
        if os.path.exists(test_path):
            image_pairs.append((template_path, test_path, template_filename))
    
    return image_pairs

def process_single_pair(template_path, test_path, use_xor=True):
    """Process image pair"""
    template_img = cv2.imread(template_path)
    test_img = cv2.imread(test_path)
    
    if template_img is None:
        print(f"Error: Could not load template image {template_path}")
        return None
    if test_img is None:
        print(f"Error: Could not load test image {test_path}")
        return None
    
    print(f"Processing: {os.path.basename(template_path)}")
    print(f"Template shape: {template_img.shape}, Test shape: {test_img.shape}")
    
    print("Aligning images...")
    aligned_test, _ = align_images(test_img, template_img)
    if aligned_test is None or aligned_test.shape != template_img.shape:
        print("Feature-based alignment failed, using simple alignment...")
        aligned_test, _ = simple_alignment(test_img, template_img)
    
    if use_xor:
        print("Using XOR-based defect detection...")
        defects_dict = xor_defect_detection(aligned_test, template_img)
        defect_mask = defects_dict['combined']
        result_img = highlight_xor_defects(aligned_test, defects_dict)
    else:
        from preprocessing.subtraction import image_subtraction, highlight_defects
        diff, thresh, defect_mask = image_subtraction(aligned_test, template_img)
        result_img = highlight_defects(aligned_test, defect_mask)
    print("Detecting contours...")
    contours, bounding_boxes = detect_contours(defect_mask)
    contours_img = draw_contours_and_boxes(aligned_test, contours, bounding_boxes)
    
    print(f"Detected {len(contours)} defects")
    
    return {
        'template': template_img,
        'test': aligned_test,
        'defect_mask': defect_mask,
        'result': result_img,
        'contours_img': contours_img,
        'contours': contours,
        'bounding_boxes': bounding_boxes
    }

def main():
    parser = argparse.ArgumentParser(description='PCB Defect Detection - XOR Method')
    parser.add_argument('--template_dir', type=str, default='data/interim/template', help='Template dir')
    parser.add_argument('--test_dir', type=str, default='data/interim/test', help='Test dir')
    parser.add_argument('--image_name', type=str, default=None, help='Base image name')
    parser.add_argument('--all', action='store_true', help='Process all pairs')
    parser.add_argument('--traditional', action='store_true', help='Use subtraction instead of XOR')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.template_dir):
        print(f"Template directory does not exist: {args.template_dir}")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"Test directory does not exist: {args.test_dir}")
        return
    
    print(f"Template directory: {args.template_dir}")
    print(f"Test directory: {args.test_dir}")
    if args.image_name:
        image_pairs = load_specific_pair(args.template_dir, args.test_dir, args.image_name)
    elif args.all:
        image_pairs = load_all_pairs(args.template_dir, args.test_dir)
    else:
        print("Please specify either --image_name or --all flag")
        return
    
    if not image_pairs:
        print("No matching image pairs found!")
        return
    
    print(f"Processing {len(image_pairs)} image pair(s)")
    for i, (template_path, test_path, filename) in enumerate(image_pairs):
        print(f"\n--- Processing pair {i+1}/{len(image_pairs)}: {filename} ---")
        results = process_single_pair(template_path, test_path, use_xor=not args.traditional)
        if results:
            display_results(
                results['template'],
                results['test'],
                results['result'],
                results['contours_img'],
                window_name=f"PCB Defect Detection - {filename}"
            )

if __name__ == "__main__":
    main()