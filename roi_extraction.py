import cv2
import numpy as np
import os
import glob
from collections import defaultdict


# === CONFIGURATION ===
TARGET_GROUP_PATH = r'C:\Users\DELL\OneDrive\Desktop\New folder\PCBData\group92000'
IDEAL_KERNEL_SIZE = 5
OUTPUT_ROOT_DIR = f'extracted_rois_group_{os.path.basename(TARGET_GROUP_PATH)}'
LEFT_CROP_WIDTH = 40  # Width in pixels to ignore on left side


# === PARAMETERS ===
MAX_ASPECT_RATIO = 10.0        # Max allowed w/h ratio
IOU_THRESHOLD = 0.4            # Minimum IoU for match
BOX_PADDING = 15               # Padding pixels for ROI
MIN_DEFECT_AREA = 8            # Minimum contour area
DETECTED_BOX_X_OFFSET = 3      # Left shift pixels


# === DEFECT CLASSES ===
DEFECT_CLASSES = {
    1: 'open',
    2: 'short', 
    3: 'mousebite',
    4: 'spur',
    5: 'copper',
    6: 'pin-hole',
    7: 'noise'  # Noise class
}


def calculate_iou(boxA, boxB):
    """Calculate IoU between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    return 0.0 if unionArea == 0 else interArea / unionArea


def check_iou_and_assign_label(detected_bbox, gt_labels, threshold=IOU_THRESHOLD):
    """Match detected box with ground truth and return label."""
    best_iou = 0.0
    best_match_label_name = DEFECT_CLASSES[7]  # Default to noise

    for gt in gt_labels:
        gt_box = gt['bbox']
        iou = calculate_iou(detected_bbox, gt_box)
        
        if iou > best_iou and iou >= threshold:
            best_iou = iou
            best_match_label_name = gt['label_name']

    return best_match_label_name


def load_ground_truth_labels(label_path, stats):
    """Load and parse ground truth labels."""
    gt_labels = []
    if not os.path.exists(label_path):
        return gt_labels
    
    with open(label_path, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split()
                if len(parts) == 5:
                    xmin, ymin, xmax, ymax, class_id = map(int, parts)
                    
                    # Skip boxes that start in the neglected left region
                    if xmin < LEFT_CROP_WIDTH:
                        stats['filter_boundary'] += 1
                        continue

                    # Adjust coordinates relative to cropped region
                    xmin_adj = xmin - LEFT_CROP_WIDTH
                    xmax_adj = xmax - LEFT_CROP_WIDTH
                    
                    label_name = DEFECT_CLASSES.get(class_id, DEFECT_CLASSES[7])
                    
                    gt_labels.append({
                        'bbox': (xmin_adj, ymin, xmax_adj, ymax),
                        'label_name': label_name
                    })
            except Exception as e:
                print(f"Error parsing line in {label_path}: {e}")
    
    return gt_labels


def preprocess_and_highlight_defects(template_path, test_path, kernel_dim):
    """Generate defect mask from images."""
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    
    if template is None or test is None:
        return None, "Error loading images"

    # Crop left side of both images
    template = template[:, LEFT_CROP_WIDTH:]
    test = test[:, LEFT_CROP_WIDTH:]

    # Align images
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)

    try:
        _, warp_matrix = cv2.findTransformECC(template, test, warp_matrix, warp_mode, criteria)
        aligned_test = cv2.warpAffine(
            test,
            warp_matrix,
            (template.shape[1], template.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    except cv2.error:
        aligned_test = test

    # Process images
    difference = cv2.absdiff(template, aligned_test)
    blurred_diff = cv2.GaussianBlur(difference, (3, 3), 0)
    _, defect_mask = cv2.threshold(blurred_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return defect_mask, None


def extract_and_save_rois(defect_mask, original_image_path, set_id, gt_labels, output_root_dir, stats):
    """Extract and save ROIs."""
    if defect_mask is None:
        return 0

    # Load and crop original image
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"Error: Could not load original image {original_image_path}")
        return 0
    
    # Crop left side of original image
    original_image = original_image[:, LEFT_CROP_WIDTH:]
    H, W, _ = original_image.shape
    
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_count = 0
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # Apply filters
        if area < MIN_DEFECT_AREA:
            stats['filter_area'] += 1
            continue
        if w > 0 and h > 0 and max(w/h, h/w) > MAX_ASPECT_RATIO:
            stats['filter_aspect'] += 1
            continue

        # Calculate coordinates
        x_start = max(0, x - BOX_PADDING - DETECTED_BOX_X_OFFSET)
        y_start = max(0, y - BOX_PADDING)
        x_end = min(W, x + w + BOX_PADDING - DETECTED_BOX_X_OFFSET)
        y_end = min(H, y + h + BOX_PADDING)

        # Ensure minimum width
        if x_end <= x_start:
            x_end = x_start + 1

        # Process detection
        detected_box = (x_start, y_start, x_end, y_end)
        label_name = check_iou_and_assign_label(detected_box, gt_labels)
        
        # Update statistics
        is_true_positive = (label_name != DEFECT_CLASSES[7])
        if is_true_positive:
            stats['true_positives'] += 1
        else:
            stats['false_positives'] += 1

        # Save ROI
        cropped_roi = original_image[y_start:y_end, x_start:x_end]
        target_dir = os.path.join(output_root_dir, label_name)
        os.makedirs(target_dir, exist_ok=True)
        
        filename = f"{set_id}_{i}.jpg"
        output_path = os.path.join(target_dir, filename)
        cv2.imwrite(output_path, cropped_roi)
        
        stats['extracted_by_class'][label_name] += 1
        roi_count += 1
        
    return roi_count


def find_image_sets_in_group(group_path):
    """Find image sets in group."""
    image_folders = glob.glob(os.path.join(group_path, '*', '*_temp.jpg'))
    
    for template_path in image_folders:
        file_base_name = os.path.basename(template_path).replace('_temp.jpg', '')
        test_path = template_path.replace('_temp.jpg', '_test.jpg')
        
        inner_dir = os.path.basename(os.path.dirname(template_path))
        label_dir = os.path.join(group_path, inner_dir + '_not')
        label_path = os.path.join(label_dir, file_base_name + '.txt')

        if os.path.exists(test_path) and os.path.exists(label_path):
            yield template_path, test_path, label_path, file_base_name


def main_pipeline():
    """Main execution pipeline."""
    print("="*60)
    print(f"Starting ROI Extraction for Group: {os.path.basename(TARGET_GROUP_PATH)}")
    print(f"Using Kernel Size: {IDEAL_KERNEL_SIZE}x{IDEAL_KERNEL_SIZE}")
    print(f"Left Crop Width: {LEFT_CROP_WIDTH} pixels")
    print(f"X-Axis Shift: {DETECTED_BOX_X_OFFSET} pixels")
    print("="*60)

    # Initialize statistics
    total_images_processed = 0
    stats = defaultdict(int, extracted_by_class={name: 0 for name in DEFECT_CLASSES.values()})
    stats['true_positives'] = 0
    stats['false_positives'] = 0
    stats['filter_boundary'] = 0

    # Process images
    for template_path, test_path, label_path, set_id in find_image_sets_in_group(TARGET_GROUP_PATH):
        total_images_processed += 1
        
        ground_truth_labels = load_ground_truth_labels(label_path, stats)
        if not ground_truth_labels:
            stats['skipped_no_gt'] += 1
            continue

        defect_mask, error = preprocess_and_highlight_defects(
            template_path, test_path, IDEAL_KERNEL_SIZE
        )
        if error:
            print(f"Error processing {set_id}: {error}")
            stats['skipped_error'] += 1
            continue

        roi_count = extract_and_save_rois(
            defect_mask,
            test_path,
            set_id,
            ground_truth_labels,
            OUTPUT_ROOT_DIR,
            stats
        )
        print(f"-> Processed {set_id}: {roi_count} ROIs extracted")

    # Print report
    total_extracted = stats['true_positives'] + stats['false_positives']
    print("\n" + "="*60)
    print("ROI EXTRACTION SUMMARY")
    print("="*60)
    print(f"Target Group: {os.path.basename(TARGET_GROUP_PATH)}")
    print(f"Images Processed: {total_images_processed}")
    print(f"Total ROIs Extracted: {total_extracted}")
    print(f"Left Side Ignored: First {LEFT_CROP_WIDTH} pixels")
    print("-"*30)
    print(f"True Positives: {stats['true_positives']}")
    print(f"False Positives: {stats['false_positives']}")
    print(f"Boundary Filtered: {stats['filter_boundary']}")
    print("-"*30)
    print("Breakdown by Class:")
    
    for cls, count in sorted(stats['extracted_by_class'].items()):
        if count > 0 and cls != DEFECT_CLASSES[7]:
            print(f"  - {cls.capitalize()}: {count}")
    print(f"  - Noise: {stats['false_positives']}")
    print("\nProcessing complete.")


if __name__ == '__main__':
    main_pipeline()