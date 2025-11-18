import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


#CONFIGURATION
TEMPLATE_IMAGE_PATH = r'C:\Users\DELL\OneDrive\Desktop\New folder\PCBData\group00041\00041\00041003_temp.jpg'
TEST_IMAGE_PATH = r'C:\Users\DELL\OneDrive\Desktop\New folder\PCBData\group00041\00041\00041003_test.jpg'
OUTPUT_MASK_PATH = 'defect_difference_mask.jpg'
OUTPUT_ROI_VIS_PATH = 'defect_roi_visualization.jpg'
ALIGNED_TEST_PATH = 'aligned_test_image.jpg'
LABEL_FILE_PATH = r'C:\Users\DELL\OneDrive\Desktop\New folder\PCBData\group00041\00041_not\00041003.txt'


#PARAMETERS
MAX_ASPECT_RATIO = 10.0        # Max allowed w/h ratio for bounding boxes
IOU_THRESHOLD = 0.4            # Minimum IoU for a match
BOX_PADDING = 15               # Padding pixels for detected boxes
DETECTED_BOX_X_OFFSET = 3      # Pixels to shift detected boxes left


#COLORS (BGR)
GREEN = (0, 255, 0)            # Matched detections
WHITE = (255, 255, 255)        # False Positive box
YELLOW = (0, 255, 255)         # False Positive text
BLUE = (255, 0, 0)             # Ground Truth


#DEFECT CLASSES
DEFECT_CLASSES = {
    1: 'open',
    2: 'short',
    3: 'mousebite',
    4: 'spur',
    5: 'copper',
    6: 'pin-hole',
    7: 'unknown'
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
    """Match detected box with ground truth."""
    best_iou = 0.0
    best_match_label = 'Noise/False Positive'

    for gt in gt_labels:
        gt_box = gt['bbox']
        iou = calculate_iou(detected_bbox, gt_box)
        
        if iou > best_iou and iou >= threshold:
            best_iou = iou
            best_match_label = gt['label_name']

    if best_iou >= threshold:
        return f"{best_match_label} (Match, IoU:{best_iou:.2f})", GREEN, GREEN
    return best_match_label, WHITE, YELLOW


def load_ground_truth_labels(label_path, crop_width):
    """Load and parse ground truth labels."""
    gt_labels = []
    skipped_lines = total_lines = 0
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                total_lines += 1
                try:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        xmin, ymin, xmax, ymax, class_id = map(int, parts)
                        
                        xmin_cropped = max(0, xmin - crop_width)
                        xmax_cropped = max(0, xmax - crop_width)
                        label_name = DEFECT_CLASSES.get(class_id, DEFECT_CLASSES[7])
                        
                        gt_labels.append({
                            'bbox': (xmin_cropped, ymin, xmax_cropped, ymax),
                            'label_name': label_name
                        })
                    else:
                        print(f"Skipped line {total_lines}: Invalid format")
                        skipped_lines += 1
                except Exception as e:
                    print(f"Error on line {total_lines}: {e}")
                    skipped_lines += 1

        print(f"Loaded {len(gt_labels)} labels, skipped {skipped_lines} lines")
    except Exception as e:
        print(f"Error loading labels: {e}")
    
    return gt_labels


def preprocess_and_highlight_defects(template_path, test_path, output_mask_path, aligned_path):
    """Generate defect mask from images."""
    print("Starting defect detection...")

    #Load images
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    
    if template is None or test is None:
        print("Error loading images")
        return None

    #Align images
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
        print("Alignment failed, using original test image")
        aligned_test = test

    #Process images
    difference = cv2.absdiff(template, aligned_test)
    blurred_diff = cv2.GaussianBlur(difference, (3, 3), 0)
    _, defect_mask = cv2.threshold(blurred_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(output_mask_path, defect_mask)
    return defect_mask


def visualize_detections_and_ground_truth(
        defect_mask, 
        original_image_path, 
        output_vis_path,
        gt_labels, 
        min_defect_area=18, 
        max_aspect_ratio=MAX_ASPECT_RATIO
    ):
    """Visualize detected defects and ground truth boxes."""
    if defect_mask is None:
        return []

    #Load and prepare image
    vis_image = cv2.imread(original_image_path)
    if vis_image is None:
        return []
    
    H, W, _ = vis_image.shape
    detected_rois = []

    #Process detected defects
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        #Apply filters
        if area < min_defect_area:
            continue
        if w > 0 and h > 0 and max(w/h, h/w) > max_aspect_ratio:
            continue

        #Calculate box coordinates with padding and offset
        x_padded = max(0, x - BOX_PADDING - DETECTED_BOX_X_OFFSET)
        y_padded = max(0, y - BOX_PADDING)
        x_max_padded = min(W, x + w + BOX_PADDING - DETECTED_BOX_X_OFFSET)
        y_max_padded = min(H, y + h + BOX_PADDING)

        #Process detection
        detected_box = (x_padded, y_padded, x_max_padded, y_max_padded)
        label, box_color, text_color = check_iou_and_assign_label(detected_box, gt_labels)

        #Draw detection
        cv2.rectangle(vis_image, (x_padded, y_padded), (x_max_padded, y_max_padded), box_color, 2)
        cv2.putText(
            vis_image, 
            label, 
            (x_padded, y_padded-5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            text_color, 
            1, 
            cv2.LINE_AA
        )

        detected_rois.append({
            'bbox': (x_padded, y_padded, x_max_padded-x_padded, y_max_padded-y_padded),
            'label': label,
            'area': int(area)
        })

    #Draw ground truth boxes
    for gt in gt_labels:
        xmin, ymin, xmax, ymax = gt['bbox']
        if xmax > xmin and ymax > ymin:
            cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), BLUE, 3)
            cv2.putText(
                vis_image,
                f"{gt['label_name']} (GT)",
                (xmin, ymax+15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BLUE,
                2,
                cv2.LINE_AA
            )

    cv2.imwrite(output_vis_path, vis_image)
    return detected_rois


def main():
    """Main execution pipeline."""
    #Initialize
    CROP_WIDTH = 0
    ground_truth_labels = load_ground_truth_labels(LABEL_FILE_PATH, CROP_WIDTH)
    
    #Process images
    defect_mask = preprocess_and_highlight_defects(
        TEMPLATE_IMAGE_PATH,
        TEST_IMAGE_PATH,
        OUTPUT_MASK_PATH,
        ALIGNED_TEST_PATH
    )

    #Detect and visualize
    if defect_mask is not None:
        detections = visualize_detections_and_ground_truth(
            defect_mask,
            TEST_IMAGE_PATH,
            OUTPUT_ROI_VIS_PATH,
            ground_truth_labels,
            min_defect_area=8,
            max_aspect_ratio=MAX_ASPECT_RATIO
        )
    
    print("\nProcessing complete")
    print(f"Output files:\n- {OUTPUT_MASK_PATH}\n- {OUTPUT_ROI_VIS_PATH}")


if __name__ == '__main__':
    main()