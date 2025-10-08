import cv2
import numpy as np

def detect_contours(defect_mask, min_area=10):
    """
    Detect contours in the defect mask and extract bounding boxes
    """
    # Find contours
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            filtered_contours.append(contour)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
    
    return filtered_contours, bounding_boxes

def draw_contours_and_boxes(image, contours, bounding_boxes):
    """
    Draw only bounding boxes on the image (no green contours)
    """
    if len(image.shape) == 2:
        result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result_img = image.copy()
    
    # Skip drawing green contours - only draw blue bounding boxes
    # cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)  # REMOVED
    
    # Draw bounding boxes in blue
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue boxes
    
    return result_img

def extract_defect_regions(test_img, bounding_boxes, margin=5):
    """
    Extract defect regions as ROIs
    """
    defect_regions = []
    
    for (x, y, w, h) in bounding_boxes:
        # Add margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(test_img.shape[1], x + w + margin)
        y2 = min(test_img.shape[0], y + h + margin)
        
        # Extract ROI
        roi = test_img[y1:y2, x1:x2]
        defect_regions.append(roi)
    
    return defect_regions