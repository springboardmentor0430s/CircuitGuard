import cv2
import numpy as np

def detect_contours(defect_mask, min_area=10, max_area=500):
    """Detect and filter defect contours"""
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        if w < 2 or h < 2:
            continue
            
        # Filter by aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 8 or aspect_ratio < 0.125:
            continue
            
        filtered_contours.append(contour)
        bounding_boxes.append((x, y, w, h))
    
    return filtered_contours, bounding_boxes

def draw_contours_and_boxes(image, contours, bounding_boxes):
    """Draw bounding boxes on image"""
    if len(image.shape) == 2:
        result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result_img = image.copy()
    
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return result_img

def extract_defect_regions(test_img, bounding_boxes, margin=2):
    """Extract defect ROIs with margin"""
    defect_regions = []
    
    for x, y, w, h in bounding_boxes:
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(test_img.shape[1], x + w + margin)
        y2 = min(test_img.shape[0], y + h + margin)
        
        roi = test_img[y1:y2, x1:x2]
        defect_regions.append(roi)
    
    return defect_regions