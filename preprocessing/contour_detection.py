import cv2
import numpy as np

def detect_contours(defect_mask, min_area=10, max_area=500):
    """
    Improved contour detection for PCB defects
    """
    # Find contours
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    filtered_contours = []
    
    print(f"Found {len(contours)} raw contours")
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Basic area filtering
        if area < min_area or area > max_area:
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # CRITICAL FIX: Skip if bounding box has zero width or height
        if w < 1 or h < 1:
            print(f"Skipping contour with zero dimension: w={w}, h={h}")
            continue
            
        # Filter based on dimensions
        if w < 2 or h < 2:  # Too small in one dimension
            continue
            
        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        # Filter extreme aspect ratios (likely noise)
        if aspect_ratio > 8 or aspect_ratio < 0.125:
            continue
            
        filtered_contours.append(contour)
        bounding_boxes.append((x, y, w, h))
    
    print(f"After filtering: {len(filtered_contours)} contours")
    
    return filtered_contours, bounding_boxes

def draw_contours_and_boxes(image, contours, bounding_boxes):
    """
    Draw bounding boxes with defect type information
    """
    if len(image.shape) == 2:
        result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result_img = image.copy()
    
    # Draw bounding boxes only (skip contours for cleaner look)
    valid_boxes = 0
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Double-check: Skip if box has zero width or height
        if w < 1 or h < 1:
            print(f"Warning: Found zero-dimension box in drawing: w={w}, h={h}")
            continue
            
        # Draw bounding box
        color = (255, 0, 0)  # Blue
        
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        
        # Add defect number
        valid_boxes += 1
    
    # Add count information
    
    print(f"Drew {valid_boxes} valid bounding boxes")
    
    return result_img

def extract_defect_regions(test_img, bounding_boxes, margin=2):
    """
    Extract defect regions as ROIs
    """
    defect_regions = []
    
    for (x, y, w, h) in bounding_boxes:
        # Skip zero-dimension boxes
        if w < 1 or h < 1:
            continue
            
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(test_img.shape[1], x + w + margin)
        y2 = min(test_img.shape[0], y + h + margin)
        
        roi = test_img[y1:y2, x1:x2]
        defect_regions.append(roi)
    
    return defect_regions