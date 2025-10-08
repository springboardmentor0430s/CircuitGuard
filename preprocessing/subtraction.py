import cv2
import numpy as np

def preprocess_images(test_img, template_img):
    """
    Preprocess images for subtraction
    """
    # Convert to grayscale
    if len(test_img.shape) == 3:
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    else:
        test_gray = test_img.copy()
        
    if len(template_img.shape) == 3:
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_img.copy()
    
    # Apply Gaussian blur to reduce noise
    test_blur = cv2.GaussianBlur(test_gray, (5, 5), 0)
    template_blur = cv2.GaussianBlur(template_gray, (5, 5), 0)
    
    # Normalize images
    test_norm = cv2.normalize(test_blur, None, 0, 255, cv2.NORM_MINMAX)
    template_norm = cv2.normalize(template_blur, None, 0, 255, cv2.NORM_MINMAX)
    
    return test_norm, template_norm

def image_subtraction(test_img, template_img):
    """
    Perform image subtraction and thresholding to highlight defects
    """
    # Preprocess images
    test_proc, template_proc = preprocess_images(test_img, template_img)
    
    # Absolute difference
    diff = cv2.absdiff(test_proc, template_proc)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return diff, thresh, cleaned

def highlight_defects(test_img, defect_mask, color=(0, 0, 255)):
    """
    Highlight defects on the test image
    """
    if len(test_img.shape) == 3:
        result_img = test_img.copy()
    else:
        result_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    
    # Create colored mask
    colored_mask = np.zeros_like(result_img)
    colored_mask[defect_mask > 0] = color
    
    # Blend with original image
    alpha = 0.3
    result_img = cv2.addWeighted(result_img, 1, colored_mask, alpha, 0)
    
    return result_img