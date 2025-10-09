import cv2
import numpy as np

def xor_defect_detection(test_img, template_img):
    """XOR-based detection on binary images."""
    if len(test_img.shape) == 3:
        test_binary = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        template_binary = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    else:
        test_binary = test_img.copy()
        template_binary = template_img.copy()
    _, test_binary = cv2.threshold(test_binary, 127, 255, cv2.THRESH_BINARY)
    _, template_binary = cv2.threshold(template_binary, 127, 255, cv2.THRESH_BINARY)
    
    
    xor_result = cv2.bitwise_xor(test_binary, template_binary)
    missing_copper = cv2.bitwise_and(template_binary, cv2.bitwise_not(test_binary))
    extra_copper = cv2.bitwise_and(test_binary, cv2.bitwise_not(template_binary))
    combined_defects = cv2.bitwise_or(xor_result, missing_copper)
    combined_defects = cv2.bitwise_or(combined_defects, extra_copper)
    kernel = np.ones((2, 2), np.uint8)
    cleaned_defects = cv2.morphologyEx(combined_defects, cv2.MORPH_OPEN, kernel)
    cleaned_defects = remove_tiny_components_with_validation(cleaned_defects, min_size=5)
    
    
    return {
        'xor': xor_result,
        'missing': missing_copper,
        'extra': extra_copper,
        'combined': cleaned_defects
    }

def remove_tiny_components_with_validation(binary_img, min_size=5):
    """Remove small components with simple validation."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    output = np.zeros_like(binary_img)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= min_size or width >= 2 or height >= 2:
            output[labels == i] = 255
    
    
    return output

def highlight_xor_defects(test_img, defects_dict, color=(0, 0, 255)):
    """Overlay defect types in color."""
    if len(test_img.shape) == 2:
        result_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    else:
        result_img = test_img.copy()
    missing_mask = np.zeros_like(result_img)
    missing_mask[defects_dict['missing'] > 0] = (0, 0, 255)  # Red
    result_img = cv2.addWeighted(result_img, 1, missing_mask, 0.4, 0)
    extra_mask = np.zeros_like(result_img)
    extra_mask[defects_dict['extra'] > 0] = (255, 0, 0)  # Blue
    result_img = cv2.addWeighted(result_img, 1, extra_mask, 0.4, 0)
    
    return result_img