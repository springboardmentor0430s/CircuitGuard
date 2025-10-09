import cv2
import numpy as np

def xor_defect_detection(test_img, template_img):
    """
    XOR-based defect detection for binary PCB images
    This works much better for black/white binarized images
    """
    # Ensure images are binary (0 and 255)
    if len(test_img.shape) == 3:
        test_binary = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        template_binary = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    else:
        test_binary = test_img.copy()
        template_binary = template_img.copy()
    
    # Make sure images are properly binary (0 and 255)
    _, test_binary = cv2.threshold(test_binary, 127, 255, cv2.THRESH_BINARY)
    _, template_binary = cv2.threshold(template_binary, 127, 255, cv2.THRESH_BINARY)
    
    print(f"Test binary unique: {np.unique(test_binary)}")
    print(f"Template binary unique: {np.unique(template_binary)}")
    
    # Method 1: XOR operation - finds pixels that are different
    xor_result = cv2.bitwise_xor(test_binary, template_binary)
    
    # Method 2: AND operation to find missing copper (test has 0, template has 255)
    missing_copper = cv2.bitwise_and(template_binary, cv2.bitwise_not(test_binary))
    
    # Method 3: AND operation to find extra copper (test has 255, template has 0)
    extra_copper = cv2.bitwise_and(test_binary, cv2.bitwise_not(template_binary))
    
    # Combine all defect types
    combined_defects = cv2.bitwise_or(xor_result, missing_copper)
    combined_defects = cv2.bitwise_or(combined_defects, extra_copper)
    
    # Clean up the result
    kernel = np.ones((2, 2), np.uint8)
    cleaned_defects = cv2.morphologyEx(combined_defects, cv2.MORPH_OPEN, kernel)
    
    # Remove very small noise with validation
    cleaned_defects = remove_tiny_components_with_validation(cleaned_defects, min_size=5)
    
    print(f"XOR defects: {np.count_nonzero(xor_result)}")
    print(f"Missing copper: {np.count_nonzero(missing_copper)}")
    print(f"Extra copper: {np.count_nonzero(extra_copper)}")
    print(f"Final defects: {np.count_nonzero(cleaned_defects)}")
    
    return {
        'xor': xor_result,
        'missing': missing_copper,
        'extra': extra_copper,
        'combined': cleaned_defects
    }

def remove_tiny_components(binary_img, min_size=5):
    """
    Remove very small connected components
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    output = np.zeros_like(binary_img)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255
    
    return output

def remove_tiny_components_with_validation(binary_img, min_size=5):
    """
    Remove small components with additional validation to prevent zero-dimension boxes
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    output = np.zeros_like(binary_img)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Skip components that would create zero-dimension bounding boxes
        if area >= min_size or width >= 2 or height >= 2:
            output[labels == i] = 255
    
    removed_count = num_labels - 1 - (np.count_nonzero(output) // 255)
    if removed_count > 0:
        print(f"Removed {removed_count} components that would create invalid bounding boxes")
    
    return output

def highlight_xor_defects(test_img, defects_dict, color=(0, 0, 255)):
    """
    Highlight different types of defects with different colors
    """
    if len(test_img.shape) == 2:
        result_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    else:
        result_img = test_img.copy()
    
    # Highlight missing copper (holes, breaks) in red
    missing_mask = np.zeros_like(result_img)
    missing_mask[defects_dict['missing'] > 0] = (0, 0, 255)  # Red
    result_img = cv2.addWeighted(result_img, 1, missing_mask, 0.4, 0)
    
    # Highlight extra copper (shorts, spurious) in blue
    extra_mask = np.zeros_like(result_img)
    extra_mask[defects_dict['extra'] > 0] = (255, 0, 0)  # Blue
    result_img = cv2.addWeighted(result_img, 1, extra_mask, 0.4, 0)
    
    return result_img