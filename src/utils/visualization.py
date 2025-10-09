import cv2
import numpy as np

def resize_image_to_height(img, target_height):
    """Height-preserving resize"""
    if img is None:
        return None
        
    h, w = img.shape[:2]
    if h == target_height:
        return img
        
    scale = target_height / h
    new_width = int(w * scale)
    return cv2.resize(img, (new_width, target_height))

def create_comparison_display(template_img, test_img, result_img, contours_img, target_height=400):
    """Generate comparison grid"""
    template_resized = resize_image_to_height(template_img, target_height)
    test_resized = resize_image_to_height(test_img, target_height)
    contours_resized = resize_image_to_height(contours_img, target_height)
    
    if len(template_resized.shape) == 2:
        template_resized = cv2.cvtColor(template_resized, cv2.COLOR_GRAY2BGR)
    if len(contours_resized.shape) == 2:
        contours_resized = cv2.cvtColor(contours_resized, cv2.COLOR_GRAY2BGR)
    if len(test_resized.shape) == 2:
        test_resized = cv2.cvtColor(test_resized, cv2.COLOR_GRAY2BGR)
    
    combined = np.hstack([template_resized, test_resized, contours_resized])
    line_color = (128, 0, 128)
    line_thickness = 5
    v_line_x = template_resized.shape[1]
    cv2.line(combined, (v_line_x, 0), (v_line_x, combined.shape[0]), line_color, line_thickness)
    
    return combined

def display_results(template_img, test_img, result_img, contours_img, window_name="PCB Defect Detection"):
    """Show detection results"""
    display_img = create_comparison_display(template_img, test_img, result_img, contours_img, target_height=500)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = min(display_img.shape[1], 1600)
    window_height = min(display_img.shape[0], 1200)
    cv2.resizeWindow(window_name, window_width, window_height)
    cv2.imshow(window_name, display_img)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()