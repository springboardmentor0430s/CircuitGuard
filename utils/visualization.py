import cv2
import numpy as np

def resize_image_to_height(img, target_height):
    """Resize image maintaining aspect ratio"""
    if img is None:
        return None
        
    h, w = img.shape[:2]
    if h == target_height:
        return img
        
    scale = target_height / h
    new_width = int(w * scale)
    return cv2.resize(img, (new_width, target_height))

def create_comparison_display(template_img, test_img, result_img, contours_img, target_height=400):
    """
    Create a side-by-side comparison display with proper aspect ratio
    """
    # Resize all images to the same height while maintaining aspect ratio
    template_resized = resize_image_to_height(template_img, target_height)
    test_resized = resize_image_to_height(test_img, target_height)
    contours_resized = resize_image_to_height(contours_img, target_height)
    
    # Ensure all images are in color
    if len(template_resized.shape) == 2:
        template_resized = cv2.cvtColor(template_resized, cv2.COLOR_GRAY2BGR)
    if len(contours_resized.shape) == 2:
        contours_resized = cv2.cvtColor(contours_resized, cv2.COLOR_GRAY2BGR)
    if len(test_resized) == 2:
        test_resized = cv2.cvtColor(test_resized, cv2.COLOR_GRAY2BGR)
    
    # Create single row: Template and Contours images
    combined = np.hstack([template_resized ,test_resized, contours_resized])
    
    # Add a distinct vertical separator line between images
    line_color = (128,0,128)  # White line
    line_thickness = 5
    
    # Vertical line at the junction
    v_line_x = template_resized.shape[1]
    cv2.line(combined, (v_line_x, 0), (v_line_x, combined.shape[0]), line_color, line_thickness)
    
    return combined

def display_results(template_img, test_img, result_img, contours_img, window_name="PCB Defect Detection"):
    """
    Display the results in a properly sized popup window
    """
    # Create comparison display with larger target height
    display_img = create_comparison_display(template_img, test_img, result_img, contours_img, target_height=500)
    
    # Create a resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set window size
    window_width = min(display_img.shape[1], 1600)
    window_height = min(display_img.shape[0], 1200)
    cv2.resizeWindow(window_name, window_width, window_height)
    
    # Show image
    cv2.imshow(window_name, display_img)
    
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()