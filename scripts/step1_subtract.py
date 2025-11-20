import cv2
import numpy as np

# --- 1. Load and Preprocess Images ---
template_path = 'data/PCB_DATASET/PCB_USED/08.jpg'
test_image_path = 'data/PCB_DATASET/PCB_USED/09.jpg'

template = cv2.imread(template_path)
test_image = cv2.imread(test_image_path)

if template is None or test_image is None:
    print("Error: Could not load one or both images. Check the file paths.")
else:
    height, width, _ = template.shape
    test_image_resized = cv2.resize(test_image, (width, height))
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    test_image_gray = cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2GRAY)

    # --- 2. Create Difference and Mask ---
    difference = cv2.absdiff(template_gray, test_image_gray)
    _, threshold_mask = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Noise Reduction using Morphology ---
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel)

    # --- 3. Find Contours and Extract ROIs ---
    print("Finding defect contours...")
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = test_image_resized.copy()
    
    defect_count = 0
    for contour in contours:
        if cv2.contourArea(contour) < 20:
            continue
        
        defect_count += 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = test_image_resized[y:y+h, x:x+w]

        # --- NEW: Resize the ROI for better viewing ---
        # We'll create a 200x200 pixel version of the ROI to display
        display_roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_NEAREST)
        
        print(f"Displaying defect {defect_count}. Press any key to continue...")
        # Display the NEW resized ROI
        cv2.imshow("Cropped Defect (ROI)", display_roi)
        cv2.waitKey(0)

    # --- 4. Display the Final Result ---
    print(f"Found and displayed {defect_count} defects.")
    cv2.imshow("Cleaned Mask", cleaned_mask)
    cv2.imshow("Final Result with Bounding Boxes", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()