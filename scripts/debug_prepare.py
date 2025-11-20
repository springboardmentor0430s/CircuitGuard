# scripts/debug_prepare.py
import os
import cv2

# --- CONFIGURATION ---
# We will focus on just ONE file to see what's happening.
# Let's use '00041008.txt' as an example from your file list.
ANNOTATION_FILENAME = '00041008.txt'

ANNOTATIONS_DIR = 'data/raw/annotations'
IMAGES_DIR = 'data/raw/test'

def debug_single_file():
    print(f"--- STARTING DEBUG FOR: {ANNOTATION_FILENAME} ---")

    # 1. Construct file paths
    txt_path = os.path.join(ANNOTATIONS_DIR, ANNOTATION_FILENAME)
    image_filename = ANNOTATION_FILENAME.replace('.txt', '_test.jpg')
    image_path = os.path.join(IMAGES_DIR, image_filename)

    print(f"\n[1] Looking for annotation file at: {os.path.abspath(txt_path)}")
    print(f"[2] Looking for image file at: {os.path.abspath(image_path)}")

    # 2. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("\n--- ERROR: Could not load the image. The debug cannot continue. ---")
        return
    
    image_h, image_w, _ = image.shape
    print(f"\n[3] Successfully loaded image. Dimensions (H, W): ({image_h}, {image_w})")

    # 4. Parse the annotation file line by line
    print(f"\n[4] Reading annotation file: {txt_path}")
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                print("  -> File is empty. No defects to process.")
                return

            for i, line in enumerate(lines):
                print(f"\n--- Processing Line #{i+1} ---")
                print(f"  -> Raw line: '{line.strip()}'")
                
                parts = line.strip().split()
                if len(parts) < 5:
                    print("  -> SKIPPING line: Not enough parts.")
                    continue

                x_center, y_center, w, h = map(float, parts[:4])
                class_id = int(parts[4])
                print(f"  -> Parsed normalized values: x_center={x_center}, y_center={y_center}, w={w}, h={h}, class_id={class_id}")
                
                w_px = w * image_w
                h_px = h * image_h
                x_center_px = x_center * image_w
                y_center_px = y_center * image_h
                print(f"  -> Calculated pixel values (center-based): x_center_px={x_center_px:.2f}, y_center_px={y_center_px:.2f}, w_px={w_px:.2f}, h_px={h_px:.2f}")
                
                xmin = int(x_center_px - (w_px / 2))
                ymin = int(y_center_px - (h_px / 2))
                xmax = int(x_center_px + (w_px / 2))
                ymax = int(y_center_px + (h_px / 2))
                print(f"  -> Calculated Bounding Box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

                class_map = ['Open_circuit', 'Short', 'Mouse_bite', 'Spur', 'Spurious_copper', 'Missing_hole']
                class_name = class_map[class_id]
                print(f"  -> Defect class name: {class_name}")

                roi = image[ymin:ymax, xmin:xmax]
                print(f"  -> Cropped ROI. Size of cropped image (Height, Width, Channels): {roi.shape}")
                
                if roi.size == 0:
                    print("  -> RESULT: The cropped image is EMPTY. This is why it's not being saved.")
                else:
                    print("  -> RESULT: The cropped image is VALID. It should be saved.")

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED: {e} ---")

    print("\n--- DEBUG COMPLETE ---")


if __name__ == "__main__":
    debug_single_file()