import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===================================================
# PATHS
# ===================================================
MASK_PATH = r"C:\Users\prati\Desktop\PCB_DATASET\outputs\mask_images"
DIFF_PATH = r"C:\Users\prati\Desktop\PCB_DATASET\outputs\diff_images"
ROI_SAVE_PATH = r"C:\Users\prati\Desktop\PCB_DATASET\outputs\roi_images"

os.makedirs(ROI_SAVE_PATH, exist_ok=True)

# ===================================================
# FUNCTION TO EXTRACT CONTOURS AND ROIs
# ===================================================
def extract_rois(mask_img, diff_img, min_area=20):
    """
    mask_img : binary mask image of defects
    diff_img : original difference image for visualization
    min_area : ignore tiny noise regions
    """
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue  # skip small noise
        x, y, w, h = cv2.boundingRect(cnt)
        roi = diff_img[y:y+h, x:x+w]
        rois.append((roi, (x, y, w, h)))
    return rois

# ===================================================
# MAIN LOOP: Extract ROIs
# ===================================================
print("🔧 Extracting defect ROIs...")

for cls in sorted(os.listdir(MASK_PATH)):
    mask_class_path = os.path.join(MASK_PATH, cls)
    diff_class_path = os.path.join(DIFF_PATH, cls)
    roi_class_path = os.path.join(ROI_SAVE_PATH, cls)
    os.makedirs(roi_class_path, exist_ok=True)

    mask_files = [f for f in os.listdir(mask_class_path) if f.lower().endswith(('.jpg', '.png'))]

    for mask_name in mask_files:
        mask_img = cv2.imread(os.path.join(mask_class_path, mask_name), cv2.IMREAD_GRAYSCALE)
        diff_img = cv2.imread(os.path.join(diff_class_path, mask_name))

        rois = extract_rois(mask_img, diff_img)

        for idx, (roi, (x, y, w, h)) in enumerate(rois):
            roi_name = f"{os.path.splitext(mask_name)[0]}_roi{idx}.jpg"
            cv2.imwrite(os.path.join(roi_class_path, roi_name), roi)

print("✅ ROI images saved successfully!")

# ===================================================
# OPTIONAL: Visualize some examples
# ===================================================
for cls in sorted(os.listdir(ROI_SAVE_PATH))[:2]:
    roi_files = os.listdir(os.path.join(ROI_SAVE_PATH, cls))
    if not roi_files:
        continue
    sample_roi = cv2.imread(os.path.join(ROI_SAVE_PATH, cls, roi_files[0]))
    plt.figure(figsize=(4,4))
    plt.imshow(cv2.cvtColor(sample_roi, cv2.COLOR_BGR2RGB))
    plt.title(f"Class {cls} Sample ROI")
    plt.show()
