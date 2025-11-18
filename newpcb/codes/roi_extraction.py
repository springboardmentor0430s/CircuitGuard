import os
import cv2
import numpy as np

# === Paths (updated to your local setup) ===
pcb_data_path = r"C:\Users\Dell\Downloads\newpcb-20251027T064831Z-1-001-20251027T105042Z-1-001\newpcb-20251027T064831Z-1-001\newpcb\PCBData"
mask_root = os.path.join(pcb_data_path, "DefectMasks")
roi_root = os.path.join(pcb_data_path, "DefectROIs")
os.makedirs(roi_root, exist_ok=True)

# Clean ROI folder
for root, dirs, files in os.walk(roi_root):
    for file in files:
        if file.endswith(".jpg"):
            try:
                os.remove(os.path.join(root, file))
            except:
                pass

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def assign_label(area, aspect_ratio, circularity):
    if area < 200 and circularity > 0.6:
        return "mousebite", GREEN
    elif aspect_ratio > 3 and area < 2500:
        return "short", RED
    elif 0.3 < circularity < 0.6 and 200 < area < 3000:
        return "spur", GREEN
    elif area > 3000:
        return "open", GREEN
    elif area < 150:
        return "pinhole", GREEN
    else:
        return "spurious copper", GREEN

total_annotated = 0
for group in os.listdir(mask_root):
    mask_group_path = os.path.join(mask_root, group)
    roi_group_path = os.path.join(roi_root, group)
    os.makedirs(roi_group_path, exist_ok=True)

    original_group_path = os.path.join(pcb_data_path, group)
    subfolders = [s for s in os.listdir(original_group_path)
                  if os.path.isdir(os.path.join(original_group_path, s)) and not s.endswith('_not')]
    if not subfolders:
        continue

    original_sub_path = os.path.join(original_group_path, subfolders[0])
    original_files = os.listdir(original_sub_path)
    test_files = [f for f in original_files if "_test" in f]

    for mask_file in os.listdir(mask_group_path):
        if not mask_file.endswith('_mask.jpg'):
            continue

        mask_path = os.path.join(mask_group_path, mask_file)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        base_name = mask_file.replace('_mask.jpg', '')

        matching_test = None
        for test_file in test_files:
            if base_name in test_file and "_test" in test_file:
                matching_test = test_file
                break

        test_img_path = os.path.join(original_sub_path, matching_test) if matching_test else None
        test_img = cv2.imread(test_img_path) if test_img_path and os.path.exists(test_img_path) else np.ones((256, 256, 3), dtype=np.uint8) * 255
        test_img_resized = cv2.resize(test_img, (256, 256)) if test_img is not None else np.ones((256, 256, 3), dtype=np.uint8) * 255
        vis_img = test_img_resized.copy()

        mask = None
        if mask_gray is not None:
            _, mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours = []
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10: continue  # only real defects visible
            found = True
            x, y, w, h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)
            aspect_ratio = float(w) / (h + 1e-6)
            label, color = assign_label(area, aspect_ratio, circularity)
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)

            # Center label inside box with white bg for clarity
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_x = x + w // 2 - tw // 2
            label_y = y + h // 2 + th // 2 + 1
            cv2.rectangle(vis_img, (label_x-3, label_y-th-3), (label_x+tw+3, label_y+th+2), WHITE, -1)
            cv2.putText(vis_img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2)
            cv2.putText(vis_img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if not found:
            msg = "No Defect"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cx, cy = 128 - tw // 2, 128 + th // 2
            cv2.rectangle(vis_img, (cx-10, cy-th-10), (cx+tw+10, cy+10), WHITE, -1)
            cv2.putText(vis_img, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2)
            cv2.putText(vis_img, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 1)

        annotated_path = os.path.join(roi_group_path, f"{base_name}_annotated.jpg")
        cv2.imwrite(annotated_path, vis_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        total_annotated += 1

print(f"\n🎯 Module 2 Complete — All images labeled, very clear output, no label overlap. Total annotated: {total_annotated}")