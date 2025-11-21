import os
import cv2
import shutil
import random
import numpy as np


paired_path = r"C:\Users\laksh\OneDrive\Desktop\coding\Circuitguard_Project\Dataset\PCBData_Paired"
output_base = r"C:/Users/laksh/OneDrive/Desktop/coding/Circuitguard_Project/preprocessing/output_dataset"

label_map = {
    "1": "open",
    "2": "short",
    "3": "mousebite",
    "4": "missing_hole",
    "5": "spur",
    "6": "spurious_copper",
    "0": "nondefect" 
}

for label in label_map.values():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_base, split, label), exist_ok=True)


train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]

all_defect_images = {label: [] for label in label_map.values()}

for pair_folder in os.listdir(paired_path):
    folder_path = os.path.join(paired_path, pair_folder)
    test_path = os.path.join(folder_path, "test.jpg")
    template_path = os.path.join(folder_path, "template.jpg")
    label_file = os.path.join(folder_path, f"{pair_folder}.txt")

    if not os.path.exists(test_path) or not os.path.exists(template_path):
        print(f"Skipping {pair_folder}: Missing test or template image")
        continue

    test_img = cv2.imread(test_path)
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if test_img is None or template_img is None:
        print(f"Skipping {pair_folder}: Cannot read images")
        continue

    diff = cv2.absdiff(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), template_img)
    blur = cv2.GaussianBlur(diff, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nondefect_mask = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(nondefect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count_nondefect = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h < 50: 
            continue
        roi = test_img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (128,128))
        all_defect_images["nondefect"].append((roi, count_nondefect))
        count_nondefect += 1

    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()

        for defect_id, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            x_min, y_min, x_max, y_max, class_id = map(int, parts)
            label_name = label_map.get(str(class_id), None)
            if label_name is None:
                continue
            roi = safe_crop(test_img, x_min, y_min, x_max, y_max)
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (128,128))
            all_defect_images[label_name].append((roi, defect_id))

print(" Collected all defect and non-defective images by class.")

for label, images in all_defect_images.items():
    random.shuffle(images)
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    for i, (img_data, idx) in enumerate(images):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        save_path = os.path.join(output_base, split, label, f"{label}_{i}.jpg")
        cv2.imwrite(save_path, img_data)

print(" Dataset split complete! Train/Val/Test folders ready for each defect type, including non-defect images.")
