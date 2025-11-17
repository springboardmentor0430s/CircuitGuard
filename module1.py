import os
import cv2
import numpy as np

BASE_PATH = r"C:\Users\prati\Desktop\PCB_DATASET\pcbimages"
SAVE_DIFF = r"C:\Users\prati\Desktop\PCB_DATASET\outputs\diff_images"
SAVE_MASK = r"C:\Users\prati\Desktop\PCB_DATASET\outputs\mask_images"

os.makedirs(SAVE_DIFF, exist_ok=True)
os.makedirs(SAVE_MASK, exist_ok=True)

for cls in os.listdir(BASE_PATH):
    class_path = os.path.join(BASE_PATH, cls)
    if not os.path.isdir(class_path):
        continue

    diff_class_path = os.path.join(SAVE_DIFF, cls)
    mask_class_path = os.path.join(SAVE_MASK, cls)
    os.makedirs(diff_class_path, exist_ok=True)
    os.makedirs(mask_class_path, exist_ok=True)

    files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
    test_files = [f for f in files if '_test' in f]

    for test_file in test_files:
        prefix = test_file.split('_')[0]  # e.g., 12100001
        temp_file = f"{prefix}_temp.jpg"
        temp_path = os.path.join(class_path, temp_file)
        test_path = os.path.join(class_path, test_file)

        if not os.path.exists(temp_path):
            print(f"No template found for {test_file}")
            continue

        img_test = cv2.imread(test_path)
        img_temp = cv2.imread(temp_path)
        img_temp = cv2.resize(img_temp, (img_test.shape[1], img_test.shape[0]))

        diff = cv2.absdiff(img_test, img_temp)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        out_name = f"{prefix}.jpg"
        cv2.imwrite(os.path.join(diff_class_path, out_name), diff)
        cv2.imwrite(os.path.join(mask_class_path, out_name), mask)

print("Done! Diff & mask images are saved.")
