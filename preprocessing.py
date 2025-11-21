import cv2
import os
import shutil
import numpy as np

input_path = "C:/Users/laksh/OneDrive/Desktop/coding/Circuitguard_Project/Dataset/PCBData"
paired_path = "C:/Users/laksh/OneDrive/Desktop/coding/Circuitguard_Project/Dataset/PCBData_Paired"
base_output = "C:/Users/laksh/OneDrive/Desktop/coding/Circuitguard_Project/preprocessing/output"
diff_path = os.path.join(base_output, "diff_images")
mask_path = os.path.join(base_output, "mask_images")
vis_path = os.path.join(base_output, "vis_images")
roi_output_path = os.path.join(base_output, "ROIs")
combined_roi_path = os.path.join(base_output, "combined_rois")

for folder in [paired_path, base_output, diff_path, mask_path, vis_path, roi_output_path, combined_roi_path]:
    os.makedirs(folder, exist_ok=True)

for root, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith("_test.jpg"):
            base = file.replace("_test.jpg", "")
            test_file = os.path.join(root, file)
            temp_file = os.path.join(root, base + "_temp.jpg")

            if not os.path.exists(temp_file):
                print(f" Missing template for {file}")
                continue

            pair_folder = os.path.join(paired_path, base)
            os.makedirs(pair_folder, exist_ok=True)

            shutil.copy(test_file, os.path.join(pair_folder, "test.jpg"))
            shutil.copy(temp_file, os.path.join(pair_folder, "template.jpg"))

            print(f"Created pair folder: {base}")

for pair_folder in os.listdir(paired_path):
    folder_path = os.path.join(paired_path, pair_folder)
    template_path = os.path.join(folder_path, "template.jpg")
    test_path = os.path.join(folder_path, "test.jpg")

    if not (os.path.exists(template_path) and os.path.exists(test_path)):
        continue

    temp_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if temp_img is None or test_img is None:
        print(f" {pair_folder}, image not readable.")
        continue

    
    diff = cv2.absdiff(test_img, temp_img)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    diff_file = os.path.join(diff_path, f"{pair_folder}_diff.jpg")
    mask_file = os.path.join(mask_path, f"{pair_folder}_mask.jpg")
    cv2.imwrite(diff_file, diff)
    cv2.imwrite(mask_file, mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    roi_images = []
    roi_count = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 50:
            continue

        roi = test_img[y:y+h, x:x+w]
        roi_name = f"{pair_folder}_ROI{roi_count}.jpg"
        cv2.imwrite(os.path.join(roi_output_path, roi_name), roi)
        roi_images.append(roi)

        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        label_text = f"ROI{roi_count}"
        cv2.putText(vis_img, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)

        roi_count += 1

    vis_file = os.path.join(vis_path, f"{pair_folder}_vis.jpg")
    cv2.imwrite(vis_file, vis_img)

    if roi_images:
        roi_images_resized = [cv2.resize(r, (100, 100)) for r in roi_images]
        cols = 3
        rows = (len(roi_images_resized) + cols - 1) // cols
        while len(roi_images_resized) < rows * cols:
            roi_images_resized.append(np.zeros((100, 100), dtype=np.uint8))
        combined_rows = [
            np.hstack(roi_images_resized[i * cols:(i + 1) * cols])
            for i in range(rows)
        ]
        combined_image = np.vstack(combined_rows)
        combined_path = os.path.join(combined_roi_path, f"{pair_folder}_combined.jpg")
        cv2.imwrite(combined_path, combined_image)

    print(f" Processed: {pair_folder} | ROIs: {roi_count}")

print(" Preprocessing done! All ROIs, masks, diffs, and labeled visualizations are saved.")
