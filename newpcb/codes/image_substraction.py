import os
import cv2
import numpy as np
import shutil

# Paths (update to your local dataset)
pcb_data_path = r"C:\Users\Dell\Downloads\newpcb-20251027T064831Z-1-001-20251027T105042Z-1-001\newpcb-20251027T064831Z-1-001\newpcb\PCBData"
mask_root = os.path.join(pcb_data_path, "DefectMasks")

# Recreate DefectMasks output folder
if os.path.exists(mask_root):
    shutil.rmtree(mask_root)
os.makedirs(mask_root, exist_ok=True)

print("🧹 Cleaning only output masks directory...")

total_masks_created = 0

# Process all PCB groups
for group in os.listdir(pcb_data_path):
    if not group.startswith("group"):
        continue

    group_path = os.path.join(pcb_data_path, group)
    subfolders = [
        s for s in os.listdir(group_path)
        if os.path.isdir(os.path.join(group_path, s)) and not s.endswith('_not')
    ]

    for sub in subfolders:
        sub_path = os.path.join(group_path, sub)
        files = os.listdir(sub_path)

        temps = [f for f in files if "_temp" in f and (f.endswith('.png') or f.endswith('.jpg'))]

        mask_group = os.path.join(mask_root, group)
        os.makedirs(mask_group, exist_ok=True)

        for temp in temps:
            base = temp.replace("_temp.png", "").replace("_temp.jpg", "")
            test_name = f"{base}_test.png" if f"{base}_test.png" in files else f"{base}_test.jpg"

            if test_name not in files:
                print(f"⚠ Missing test image for {temp}, skipped.")
                continue

            temp_path = os.path.join(sub_path, temp)
            test_path = os.path.join(sub_path, test_name)

            try:
                temp_img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
                test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

                if temp_img is None or test_img is None:
                    print(f"⚠ Unreadable file: {base}")
                    continue

                # Resize to 128x128
                temp_img = cv2.resize(temp_img, (128, 128))
                test_img = cv2.resize(test_img, (128, 128))

                diff = cv2.absdiff(test_img, temp_img)
                _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Morphological cleanup
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # Optional debug visualization
                combined = np.hstack((temp_img, test_img, diff, mask))
                debug_path = os.path.join(mask_group, f"{base}_debug.jpg")
                cv2.imwrite(debug_path, combined, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                out_name = f"{base}_mask.jpg"
                cv2.imwrite(os.path.join(mask_group, out_name), mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                total_masks_created += 1
                if total_masks_created <= 5:
                    print(f"✅ Created mask: {group}/{out_name}")

            except Exception as e:
                print(f"❌ Error processing {temp}: {e}")

print(f"\n🎯 Module 1 Complete! Total masks created: {total_masks_created}")
print("📁 DefectMasks and debug previews updated.")