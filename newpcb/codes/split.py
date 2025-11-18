import os
import shutil
import random

# ✅ --- Update dataset paths for Google Drive ---
organized_root = r"C:\Users\Dell\Downloads\newpcb-20251027T064831Z-1-001-20251027T105042Z-1-001\newpcb-20251027T064831Z-1-001\newpcb\PCBData\DefectROIs_organized"
split_root = r"C:\Users\Dell\Downloads\newpcb-20251027T064831Z-1-001-20251027T105042Z-1-001\newpcb-20251027T064831Z-1-001\newpcb\PCBData\Dataset_split"

# ✅ --- Verify input directory exists ---
if not os.path.exists(organized_root):
    raise FileNotFoundError(f"❌ Input folder not found: {organized_root}. Check your Drive mount path and folder name!")

# ✅ --- Defect classes ---
classes = ["mousebite", "spur", "short", "open", "pinhole", "spurious copper"]

# ✅ --- Create split folders ---
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(split_root, split, cls), exist_ok=True)

# ✅ --- Ratios ---
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# ✅ --- Random seed for reproducibility ---
random.seed(42)

# ✅ --- Split and copy images ---
for cls in classes:
    cls_path = os.path.join(organized_root, cls)
    if not os.path.exists(cls_path):
        print(f"⚠️ Warning: Class folder '{cls}' not found in {organized_root}, skipping...")
        continue

    images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".png"))]
    if len(images) == 0:
        print(f"⚠️ No images found for class {cls}, skipping...")
        continue

    random.shuffle(images)

    total = len(images)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for img in train_imgs:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(split_root, "train", cls, img))
    for img in val_imgs:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(split_root, "val", cls, img))
    for img in test_imgs:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(split_root, "test", cls, img))

    print(f"✅ {cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images")

print("\n🎯 Dataset organized into train/val/test split successfully!")