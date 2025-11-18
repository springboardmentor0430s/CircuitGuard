import os
import shutil

root_dir = r"C:\Users\Dell\Downloads\DeepPCB-master\DeepPCB-master\PCBData\Dataset_Split"

valid_classes = {"mousebite", "short", "spur", "open", "pinhole", "spurious copper"}

for split in ["train", "val", "test"]:
    split_path = os.path.join(root_dir, split)
    if not os.path.exists(split_path):
        print(f"❌ Missing folder: {split_path}")
        continue

    for folder in os.listdir(split_path):
        full_path = os.path.join(split_path, folder)
        if os.path.isdir(full_path) and folder not in valid_classes:
            print(f"🗑️ Removing invalid folder: {full_path}")
            shutil.rmtree(full_path)

print("✅ Dataset cleaned. Only 6 valid classes remain in train/val/test.")
