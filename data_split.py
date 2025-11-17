import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# IMPORTANT: This path has been updated to the one provided in your console output.
SOURCE_ROOT = r'C:\Users\DELL\OneDrive\Desktop\New folder\extracted_rois_group_group00041'
TARGET_ROOT = 'Final_PCB_Split'

# Define split ratios
TRAIN_RATIO = 0.7  # 70% for Training
VAL_RATIO = 0.1    # 10% for Validation
TEST_RATIO = 0.2   # 20% for Testing (must sum to 1.0)

# --- EXECUTION ---

def stratified_split(source_root, target_root):
    print("Starting stratified split...")

    all_files = []
    all_labels = []

    # Ensure os.listdir works on the source_root
    if not os.path.exists(source_root):
        print(f"Error: SOURCE_ROOT path does not exist: {source_root}")
        return

    class_names = sorted([d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))])

    # 1. Collect all file paths and labels
    for class_name in class_names:
        class_path = os.path.join(source_root, class_name)
        files = glob(os.path.join(class_path, '*'))
        all_files.extend(files)
        all_labels.extend([class_name] * len(files))

    if not all_files:
        print(f"Error: No files found in any class subfolder within {source_root}. Check contents.")
        return

    print(f"Total files collected: {len(all_files)}")
    print(f"Defect classes found: {class_names}")

    # 2. First split: Train vs (Validation + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_files, all_labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=all_labels,
        random_state=42  # for reproducibility
    )

    # 3. Second split: Validation vs Test from the temporary set
    # The ratio needs to be calculated based on the size of the temporary set
    test_val_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_val_ratio,
        stratify=y_temp,
        random_state=42
    )

    split_sets = {
        'train': (X_train, y_train),
        'validation': (X_val, y_val),
        'test': (X_test, y_test)
    }

    print(f"\nSplit Sizes:")
    print(f"- Train: {len(X_train)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"- Validation: {len(X_val)} ({VAL_RATIO*100:.0f}%)")
    print(f"- Test: {len(X_test)} ({TEST_RATIO*100:.0f}%)")

    # 4. Move files to the target structure
    for set_name, (file_paths, labels) in split_sets.items():
        print(f"\nCreating {set_name} directory structure...")
        for class_name in class_names:
            os.makedirs(os.path.join(target_root, set_name, class_name), exist_ok=True)

        for file_path, label in zip(file_paths, labels):
            # Get just the filename (e.g., 00041003_0.jpg)
            filename = os.path.basename(file_path)

            target_path = os.path.join(target_root, set_name, label, filename)

            # Copy the file (using copy is safer than move initially)
            shutil.copy(file_path, target_path)

    print("\nData splitting and organization complete!")


if __name__ == '__main__':
    # --- PRE-REQUISITE: Install scikit-learn if you haven't already ---
    # !pip install scikit-learn

    # The main SOURCE_ROOT is defined at the top of the file.
    # Use that variable directly, which now contains the correct path.

    stratified_split(SOURCE_ROOT, TARGET_ROOT)
