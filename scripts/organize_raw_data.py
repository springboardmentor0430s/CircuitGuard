import os
import shutil
from tqdm import tqdm

TEMPLATE_DEST = 'data/raw/template'
TEST_DEST = 'data/raw/test'
ANNOTATIONS_DEST = 'data/raw/annotations'

def organize_files():
    print("--- Please drag and drop the 'PCBData' folder into this window and press Enter ---")
    source_path_input = input("Path to 'PCBData' folder: ")
    
    SOURCE_DATA_DIR = source_path_input.strip().strip('"').strip("'")

    print(f"\nScanning source directory: {SOURCE_DATA_DIR}")
    
    if not os.path.isdir(SOURCE_DATA_DIR):
        print(f"\n--- ERROR ---")
        print(f"Source directory not found at '{SOURCE_DATA_DIR}'")
        print("Please run the script again and provide the correct path.")
        return

    all_files = []
    for root, _, files in os.walk(SOURCE_DATA_DIR):
        for file in files:
            all_files.append(os.path.join(root, file))

    print(f"Found {len(all_files)} total files. Sorting and copying...")
    for file_path in tqdm(all_files, desc="Copying raw files"):
        filename = os.path.basename(file_path)
        if filename.endswith('_temp.jpg'):
            shutil.copy(file_path, os.path.join(TEMPLATE_DEST, filename))
        elif filename.endswith('.jpg'):
            shutil.copy(file_path, os.path.join(TEST_DEST, filename))
        elif filename.endswith('.txt'):
            shutil.copy(file_path, os.path.join(ANNOTATIONS_DEST, filename))

    print("\nRaw file organization complete!")
    print(f" - {len(os.listdir(TEMPLATE_DEST))} template images copied.")
    print(f" - {len(os.listdir(TEST_DEST))} test images copied.")
    print(f" - {len(os.listdir(ANNOTATIONS_DEST))} annotation files copied.")

if __name__ == "__main__":
    organize_files()