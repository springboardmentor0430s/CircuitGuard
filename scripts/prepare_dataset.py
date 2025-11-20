# scripts/prepare_dataset.py
import os
import cv2
import random
from tqdm import tqdm

# Input directories
ANNOTATIONS_DIR = 'data/raw/annotations'
IMAGES_DIR = 'data/raw/test'

# Output directories
TRAIN_DIR = 'data/processed/train'
TEST_DIR = 'data/processed/test'
SPLIT_RATIO = 0.8

# --------------------------------------------------------------------
# THIS FUNCTION HAS BEEN COMPLETELY REWRITTEN AND IS NOW CORRECT
# --------------------------------------------------------------------
def parse_annotation(txt_path):
    objects = []
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                # The format is direct pixel coordinates: xmin ymin xmax ymax class_id
                xmin = int(parts[0])
                ymin = int(parts[1])
                xmax = int(parts[2])
                ymax = int(parts[3])
                class_id = int(parts[4])
                
                # The class IDs in the file are 1-6. We convert them to 0-5 for our list.
                # Class mapping: 1:open, 2:short, 3:mousebite, 4:spur, 5:copper, 6:pinhole
                class_map = ['Open_circuit', 'Short', 'Mouse_bite', 'Spur', 'Spurious_copper', 'Missing_hole']
                
                # Subtract 1 from class_id to get the correct index
                if 1 <= class_id <= len(class_map):
                    class_name = class_map[class_id - 1]
                    objects.append({'name': class_name, 'bndbox': (xmin, ymin, xmax, ymax)})

    except Exception as e:
        print(f"  - Error parsing {txt_path}: {e}")
    return objects
# --------------------------------------------------------------------

def prepare_dataset():
    print("Starting dataset preparation...")
    
    ignore_list = ['test.txt', 'trainval.txt']
    txt_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.txt') and f not in ignore_list]
    
    random.shuffle(txt_files)
    split_index = int(len(txt_files) * SPLIT_RATIO)
    train_files, test_files = txt_files[:split_index], txt_files[split_index:]

    for dataset_type, file_list, output_dir in [('TRAIN', train_files, TRAIN_DIR), ('TEST', test_files, TEST_DIR)]:
        print(f"\nProcessing {dataset_type} set...")
        for txt_file in tqdm(file_list, desc=f"Processing {dataset_type} files"):
            image_filename = txt_file.replace('.txt', '_test.jpg')
            image_path = os.path.join(IMAGES_DIR, image_filename)
            
            image = cv2.imread(image_path)
            if image is None: continue
            
            # The parse function no longer needs image width/height
            objects = parse_annotation(os.path.join(ANNOTATIONS_DIR, txt_file))
            
            for i, obj in enumerate(objects):
                class_name = obj['name']
                xmin, ymin, xmax, ymax = obj['bndbox']
                roi = image[ymin:ymax, xmin:xmax]
                
                if roi.size == 0:
                    continue

                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                save_path = os.path.join(class_dir, f"{os.path.splitext(image_filename)[0]}_{i}.jpg")
                cv2.imwrite(save_path, roi)
                
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    prepare_dataset()