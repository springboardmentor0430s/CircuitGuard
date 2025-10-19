import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class DeepPCBDataset:
    def __init__(self, base_path="data/interim"):
        self.base_path = base_path
        self.annotation_dir = os.path.join(base_path, "annotations")
        self.template_dir = os.path.join(base_path, "template") 
        self.test_dir = os.path.join(base_path, "test")
        
        # DeepPCB defect type mapping      
        self.defect_types = {
            1: 'open_circuit',
            2: 'short', 
            3: 'mouse_bite',
            4: 'spur',
            5: 'spurious_copper',
            6: 'missing_hole'
        }
    
    def parse_annotation(self, annotation_path):
        """
        Parse annotation file in format: x1 y1 x2 y2 defect_type
        """
        defects = []
        
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) == 5:
                    x1, y1, x2, y2, defect_type = map(int, parts)
                    
                    # Convert to proper defect name
                    defect_name = self.defect_types.get(defect_type, f"unknown_{defect_type}")
                    
                    defects.append({
                        'type': defect_name,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
        return defects
    
    def extract_roi(self, image, bbox, margin=5, target_size=(128, 128)):
        """
        Extract and resize Region of Interest
        """
        x1, y1, x2, y2 = bbox
        
        # Add margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        
        # Resize to target size
        if roi.size > 0:
            roi = cv2.resize(roi, target_size)
            return roi
        return None
    
    def create_dataset(self, output_dir="dataset", image_size=128, train_ratio=0.7, val_ratio=0.15):
        """
        Create complete dataset from annotations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all annotation files
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.txt')]
        
        print(f"Found {len(annotation_files)} annotation files")
        
        # Collect all defects
        all_defects = []
        defect_counts = {defect_type: 0 for defect_type in self.defect_types.values()}
        
        print("Extracting defects from annotations...")
        for ann_file in tqdm(annotation_files):
            ann_path = os.path.join(self.annotation_dir, ann_file)
            
            # Get corresponding test image name
            base_name = ann_file.replace('.txt', '_test.jpg')
            test_path = os.path.join(self.test_dir, base_name)
            
            if not os.path.exists(test_path):
                # Try different extensions
                test_path = test_path.replace('.jpg', '.png')
                if not os.path.exists(test_path):
                    continue
            
            # Load test image
            test_img = cv2.imread(test_path)
            if test_img is None:
                continue
            
            # Parse annotations
            defects = self.parse_annotation(ann_path)
            
            for defect in defects:
                roi = self.extract_roi(test_img, defect['bbox'], target_size=(image_size, image_size))
                if roi is not None:
                    defect['roi'] = roi
                    defect['source_file'] = base_name
                    all_defects.append(defect)
                    defect_counts[defect['type']] += 1
        
        print("\nDefect Distribution:")
        for defect_type, count in defect_counts.items():
            print(f"  {defect_type}: {count}")
        
        # Split dataset
        train_defects, temp_defects = train_test_split(
            all_defects, test_size=(1 - train_ratio), random_state=42, 
            stratify=[d['type'] for d in all_defects]
        )
        
        val_defects, test_defects = train_test_split(
            temp_defects, test_size=0.5, random_state=42,
            stratify=[d['type'] for d in temp_defects]
        )
        
        print(f"\nDataset Split:")
        print(f"  Train: {len(train_defects)} defects")
        print(f"  Val: {len(val_defects)} defects") 
        print(f"  Test: {len(test_defects)} defects")
        
        # Save datasets
        self._save_defects(train_defects, os.path.join(output_dir, "train"))
        self._save_defects(val_defects, os.path.join(output_dir, "val"))
        self._save_defects(test_defects, os.path.join(output_dir, "test"))
        
        return len(all_defects)
    
    def _save_defects(self, defects, output_dir):
        """
        Save defects to organized directory structure
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for defect_type in self.defect_types.values():
            os.makedirs(os.path.join(output_dir, defect_type), exist_ok=True)
        
        for i, defect in enumerate(defects):
            defect_type = defect['type']
            filename = f"{defect_type}_{defect['source_file']}_{i}.png"
            filepath = os.path.join(output_dir, defect_type, filename)
            cv2.imwrite(filepath, defect['roi'])

def main():
    """Main function to create dataset"""
    dataset = DeepPCBDataset()
    total_defects = dataset.create_dataset(output_dir="data/defect_dataset")
    print(f"\n✅ Dataset creation complete! Total defects: {total_defects}")

if __name__ == "__main__":
    main()