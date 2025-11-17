"""
Extract and organize defect ROIs for classification training
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import shutil

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.file_operations import load_config, create_directory


class ROIExtractor:
    """
    Extracts defect ROIs from PCB images using label files
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize ROI extractor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.splits_path = self.config['data']['splits_path']
        self.roi_dataset_path = self.config['data']['roi_dataset_path']
        self.roi_size = tuple(self.config['preprocessing']['roi_size'])
        
        # DeepPCB uses 1-6 class IDs
        self.label_to_class = self.config['classes']  # 1-6 mapping
        self.class_names = self.config['class_names']  # 0-5 for model
        
        # Create class directories
        for split in ['train', 'val', 'test']:
            for class_name in self.class_names:
                class_dir = os.path.join(self.roi_dataset_path, split, class_name)
                create_directory(class_dir)
    
    def parse_label_file(self, label_path: str) -> List[Dict]:
        """
        Parse DeepPCB label file
        
        Label format: x1 y1 x2 y2 class_id
        (absolute pixel coordinates, class_id from 1-6)
        
        Args:
            label_path: Path to label file
            
        Returns:
            List of defect dictionaries
        """
        if not os.path.exists(label_path):
            return []
        
        defects = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    x1 = int(parts[0])
                    y1 = int(parts[1])
                    x2 = int(parts[2])
                    y2 = int(parts[3])
                    class_id = int(parts[4])
                    
                    # Only process valid class IDs (1-6)
                    if class_id in self.label_to_class:
                        defects.append({
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'class_id': class_id,
                            'class_name': self.label_to_class[class_id]
                        })
        except Exception as e:
            print(f"Error parsing {label_path}: {e}")
            return []
        
        return defects
    
    def extract_roi(self, image: np.ndarray, defect: Dict, 
                    padding: int = 10) -> np.ndarray:
        """
        Extract ROI from image with padding
        
        Args:
            image: Source image
            defect: Defect dictionary with x1, y1, x2, y2
            padding: Padding around bbox
            
        Returns:
            Extracted and resized ROI
        """
        x1, y1, x2, y2 = defect['x1'], defect['y1'], defect['x2'], defect['y2']
        img_h, img_w = image.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        
        # Resize to standard size
        if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
            roi = cv2.resize(roi, self.roi_size, interpolation=cv2.INTER_LINEAR)
            return roi
        
        return None
    
    def process_split(self, split: str) -> Dict:
        """
        Process one split (train/val/test)
        
        Args:
            split: Split name
            
        Returns:
            Statistics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        
        # Paths
        images_dir = os.path.join(self.splits_path, split, 'test_images')
        labels_dir = os.path.join(self.splits_path, split, 'labels')
        
        # Check if directories exist
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return None
        
        if not os.path.exists(labels_dir):
            print(f"Labels directory not found: {labels_dir}")
            return None
        
        # Get all image files
        image_files = sorted(list(Path(images_dir).glob('*_test.jpg')))
        
        stats = {
            'total_images': len(image_files),
            'images_with_labels': 0,
            'total_defects': 0,
            'class_counts': {name: 0 for name in self.class_names}
        }
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Extracting {split} ROIs"):
            # Get corresponding label file
            image_id = img_path.stem.replace('_test', '')
            label_path = os.path.join(labels_dir, f"{image_id}.txt")
            
            # Parse labels
            defects = self.parse_label_file(label_path)
            
            if len(defects) == 0:
                continue
            
            stats['images_with_labels'] += 1
            
            # Load image
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            
            # Extract each defect ROI
            for idx, defect in enumerate(defects):
                class_name = defect['class_name']
                
                # Extract ROI
                roi = self.extract_roi(image, defect, padding=10)
                
                if roi is None:
                    continue
                
                # Save ROI
                output_dir = os.path.join(self.roi_dataset_path, split, class_name)
                output_path = os.path.join(output_dir, f"{image_id}_{idx:03d}.jpg")
                cv2.imwrite(output_path, roi)
                
                # Update stats
                stats['total_defects'] += 1
                stats['class_counts'][class_name] += 1
        
        # Print stats
        print(f"\n{split.upper()} Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Images with labels: {stats['images_with_labels']}")
        print(f"  Total defects extracted: {stats['total_defects']}")
        print(f"  Per-class counts:")
        for class_name, count in stats['class_counts'].items():
            print(f"    {class_name}: {count}")
        
        return stats
    
    def create_dataset(self):
        """
        Create complete ROI dataset for all splits
        """
        print("="*60)
        print("CREATING ROI DATASET FOR CLASSIFICATION")
        print("="*60)
        
        all_stats = {}
        
        # Process each split
        for split in ['train', 'val', 'test']:
            stats = self.process_split(split)
            if stats:
                all_stats[split] = stats
        
        if not all_stats:
            print("\n✗ No data processed!")
            return None
        
        # Create summary DataFrame
        summary_data = []
        for split, stats in all_stats.items():
            for class_name, count in stats['class_counts'].items():
                summary_data.append({
                    'split': split,
                    'class': class_name,
                    'count': count
                })
        
        summary_df = pd.DataFrame(summary_data)
        pivot_df = summary_df.pivot(index='class', columns='split', values='count').fillna(0).astype(int)
        
        print("\n" + "="*60)
        print("DATASET CREATION COMPLETE")
        print("="*60)
        print("\nClass distribution across splits:")
        print(pivot_df)
        print(f"\nTotal defects: {summary_df['count'].sum()}")
        
        # Save to CSV
        stats_path = os.path.join(self.roi_dataset_path, 'dataset_statistics.csv')
        pivot_df.to_csv(stats_path)
        print(f"\nStatistics saved to: {stats_path}")
        
        return all_stats


def main():
    """
    Main function to create ROI dataset
    """
    extractor = ROIExtractor()
    stats = extractor.create_dataset()


if __name__ == "__main__":
    main()