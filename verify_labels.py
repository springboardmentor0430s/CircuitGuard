"""
Verify that labels were copied correctly
"""

import os
from pathlib import Path
from src.utils.file_operations import load_config

def verify_labels():
    """
    Check if labels exist in splits
    """
    config = load_config()
    splits_path = config['data']['splits_path']
    
    print("="*60)
    print("VERIFYING LABEL FILES IN SPLITS")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        
        labels_dir = os.path.join(splits_path, split, 'labels')
        images_dir = os.path.join(splits_path, split, 'test_images')
        
        if not os.path.exists(labels_dir):
            print(f"  ✗ Labels directory doesn't exist")
            continue
        
        label_files = list(Path(labels_dir).glob('*.txt'))
        image_files = list(Path(images_dir).glob('*_test.jpg'))
        
        print(f"  Label files: {len(label_files)}")
        print(f"  Image files: {len(image_files)}")
        print(f"  Coverage: {len(label_files)/len(image_files)*100:.1f}%")
        
        if len(label_files) > 0:
            # Check first label file
            sample_label = label_files[0]
            print(f"\n  Sample label: {sample_label.name}")
            with open(sample_label, 'r') as f:
                content = f.read()
            lines = content.strip().split('\n')
            print(f"  Number of defects: {len(lines)}")
            print(f"  First defect: {lines[0] if lines else 'N/A'}")

if __name__ == "__main__":
    verify_labels()