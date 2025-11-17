"""
Debug script to understand label file format
"""

import os
from pathlib import Path
from src.utils.file_operations import load_config

def inspect_label_files():
    """
    Inspect label files to understand their format
    """
    config = load_config()
    splits_path = config['data']['splits_path']
    
    print("="*60)
    print("INSPECTING LABEL FILES")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} Split:")
        print("-"*60)
        
        labels_dir = os.path.join(splits_path, split, 'labels')
        
        # Check if directory exists
        if not os.path.exists(labels_dir):
            print(f"  ✗ Labels directory doesn't exist: {labels_dir}")
            continue
        
        # Get all label files
        label_files = list(Path(labels_dir).glob('*'))
        print(f"  Total label files: {len(label_files)}")
        
        if len(label_files) == 0:
            print(f"  ✗ No label files found!")
            continue
        
        # Inspect first 5 label files
        print(f"\n  Inspecting first 5 label files:")
        for i, label_file in enumerate(label_files[:5]):
            print(f"\n  File {i+1}: {label_file.name}")
            print(f"  Full path: {label_file}")
            print(f"  Size: {label_file.stat().st_size} bytes")
            
            # Read content
            try:
                with open(label_file, 'r') as f:
                    content = f.read()
                
                if not content.strip():
                    print(f"  Content: EMPTY FILE")
                else:
                    lines = content.strip().split('\n')
                    print(f"  Number of lines: {len(lines)}")
                    print(f"  First few lines:")
                    for line in lines[:3]:
                        print(f"    '{line}'")
            except Exception as e:
                print(f"  Error reading file: {e}")
        
        # Check corresponding images
        images_dir = os.path.join(splits_path, split, 'test_images')
        if os.path.exists(images_dir):
            image_files = list(Path(images_dir).glob('*_test.jpg'))
            print(f"\n  Corresponding test images: {len(image_files)}")
            
            # Check naming match
            if len(image_files) > 0 and len(label_files) > 0:
                sample_image = image_files[0].stem.replace('_test', '')
                sample_label = label_files[0].stem
                print(f"\n  Sample image ID: {sample_image}")
                print(f"  Sample label ID: {sample_label}")
                print(f"  Match: {sample_image == sample_label}")

def check_original_labels():
    """
    Check the original PCBData label format
    """
    print("\n" + "="*60)
    print("CHECKING ORIGINAL PCBData LABELS")
    print("="*60)
    
    config = load_config()
    pcb_data_path = config['data']['raw_pcb_path']
    
    # Check a sample group
    from src.utils.file_operations import get_all_groups
    groups = get_all_groups(pcb_data_path)
    
    if not groups:
        print("No groups found!")
        return
    
    sample_group = groups[0]
    print(f"\nInspecting group: {sample_group}")
    
    # Label folder
    label_folder = os.path.join(
        pcb_data_path,
        sample_group,
        sample_group.replace('group', '') + '_not'
    )
    
    print(f"Label folder: {label_folder}")
    print(f"Exists: {os.path.exists(label_folder)}")
    
    if os.path.exists(label_folder):
        label_files = list(Path(label_folder).glob('*'))[:5]
        print(f"Sample label files: {len(label_files)}")
        
        for label_file in label_files:
            print(f"\n  File: {label_file.name}")
            print(f"  Size: {label_file.stat().st_size} bytes")
            
            with open(label_file, 'r') as f:
                content = f.read()
            
            if content.strip():
                print(f"  Content preview:\n{content[:200]}")
            else:
                print(f"  Content: EMPTY")

if __name__ == "__main__":
    inspect_label_files()
    check_original_labels()