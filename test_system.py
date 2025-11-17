"""
System testing and validation
"""

import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.web_app.backend import get_backend
from src.utils.file_operations import load_config


def test_image_formats():
    """Test different image formats"""
    
    print("\n" + "="*60)
    print("IMAGE FORMAT TESTING")
    print("="*60)
    
    backend = get_backend()
    
    # Create test images
    test_formats = {
        'RGB': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'Grayscale': np.random.randint(0, 255, (640, 640), dtype=np.uint8),
        'RGBA': np.random.randint(0, 255, (640, 640, 4), dtype=np.uint8)
    }
    
    os.makedirs('temp', exist_ok=True)
    
    for format_name, img in test_formats.items():
        print(f"\nTesting {format_name} format...")
        
        # Save temp files
        if format_name == 'RGBA':
            cv2.imwrite('temp/test_rgba.png', img)
            template = cv2.imread('temp/test_rgba.png', cv2.IMREAD_GRAYSCALE)
        else:
            cv2.imwrite('temp/test_template.jpg', img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            template = cv2.imread('temp/test_template.jpg', cv2.IMREAD_GRAYSCALE)
        
        test_img = template.copy()
        
        try:
            result = backend.process_image_pair(template, test_img)
            if result['success']:
                print(f"  ✓ {format_name}: PASSED")
            else:
                print(f"  ✗ {format_name}: FAILED - {result['error']}")
        except Exception as e:
            print(f"  ✗ {format_name}: ERROR - {str(e)}")
    
    # Cleanup
    for file in Path('temp').glob('test_*'):
        file.unlink()


def test_edge_cases():
    """Test edge cases"""
    
    print("\n" + "="*60)
    print("EDGE CASE TESTING")
    print("="*60)
    
    backend = get_backend()
    
    test_cases = {
        'Identical images': (np.ones((640, 640), dtype=np.uint8) * 128,
                            np.ones((640, 640), dtype=np.uint8) * 128),
        'Black images': (np.zeros((640, 640), dtype=np.uint8),
                        np.zeros((640, 640), dtype=np.uint8)),
        'White images': (np.ones((640, 640), dtype=np.uint8) * 255,
                        np.ones((640, 640), dtype=np.uint8) * 255),
        'Different sizes': (np.random.randint(0, 255, (640, 640), dtype=np.uint8),
                           np.random.randint(0, 255, (800, 800), dtype=np.uint8))
    }
    
    for case_name, (template, test_img) in test_cases.items():
        print(f"\nTesting: {case_name}")
        
        # Save temp files
        cv2.imwrite('temp/test_template.jpg', template)
        cv2.imwrite('temp/test_img.jpg', test_img)
        
        try:
            result = backend.process_image_pair(template, test_img)
            if result['success']:
                print(f"  ✓ Processed successfully")
                print(f"    Defects found: {result['num_defects']}")
            else:
                print(f"  ⚠ Processing failed: {result['error']}")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")


def test_accuracy():
    """Test classification accuracy on test set"""
    
    print("\n" + "="*60)
    print("ACCURACY TESTING")
    print("="*60)
    
    backend = get_backend()
    config = load_config()
    
    # Load test set
    test_dir = Path(config['data']['splits_path']) / 'test'
    template_dir = test_dir / 'templates'
    test_img_dir = test_dir / 'test_images'
    labels_dir = test_dir / 'labels'
    
    template_files = sorted(list(template_dir.glob('*_temp.jpg')))[:20]
    
    total_gt_defects = 0
    total_pred_defects = 0
    total_correct_class = 0
    
    for template_file in tqdm(template_files, desc="Testing accuracy"):
        image_id = template_file.stem.replace('_temp', '')
        test_file = test_img_dir / f'{image_id}_test.jpg'
        label_file = labels_dir / f'{image_id}.txt'
        
        if test_file.exists() and label_file.exists():
            template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            test_img = cv2.imread(str(test_file), cv2.IMREAD_GRAYSCALE)
            
            result = backend.process_image_pair(template, test_img)
            
            if result['success']:
                # Parse ground truth
                with open(label_file, 'r') as f:
                    gt_defects = len(f.readlines())
                
                total_gt_defects += gt_defects
                total_pred_defects += result['num_defects']
    
    print(f"\nResults:")
    print(f"  Total Ground Truth Defects: {total_gt_defects}")
    print(f"  Total Predicted Defects: {total_pred_defects}")
    print(f"  Detection Rate: {total_pred_defects/max(total_gt_defects,1)*100:.1f}%")


def main():
    """Run all system tests"""
    
    print("="*60)
    print("SYSTEM TESTING")
    print("="*60)
    
    # Create temp directory
    os.makedirs('temp', exist_ok=True)
    
    # Run tests
    test_image_formats()
    test_edge_cases()
    test_accuracy()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()