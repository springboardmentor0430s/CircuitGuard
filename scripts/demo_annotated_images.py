import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append('src')

from preprocessing.alignment import align_images, simple_alignment
from preprocessing.binary_defect_detection import xor_defect_detection
from preprocessing.contour_detection import detect_contours, extract_defect_regions
from inference.predict import PCBDefectPredictor

class PCBAnnotatedDemo:
    def __init__(self, model_path='model/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classifier
        self.predictor = PCBDefectPredictor(model_path)
        
        print("✅ PCB Annotation Demo Initialized")
    
    def process_image_pair(self, template_path, test_path, output_path=None):
        """Process a template-test pair and create annotated result"""
        print(f"🔍 Processing: {os.path.basename(test_path)}")
        
        # Load images
        template_img = cv2.imread(template_path)
        test_img = cv2.imread(test_path)
        
        if template_img is None or test_img is None:
            print(f"❌ Error loading images: {template_path}, {test_path}")
            return None
        
        # Step 1: Align images
        print("   📐 Aligning images...")
        aligned_test, _ = align_images(test_img, template_img)
        if aligned_test is None:
            aligned_test, _ = simple_alignment(test_img, template_img)
        
        # Step 2: Detect defects
        print("   🔎 Detecting defects...")
        defects_dict = xor_defect_detection(aligned_test, template_img)
        defect_mask = defects_dict['combined']
        
        # Step 3: Find contours and bounding boxes
        contours, bounding_boxes = detect_contours(defect_mask)
        print(f"   📦 Found {len(bounding_boxes)} defects")
        
        # Step 4: Extract ROIs and classify defects
        defect_regions = extract_defect_regions(aligned_test, bounding_boxes)
        classifications = []
        
        for i, roi in enumerate(defect_regions):
            classification = self.predictor.predict_single(roi)
            classifications.append(classification)
            print(f"     Defect {i+1}: {classification['class_name']}")
        
        # Step 5: Create annotated image
        annotated_img = self.create_annotated_image(
            aligned_test, bounding_boxes, classifications
        )
        
        # Save result if output path provided
        if output_path:
            cv2.imwrite(output_path, annotated_img)
            print(f"   💾 Saved annotated image: {output_path}")
        
        return annotated_img, classifications
    
    def create_annotated_image(self, image, bounding_boxes, classifications):
        """Create annotated image with simple blue boxes and labels"""
        # Create copy for annotation
        annotated_img = image.copy()
        
        # Blue color for all defects
        color = (255, 0, 0)  # Blue in BGR
        
        # Draw each bounding box with classification
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            if i < len(classifications):
                class_info = classifications[i]
                defect_type = class_info['class_name']
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
                
                # Create label text (just defect name)
                label = f"{defect_type}"
                
                # Draw text
                cv2.putText(annotated_img, label, 
                           (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_img
    
    def display_comparison(self, original_img, annotated_img, title="PCB Defect Detection"):
        """Display original and annotated images side by side"""
        # Convert BGR to RGB for matplotlib
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(original_rgb)
        ax1.set_title('Original PCB Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(annotated_rgb)
        ax2.set_title('Detected Defects', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def run_demo(self, num_samples=3):
        """Run demo on sample images from the dataset"""
        print("🚀 Starting PCB Defect Detection Demo")
        print("=" * 50)
        
        # Find available test images
        test_dir = Path('data/interim/test')
        template_dir = Path('data/interim/template')
        
        if not test_dir.exists() or not template_dir.exists():
            print("❌ Test or template directory not found!")
            return
        
        # Get list of test images
        test_images = list(test_dir.glob('*_test.*'))[:num_samples]
        
        if not test_images:
            print("❌ No test images found!")
            return
        
        print(f"📁 Found {len(test_images)} test images for demo")
        
        results = []
        
        for test_path in test_images:
            # Find corresponding template
            template_name = test_path.name.replace('_test.', '_temp.')
            template_path = template_dir / template_name
            
            if not template_path.exists():
                print(f"❌ Template not found for {test_path.name}")
                continue
            
            # Process the image pair
            annotated_img, classifications = self.process_image_pair(
                str(template_path), str(test_path),
                output_path=f"demo_annotated_{test_path.stem}.png"
            )
            
            if annotated_img is not None:
                # Load original test image for comparison
                original_img = cv2.imread(str(test_path))
                
                # Display comparison
                self.display_comparison(original_img, annotated_img, 
                                      f"PCB Defects - {test_path.stem}")
                
                results.append({
                    'image': test_path.name,
                    'defects_found': len(classifications),
                    'defect_types': [c['class_name'] for c in classifications]
                })
        
        # Print summary
        print("\n📊 DEMO SUMMARY:")
        print("=" * 50)
        for result in results:
            print(f"📷 {result['image']}:")
            print(f"   Defects found: {result['defects_found']}")
            if result['defects_found'] > 0:
                for i, defect_type in enumerate(result['defect_types']):
                    print(f"     {i+1}. {defect_type}")
            print()

def main():
    # Initialize demo
    demo = PCBAnnotatedDemo()
    
    # Run demo on sample images
    demo.run_demo(num_samples=3)

if __name__ == "__main__":
    main()