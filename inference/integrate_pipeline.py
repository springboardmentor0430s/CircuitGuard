# inference/integrate_pipeline.py
import cv2
import numpy as np
import torch
import os
import sys

# Add the parent directory to Python path
sys.path.append('..')

from preprocessing.alignment import align_images, simple_alignment
from preprocessing.binary_defect_detection import xor_defect_detection
from preprocessing.contour_detection import detect_contours, extract_defect_regions
from model.efficientnet import PCBDefectClassifier

class CompletePCBPipeline:
    def __init__(self, model_path='model/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classification model
        self.classifier = PCBDefectClassifier(num_classes=6).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        
        # Class names
        self.class_names = {
            0: 'missing_hole',
            1: 'mouse_bite', 
            2: 'open_circuit',
            3: 'short',
            4: 'spur',
            5: 'spurious_copper'
        }
        
        print("✅ Complete PCB Pipeline Initialized")
    
    def preprocess_for_classification(self, roi, image_size=128):
        """Preprocess ROI for classification"""
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        roi = cv2.resize(roi, (image_size, image_size))
        roi = roi.astype(np.float32) / 255.0  # Ensure float32
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        roi = (roi - mean) / std
        
        roi = torch.from_numpy(roi).permute(2, 0, 1).unsqueeze(0).float()  # Added .float()
        return roi.to(self.device)
    
    def classify_defect(self, roi):
        """Classify a single defect ROI"""
        with torch.no_grad():
            inputs = self.preprocess_for_classification(roi)
            outputs = self.classifier(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            return {
                'class_id': prediction.item(),
                'class_name': self.class_names[prediction.item()],
                'confidence': confidence.item()
            }
    
    def run_complete_pipeline(self, template_path, test_path, output_path='annotated_result.jpg'):
        """Run complete PCB inspection pipeline"""
        print("🔄 Running Complete PCB Inspection Pipeline...")
        
        # Step 1: Load images
        template_img = cv2.imread(template_path)
        test_img = cv2.imread(test_path)
        
        if template_img is None or test_img is None:
            print("❌ Error loading images")
            return
        
        print("✅ Step 1: Images loaded")
        
        # Step 2: Align images
        aligned_test, _ = align_images(test_img, template_img)
        if aligned_test is None:
            aligned_test, _ = simple_alignment(test_img, template_img)
        
        print("✅ Step 2: Images aligned")
        
        # Step 3: Detect defects
        defects_dict = xor_defect_detection(aligned_test, template_img)
        defect_mask = defects_dict['combined']
        
        print("✅ Step 3: Defects detected")
        
        # Step 4: Find contours and bounding boxes
        contours, bounding_boxes = detect_contours(defect_mask)
        
        print(f"✅ Step 4: Found {len(contours)} defects")
        
        # Step 5: Extract ROIs and classify
        defect_regions = extract_defect_regions(aligned_test, bounding_boxes)
        classifications = []
        
        for i, roi in enumerate(defect_regions):
            classification = self.classify_defect(roi)
            classifications.append(classification)
            print(f"   Defect {i+1}: {classification['class_name']} "
                  f"(confidence: {classification['confidence']:.3f})")
        
        # Step 6: Create annotated result
        result_img = aligned_test.copy()
        
        # Draw bounding boxes with class labels
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            if i < len(classifications):
                class_info = classifications[i]
                
                # Choose color based on defect type
                colors = {
                    'missing_hole': (0, 0, 255),      # Red
                    'mouse_bite': (0, 255, 255),      # Yellow
                    'open_circuit': (255, 0, 0),      # Blue
                    'short': (255, 0, 255),           # Magenta
                    'spur': (0, 255, 0),              # Green
                    'spurious_copper': (255, 165, 0)  # Orange
                }
                
                color = colors.get(class_info['class_name'], (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
                
                # Add class label
                label = f"{class_info['class_name']} ({class_info['confidence']:.2f})"
                cv2.putText(result_img, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save result
        cv2.imwrite(output_path, result_img)
        print(f"✅ Step 5: Annotated result saved to {output_path}")
        
        return {
            'defect_count': len(contours),
            'classifications': classifications,
            'result_image': result_img,
            'bounding_boxes': bounding_boxes
        }

def main():
    # Example usage
    pipeline = CompletePCBPipeline()
    
    # Run complete pipeline
    result = pipeline.run_complete_pipeline(
        template_path='data/interim/template/group00041_00041000_temp.jpg',
        test_path='data/interim/test/group00041_00041000_test.jpg',
        output_path='complete_inspection_result.jpg'
    )
    
    print(f"\n🎉 Pipeline completed!")
    print(f"   Defects found: {result['defect_count']}")
    print(f"   Classifications: {[c['class_name'] for c in result['classifications']]}")

if __name__ == "__main__":
    main()