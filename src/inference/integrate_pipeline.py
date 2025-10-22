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
    """Complete PCB inspection pipeline with classification"""
    
    def __init__(self, model_path='model/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classification model
        self.classifier = PCBDefectClassifier(num_classes=6).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        
        # Class names
        self.class_names = {
            0: 'missing_hole', 1: 'mouse_bite', 2: 'open_circuit',
            3: 'short', 4: 'spur', 5: 'spurious_copper'
        }
    
    def preprocess_for_classification(self, roi, image_size=128):
        """Preprocess ROI for classification"""
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        roi = cv2.resize(roi, (image_size, image_size))
        roi = roi.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        roi = (roi - mean) / std
        
        roi = torch.from_numpy(roi).permute(2, 0, 1).unsqueeze(0).float()
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
        # Load images
        template_img = cv2.imread(template_path)
        test_img = cv2.imread(test_path)
        
        if template_img is None or test_img is None:
            return None
        
        # Align images
        aligned_test, _ = align_images(test_img, template_img)
        if aligned_test is None:
            aligned_test, _ = simple_alignment(test_img, template_img)
        
        # Detect defects
        defects_dict = xor_defect_detection(aligned_test, template_img)
        defect_mask = defects_dict['combined']
        
        # Find contours and bounding boxes
        contours, bounding_boxes = detect_contours(defect_mask)
        
        # Extract ROIs and classify
        defect_regions = extract_defect_regions(aligned_test, bounding_boxes)
        classifications = []
        
        for roi in defect_regions:
            classification = self.classify_defect(roi)
            classifications.append(classification)
        
        # Create annotated result
        result_img = aligned_test.copy()
        colors = {
            'missing_hole': (0, 0, 255), 'mouse_bite': (0, 255, 255),
            'open_circuit': (255, 0, 0), 'short': (255, 0, 255),
            'spur': (0, 255, 0), 'spurious_copper': (255, 165, 0)
        }
        
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            if i < len(classifications):
                class_info = classifications[i]
                color = colors.get(class_info['class_name'], (255, 255, 255))
                
                cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
                label = f"{class_info['class_name']} ({class_info['confidence']:.2f})"
                cv2.putText(result_img, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save result
        cv2.imwrite(output_path, result_img)
        
        return {
            'defect_count': len(contours),
            'classifications': classifications,
            'result_image': result_img,
            'bounding_boxes': bounding_boxes
        }

def main():
    """Example usage"""
    pipeline = CompletePCBPipeline()
    
    result = pipeline.run_complete_pipeline(
        template_path='data/interim/template/group00041_00041000_temp.jpg',
        test_path='data/interim/test/group00041_00041000_test.jpg',
        output_path='complete_inspection_result.jpg'
    )
    
    if result:
        print(f"Found {result['defect_count']} defects")
        for classification in result['classifications']:
            print(f"  {classification['class_name']} ({classification['confidence']:.3f})")

if __name__ == "__main__":
    main()