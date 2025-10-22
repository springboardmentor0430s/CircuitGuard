import cv2
import os
import argparse
import glob
import numpy as np
import torch
import json
from datetime import datetime
from src.preprocessing.alignment import align_images, simple_alignment
from src.preprocessing.binary_defect_detection import xor_defect_detection
from src.preprocessing.contour_detection import detect_contours, extract_defect_regions
from model.efficientnet import PCBDefectClassifier

class CircuitGuardPipeline:
    """Complete PCB defect detection and classification pipeline"""
    
    def __init__(self, model_path='model/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classification model
        self.classifier = PCBDefectClassifier(num_classes=6).to(self.device)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.classifier.eval()
            print(f"✅ Model loaded: {checkpoint['accuracy']:.2f}% accuracy")
        else:
            print("⚠️ Model not found - classification disabled")
            self.classifier = None
        
        # Class names
        self.class_names = {
            0: 'missing_hole', 1: 'mouse_bite', 2: 'open_circuit',
            3: 'short', 4: 'spur', 5: 'spurious_copper'
        }
    
    def load_image_pair(self, template_dir, test_dir, image_name):
        """Load template-test image pair"""
        image_name = os.path.splitext(image_name)[0]
        
        # Find template file
        template_patterns = [
            os.path.join(template_dir, f"{image_name}_temp.*"),
            os.path.join(template_dir, f"*{image_name}*_temp.*")
        ]
        template_path = None
        for pattern in template_patterns:
            matches = glob.glob(pattern)
            if matches:
                template_path = matches[0]
                break
        
        # Find test file
        test_patterns = [
            os.path.join(test_dir, f"{image_name}_test.*"),
            os.path.join(test_dir, f"*{image_name}*_test.*")
        ]
        test_path = None
        for pattern in test_patterns:
            matches = glob.glob(pattern)
            if matches:
                test_path = matches[0]
                break
        
        return template_path, test_path
    
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
        if self.classifier is None:
            return {'class_id': -1, 'class_name': 'unknown'}
        
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

    def create_annotated_image(self, image, bounding_boxes, classifications):
        """Create annotated image with simple blue boxes and labels"""
        # Create copy for annotation
        annotated_img = image.copy()
        
        # Blue color for all defects (BGR)
        color = (255, 0, 0)
        
        # Draw each bounding box with classification
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            if i < len(classifications):
                class_info = classifications[i]
                defect_type = class_info.get('class_name', 'unknown')
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
                
                # Create label text (just defect name)
                label = f"{defect_type}"
                
                # Draw text (with a filled background for readability)
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x, y - text_h - baseline - 6), (x + text_w + 6, y), color, -1)
                cv2.putText(annotated_img, label, 
                           (x + 3, y - 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_img
    
    def process_image_pair(self, template_path, test_path, output_path=None):
        """Complete PCB inspection pipeline"""
        # Load images
        template_img = cv2.imread(template_path)
        test_img = cv2.imread(test_path)
        
        if template_img is None or test_img is None:
            return None
        
        print(f"Processing: {os.path.basename(template_path)}")
        
        # Align images
        aligned_test, _ = align_images(test_img, template_img)
        if aligned_test is None:
            aligned_test, _ = simple_alignment(test_img, template_img)
        
        # Detect defects
        defects_dict = xor_defect_detection(aligned_test, template_img)
        defect_mask = defects_dict['combined']
        
        # Find contours and bounding boxes
        contours, bounding_boxes = detect_contours(defect_mask)
        
        # Extract ROIs and classify defects
        defect_regions = extract_defect_regions(aligned_test, bounding_boxes)
        classifications = []
        
        for roi in defect_regions:
            classification = self.classify_defect(roi)
            classifications.append(classification)
        
        # Create annotated result (use helper to keep consistent style)
        result_img = self.create_annotated_image(aligned_test, bounding_boxes, classifications)
        
        # Save result if output path provided
        if output_path:
            cv2.imwrite(output_path, result_img)
        
        return {
            'template': template_img,
            'test': aligned_test,
            'defect_mask': defect_mask,
            'result': result_img,
            'contours': contours,
            'bounding_boxes': bounding_boxes,
            'classifications': classifications,
            'defect_count': len(contours)
        }

def load_all_pairs(template_dir, test_dir):
    """Load all template-test pairs"""
    image_pairs = []
    template_files = glob.glob(os.path.join(template_dir, "*_temp.*"))
    
    for template_path in template_files:
        template_filename = os.path.basename(template_path)
        base_name = template_filename.replace('_temp.', '_test.')
        test_path = os.path.join(test_dir, base_name)
        
        if not os.path.exists(test_path):
            # Try different extensions
            for ext in ['.jpg', '.png', '.jpeg']:
                test_path = test_path.replace('.jpg', ext).replace('.png', ext).replace('.jpeg', ext)
                if os.path.exists(test_path):
                    break
        
        if os.path.exists(test_path):
            image_pairs.append((template_path, test_path, template_filename))
    
    return image_pairs

def save_results_json(results, output_path):
    """Save results to JSON for frontend consumption"""
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'defect_count': results['defect_count'],
        'defects': []
    }
    
    for i, classification in enumerate(results['classifications']):
        if i < len(results['bounding_boxes']):
            x, y, w, h = results['bounding_boxes'][i]
            json_results['defects'].append({
                'id': i + 1,
                'class_name': classification['class_name'],
                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            })
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='CircuitGuard PCB Defect Detection Pipeline')
    parser.add_argument('--template_dir', type=str, default='data/interim/template', help='Template directory')
    parser.add_argument('--test_dir', type=str, default='data/interim/test', help='Test directory')
    parser.add_argument('--image_name', type=str, default=None, help='Specific image name to process')
    parser.add_argument('--all', action='store_true', help='Process all image pairs')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--model_path', type=str, default='model/best_model.pth', help='Path to trained model')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.template_dir):
        print(f"❌ Template directory not found: {args.template_dir}")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"❌ Test directory not found: {args.test_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = CircuitGuardPipeline(args.model_path)
    
    # Load image pairs
    if args.image_name:
        template_path, test_path = pipeline.load_image_pair(args.template_dir, args.test_dir, args.image_name)
        if template_path and test_path:
            image_pairs = [(template_path, test_path, os.path.basename(template_path))]
        else:
            print(f"❌ Image pair not found for: {args.image_name}")
            return
    elif args.all:
        image_pairs = load_all_pairs(args.template_dir, args.test_dir)
        if not image_pairs:
            print("❌ No image pairs found!")
            return
    else:
        print("❌ Please specify --image_name or --all")
        return
    
    print(f"🔍 Processing {len(image_pairs)} image pair(s)")
    
    # Process each pair
    for i, (template_path, test_path, filename) in enumerate(image_pairs):
        print(f"\n--- Processing {i+1}/{len(image_pairs)}: {filename} ---")
        
        # Generate output paths
        base_name = os.path.splitext(filename)[0].replace('_temp', '')
        result_img_path = os.path.join(args.output_dir, f"{base_name}_result.jpg")
        result_json_path = os.path.join(args.output_dir, f"{base_name}_results.json")
        
        # Process image pair
        results = pipeline.process_image_pair(template_path, test_path, result_img_path)
        
        if results:
            # Save JSON results for frontend
            save_results_json(results, result_json_path)
            
            print(f"✅ Found {results['defect_count']} defects")
            for j, classification in enumerate(results['classifications']):
                print(f"   Defect {j+1}: {classification['class_name']}")
            
            print(f"📁 Results saved: {result_img_path}, {result_json_path}")
        else:
            print(f"❌ Failed to process: {filename}")
    
    print(f"\n🎉 Processing complete! Results saved to: {args.output_dir}")

def process_for_frontend(template_path, test_path, model_path='model/best_model.pth'):
    """
    Frontend-friendly function for processing single image pair
    Returns structured data suitable for web application consumption
    """
    pipeline = CircuitGuardPipeline(model_path)
    results = pipeline.process_image_pair(template_path, test_path)
    
    if results is None:
        return None
    
    # Format for frontend consumption
    frontend_result = {
        'success': True,
        'defect_count': results['defect_count'],
        'defects': [],
        'processing_time': datetime.now().isoformat()
    }
    
    for i, classification in enumerate(results['classifications']):
        if i < len(results['bounding_boxes']):
            x, y, w, h = results['bounding_boxes'][i]
            frontend_result['defects'].append({
                'id': i + 1,
                'class_name': classification['class_name'],
                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            })
    
    return frontend_result

if __name__ == "__main__":
    main()