# inference/predict.py
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append('.')

from model.efficientnet import PCBDefectClassifier

class PCBDefectPredictor:
    def __init__(self, model_path='model/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Class names (match training)
        self.class_names = {
            0: 'missing_hole',
            1: 'mouse_bite', 
            2: 'open_circuit',
            3: 'short',
            4: 'spur',
            5: 'spurious_copper'
        }
        
        # Load model
        self.model = PCBDefectClassifier(num_classes=6).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Loaded model from {model_path}")
        print(f"📊 Model accuracy during training: {checkpoint['accuracy']:.2f}%")
    
    def preprocess_image(self, image, image_size=128):
        """Preprocess image for inference"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (image_size, image_size))
        
        # Normalize (same as training) and ensure float32
        image = image.astype(np.float32) / 255.0  # Ensure float32
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Convert to tensor and ensure float32
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # Added .float() here
        return image.to(self.device)
    
    def predict_single(self, image):
        """Predict defect type for a single image"""
        with torch.no_grad():
            inputs = self.preprocess_image(image)
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            return {
                'class_id': prediction.item(),
                'class_name': self.class_names[prediction.item()],
                'confidence': confidence.item(),
                'all_probabilities': probabilities.cpu().numpy()[0]
            }
    
    def predict_batch(self, image_list):
        """Predict defect types for multiple images"""
        results = []
        for image in image_list:
            results.append(self.predict_single(image))
        return results
    
    def visualize_prediction(self, image, prediction, save_path=None):
        """Visualize prediction with annotation"""
        # Create copy for visualization
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Add prediction text
        class_name = prediction['class_name']
        confidence = prediction['confidence']
        
        text = f"{class_name} ({confidence:.2f})"
        
        # Choose color based on confidence
        if confidence > 0.9:
            color = (0, 255, 0)  # Green
        elif confidence > 0.7:
            color = (255, 255, 0)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Add text to image
        cv2.putText(vis_image, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add confidence bar
        bar_width = int(confidence * 100)
        cv2.rectangle(vis_image, (10, 40), (10 + bar_width, 50), color, -1)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def process_test_images(self, test_dir='data/defect_dataset/test', output_dir='annotated_results'):
        """Process all test images and generate annotated results"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔍 Processing test images from {test_dir}")
        
        # Get all test images
        all_images = []
        image_paths = []
        
        for defect_type in os.listdir(test_dir):
            defect_path = os.path.join(test_dir, defect_type)
            if not os.path.isdir(defect_path):
                continue
                
            for img_file in os.listdir(defect_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(defect_path, img_file)
                    image = cv2.imread(img_path)
                    if image is not None:
                        all_images.append(image)
                        image_paths.append((img_path, defect_type))
        
        print(f"📁 Found {len(all_images)} test images")
        
        # Run predictions
        predictions = self.predict_batch(all_images)
        
        # Generate annotated images and statistics
        correct_predictions = 0
        results = []
        
        for i, (prediction, (img_path, true_class)) in enumerate(zip(predictions, image_paths)):
            # Check if prediction is correct
            is_correct = (prediction['class_name'] == true_class)
            if is_correct:
                correct_predictions += 1
            
            # Save annotated image
            base_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"pred_{base_name}")
            annotated_img = self.visualize_prediction(all_images[i], prediction, output_path)
            
            results.append({
                'image': base_name,
                'true_class': true_class,
                'predicted_class': prediction['class_name'],
                'confidence': prediction['confidence'],
                'correct': is_correct
            })
        
        # Calculate accuracy
        accuracy = (correct_predictions / len(all_images)) * 100
        
        print(f"\n📊 Test Results:")
        print(f"   Total Images: {len(all_images)}")
        print(f"   Correct Predictions: {correct_predictions}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Annotated images saved to: {output_dir}")
        
        # Save results to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('prediction_results.csv', index=False)
        
        # Generate summary
        self.generate_prediction_summary(results, accuracy)
        
        return results, accuracy
    
    def generate_prediction_summary(self, results, accuracy):
        """Generate prediction summary report"""
        summary = {
            'total_predictions': len(results),
            'correct_predictions': sum(1 for r in results if r['correct']),
            'accuracy': accuracy,
            'average_confidence': np.mean([r['confidence'] for r in results]),
            'class_accuracy': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate class-wise accuracy
        for class_name in self.class_names.values():
            class_results = [r for r in results if r['true_class'] == class_name]
            if class_results:
                class_correct = sum(1 for r in class_results if r['correct'])
                summary['class_accuracy'][class_name] = {
                    'total': len(class_results),
                    'correct': class_correct,
                    'accuracy': (class_correct / len(class_results)) * 100
                }
        
        # Save summary
        import json
        with open('prediction_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Prediction Summary:")
        print(f"   Overall Accuracy: {accuracy:.2f}%")
        print(f"   Average Confidence: {summary['average_confidence']:.3f}")
        print("   Class-wise Performance:")
        for class_name, stats in summary['class_accuracy'].items():
            print(f"     {class_name:15}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

def main():
    # Example usage
    predictor = PCBDefectPredictor()
    
    # Process all test images
    results, accuracy = predictor.process_test_images()
    
    # Example single prediction
    print("\n🎯 Single Prediction Example:")
    sample_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    prediction = predictor.predict_single(sample_image)
    print(f"   Predicted: {prediction['class_name']}")
    print(f"   Confidence: {prediction['confidence']:.3f}")

if __name__ == "__main__":
    main()