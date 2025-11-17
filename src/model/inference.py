"""
Inference pipeline for defect detection and classification
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data_preparation.preprocessing import ImagePreprocessor
from src.data_preparation.image_alignment import ImageAligner
from src.data_preparation.image_subtraction import ImageSubtractor
from src.defect_detection.thresholding import DefectThresholder
from src.defect_detection.morphological_ops import MorphologicalProcessor
from src.defect_detection.contour_extraction import ContourExtractor


class DefectPredictor:
    """
    End-to-end defect detection and classification
    """
    
    def __init__(self, model: torch.nn.Module, 
                 config: dict,
                 device: torch.device):
        """
        Initialize predictor
        
        Args:
            model: Trained classification model
            config: Configuration dictionary
            device: Device to run on
        """
        self.model = model
        self.config = config
        self.device = device
        self.class_names = config['class_names']
        
        # Initialize detection components
        self.preprocessor = ImagePreprocessor(grayscale=True, normalize=False)
        self.aligner = ImageAligner(
            method=config['alignment']['method'],
            max_features=config['alignment']['max_features']
        )
        self.subtractor = ImageSubtractor(
            method=config['subtraction']['method']
        )
        self.thresholder = DefectThresholder(
            method=config['thresholding']['method']
        )
        self.morph_processor = MorphologicalProcessor(
            erosion_kernel=tuple(config['morphology']['erosion_kernel']),
            dilation_kernel=tuple(config['morphology']['dilation_kernel'])
        )
        self.contour_extractor = ContourExtractor(
            min_area=config['contours']['min_area'],
            max_area=config['contours']['max_area']
        )
        
        # Classification transform
        self.transform = A.Compose([
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
        
        self.model.eval()
    
    def detect_defects(self, template_path: str, 
                      test_path: str) -> Dict:
        """
        Detect defects in a PCB image pair
        
        Args:
            template_path: Path to template image
            test_path: Path to test image
            
        Returns:
            Dictionary with detection results
        """
        # Load images
        template = self.preprocessor.load_image(template_path)
        test_img = self.preprocessor.load_image(test_path)
        
        # Align
        try:
            aligned, align_info = self.aligner.align_image(template, test_img)
        except Exception as e:
            return {
                'success': False,
                'error': f'Alignment failed: {str(e)}',
                'defects': []
            }
        
        # Compute difference
        diff_map = self.subtractor.compute_difference_map(template, aligned)
        
        # Threshold
        binary_mask, _ = self.thresholder.threshold_difference_map(diff_map)
        
        # Refine mask
        refined_mask = self.morph_processor.refine_mask(
            binary_mask, 
            min_area=self.config['contours']['min_area']
        )
        
        # Extract contours
        contours = self.contour_extractor.extract_contours(refined_mask)
        properties = self.contour_extractor.get_contour_properties(contours)
        
        # Extract ROIs
        rois = self.contour_extractor.extract_rois(test_img, contours, padding=10)
        
        return {
            'success': True,
            'template': template,
            'test': test_img,
            'aligned': aligned,
            'diff_map': diff_map,
            'mask': refined_mask,
            'contours': contours,
            'properties': properties,
            'rois': rois,
            'num_defects': len(contours),
            'align_info': align_info
        }
    
    def classify_defect(self, roi: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Classify a single defect ROI
        
        Args:
            roi: Defect ROI image
            
        Returns:
            Tuple of (predicted class, confidence, all probabilities)
        """
        # Resize to model input size
        roi_resized = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_LINEAR)
        
        # Add channel dimension
        roi_resized = np.expand_dims(roi_resized, axis=-1)
        
        # Apply transform
        transformed = self.transform(image=roi_resized)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]
        
        return predicted_class, confidence_score, all_probs
    
    def predict(self, template_path: str, test_path: str) -> Dict:
        """
        Complete prediction pipeline
        
        Args:
            template_path: Path to template image
            test_path: Path to test image
            
        Returns:
            Dictionary with all results
        """
        # Detect defects
        detection_result = self.detect_defects(template_path, test_path)
        
        if not detection_result['success']:
            return detection_result
        
        # Classify each defect
        classifications = []
        
        for i, roi in enumerate(detection_result['rois']):
            pred_class, confidence, probs = self.classify_defect(roi)
            
            classifications.append({
                'defect_id': i,
                'predicted_class': pred_class,
                'predicted_label': self.class_names[pred_class],
                'confidence': float(confidence),
                'probabilities': probs.tolist(),
                'bbox': detection_result['properties'][i]['bbox'],
                'area': detection_result['properties'][i]['area'],
                'centroid': detection_result['properties'][i]['centroid']
            })
        
        detection_result['classifications'] = classifications
        
        return detection_result
    
    def annotate_image(self, image: np.ndarray, 
                      classifications: List[Dict]) -> np.ndarray:
        """
        Annotate image with predictions
        
        Args:
            image: Input image
            classifications: List of classification results
            
        Returns:
            Annotated image
        """
        # Convert to BGR for color
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()
        
        # Color map for different classes
        colors = [
            (255, 0, 0),    # mousebite - blue
            (0, 255, 0),    # open - green
            (0, 0, 255),    # short - red
            (255, 255, 0),  # spur - cyan
            (255, 0, 255),  # copper - magenta
            (0, 255, 255)   # pin-hole - yellow
        ]
        
        for defect in classifications:
            x, y, w, h = defect['bbox']
            pred_class = defect['predicted_class']
            confidence = defect['confidence']
            label = defect['predicted_label']
            
            color = colors[pred_class % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            text = f"{label}: {confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(annotated, (x, y - text_h - 5), 
                         (x + text_w, y), color, -1)
            cv2.putText(annotated, text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated


def compare_with_ground_truth(predictions: List[Dict], 
                              label_path: str,
                              config: dict) -> Dict:
    """
    Compare predictions with ground truth labels
    
    Args:
        predictions: List of predictions
        label_path: Path to ground truth label file
        config: Configuration dictionary
        
    Returns:
        Comparison statistics
    """
    if not os.path.exists(label_path):
        return {'has_ground_truth': False}
    
    # Parse ground truth
    ground_truth = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                ground_truth.append({
                    'x1': int(parts[0]),
                    'y1': int(parts[1]),
                    'x2': int(parts[2]),
                    'y2': int(parts[3]),
                    'class_id': int(parts[4])
                })
    
    # Convert class IDs (1-6) to class indices (0-5)
    label_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    
    # Match predictions to ground truth
    matched = 0
    class_matches = 0
    
    for gt in ground_truth:
        gt_center_x = (gt['x1'] + gt['x2']) // 2
        gt_center_y = (gt['y1'] + gt['y2']) // 2
        gt_class_idx = label_to_idx.get(gt['class_id'], -1)
        
        # Find closest prediction
        min_dist = float('inf')
        closest_pred = None
        
        for pred in predictions:
            pred_center = pred['centroid']
            dist = np.sqrt((pred_center[0] - gt_center_x)**2 + 
                          (pred_center[1] - gt_center_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_pred = pred
        
        # If prediction is close enough (within 50 pixels), consider it matched
        if closest_pred and min_dist < 50:
            matched += 1
            if closest_pred['predicted_class'] == gt_class_idx:
                class_matches += 1
    
    detection_rate = matched / len(ground_truth) if len(ground_truth) > 0 else 0
    classification_rate = class_matches / matched if matched > 0 else 0
    
    return {
        'has_ground_truth': True,
        'num_ground_truth': len(ground_truth),
        'num_predictions': len(predictions),
        'num_matched': matched,
        'num_class_correct': class_matches,
        'detection_rate': detection_rate,
        'classification_accuracy': classification_rate,
        'precision': matched / len(predictions) if len(predictions) > 0 else 0
    }