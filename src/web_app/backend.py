"""
Backend API for web application
"""

import os
import cv2
import torch
import numpy as np
from typing import Dict, Tuple, Optional
import time
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.model.inference import DefectPredictor
from src.model.evaluator import load_best_model
from src.utils.file_operations import load_config


class PCBInspectionBackend:
    """
    Backend service for PCB defect detection web app
    """
    
    def __init__(self):
        """
        Initialize backend service
        """
        print("Initializing PCB Inspection Backend...")
        
        # Load configuration
        self.config = load_config()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading trained model...")
        self.model = load_best_model(self.config, self.device)
        
        # Create predictor
        self.predictor = DefectPredictor(
            model=self.model,
            config=self.config,
            device=self.device
        )
        
        self.class_names = self.config['class_names']
        
        print("Backend initialized successfully!")
    
    def process_image_pair(self, template_image: np.ndarray, 
                          test_image: np.ndarray) -> Dict:
        """
        Process an image pair and return results
        
        Args:
            template_image: Template image array
            test_image: Test image array
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Save temp files (predictor expects file paths)
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            template_path = os.path.join(temp_dir, "template_temp.jpg")
            test_path = os.path.join(temp_dir, "test_temp.jpg")
            
            cv2.imwrite(template_path, template_image)
            cv2.imwrite(test_path, test_image)
            
            # Run prediction
            result = self.predictor.predict(template_path, test_path)
            
            # Clean up temp files
            os.remove(template_path)
            os.remove(test_path)
            
            if not result['success']:
                return {
                    'success': False,
                    'error': result['error'],
                    'processing_time': time.time() - start_time
                }
            
            # Create annotated image
            annotated = self.predictor.annotate_image(
                result['test'],
                result['classifications']
            )
            
            # Prepare response
            response = {
                'success': True,
                'processing_time': time.time() - start_time,
                'num_defects': result['num_defects'],
                'classifications': result['classifications'],
                'images': {
                    'template': result['template'],
                    'test': result['test'],
                    'aligned': result['aligned'],
                    'difference_map': result['diff_map'],
                    'mask': result['mask'],
                    'annotated': annotated
                },
                'alignment_info': result['align_info']
            }
            
            return response
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def get_class_statistics(self, classifications: list) -> Dict:
        """
        Get statistics about detected defects
        
        Args:
            classifications: List of classification results
            
        Returns:
            Statistics dictionary
        """
        if not classifications:
            return {class_name: 0 for class_name in self.class_names}
        
        stats = {class_name: 0 for class_name in self.class_names}
        
        for defect in classifications:
            label = defect['predicted_label']
            stats[label] += 1
        
        return stats
    
    def format_results_for_display(self, result: Dict) -> Dict:
        """
        Format results for web display
        
        Args:
            result: Raw prediction result
            
        Returns:
            Formatted result dictionary
        """
        if not result['success']:
            return result
        
        # Get class statistics
        class_stats = self.get_class_statistics(result['classifications'])
        
        # Calculate average confidence
        avg_confidence = np.mean([
            d['confidence'] for d in result['classifications']
        ]) if result['classifications'] else 0.0
        
        # Format classification details
        defect_details = []
        for i, defect in enumerate(result['classifications']):
            defect_details.append({
                'id': i + 1,
                'type': defect['predicted_label'],
                'confidence (%)': round(defect['confidence']*100, 1),  # ← Keep as number
                'location': f"({defect['centroid'][0]}, {defect['centroid'][1]})",
                'area (px²)': defect['area'],  # ← Also keep as number
                'bbox': defect['bbox']
            })
        
        formatted = {
            'success': True,
            'processing_time': f"{result['processing_time']:.2f}s",
            'summary': {
                'total_defects': result['num_defects'],
                'average_confidence': f"{avg_confidence*100:.1f}%",
                'alignment_matches': result['alignment_info']['num_matches'],
                'alignment_inliers': result['alignment_info']['num_inliers']
            },
            'class_distribution': class_stats,
            'defect_details': defect_details,
            'images': result['images']
        }
        
        return formatted


# Global backend instance (singleton)
_backend_instance = None

def get_backend() -> PCBInspectionBackend:
    """
    Get or create backend instance (singleton pattern)
    
    Returns:
        Backend instance
    """
    global _backend_instance
    
    if _backend_instance is None:
        _backend_instance = PCBInspectionBackend()
    
    return _backend_instance