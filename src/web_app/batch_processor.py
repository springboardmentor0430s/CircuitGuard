"""
Batch processing for multiple PCB inspections
"""

import os
import cv2
import numpy as np
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.web_app.backend import get_backend


class BatchProcessor:
    """
    Handles batch processing of multiple PCB image pairs
    """
    
    def __init__(self):
        """Initialize batch processor"""
        self.backend = get_backend()
        self.results_history = []
    
    def process_batch(self, image_pairs: List[Dict]) -> List[Dict]:
        """
        Process multiple image pairs
        
        Args:
            image_pairs: List of dicts with 'template' and 'test' images
            
        Returns:
            List of results
        """
        results = []
        
        for i, pair in enumerate(image_pairs):
            result = self.backend.process_image_pair(
                pair['template'],
                pair['test']
            )
            
            result['pair_id'] = i
            result['timestamp'] = datetime.now().isoformat()
            
            if 'name' in pair:
                result['name'] = pair['name']
            
            results.append(result)
            self.results_history.append(result)
        
        return results
    
    def get_batch_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate statistics across batch
        
        Args:
            results: List of processing results
            
        Returns:
            Statistics dictionary
        """
        successful = [r for r in results if r['success']]
        
        if not successful:
            return {
                'total_processed': len(results),
                'successful': 0,
                'failed': len(results),
                'total_defects': 0,
                'average_defects': 0,
                'average_time': 0
            }
        
        total_defects = sum(r['num_defects'] for r in successful)
        avg_time = sum(r['processing_time'] for r in successful) / len(successful)
        
        return {
            'total_processed': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'total_defects': total_defects,
            'average_defects': total_defects / len(successful),
            'average_time': avg_time,
            'defect_distribution': self._get_defect_distribution(successful)
        }
    
    def _get_defect_distribution(self, results: List[Dict]) -> Dict:
        """Get defect type distribution across all results"""
        distribution = {name: 0 for name in self.backend.class_names}
        
        for result in results:
            for defect in result.get('classifications', []):
                label = defect['predicted_label']
                distribution[label] += 1
        
        return distribution
    
    def export_results(self, results: List[Dict], output_path: str):
        """
        Export results to JSON file
        
        Args:
            results: Processing results
            output_path: Path to save JSON
        """
        # Prepare exportable data (remove image arrays)
        exportable = []
        
        for result in results:
            if result['success']:
                export_item = {
                    'pair_id': result.get('pair_id'),
                    'name': result.get('name', 'unknown'),
                    'timestamp': result.get('timestamp'),
                    'processing_time': result['processing_time'],
                    'num_defects': result['num_defects'],
                    'classifications': result['classifications'],
                    'alignment_info': result['alignment_info']
                }
                exportable.append(export_item)
        
        with open(output_path, 'w') as f:
            json.dump(exportable, f, indent=4)