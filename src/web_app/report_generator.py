"""
Generate inspection reports
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List


class ReportGenerator:
    """
    Generates PDF reports for PCB inspections
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize report generator
        
        Args:
            class_names: List of defect class names
        """
        self.class_names = class_names
        self.colors = [
            (255, 0, 0),    # mousebite - blue
            (0, 255, 0),    # open - green
            (0, 0, 255),    # short - red
            (255, 255, 0),  # spur - cyan
            (255, 0, 255),  # copper - magenta
            (0, 255, 255)   # pin-hole - yellow
        ]
    
    def generate_single_report(self, result: Dict, output_path: str):
        """
        Generate PDF report for single inspection
        
        Args:
            result: Processing result dictionary
            output_path: Path to save PDF
        """
        if not result['success']:
            raise ValueError("Cannot generate report for failed inspection")
        
        try:
            # Ensure matplotlib figures are closed
            plt.close('all')
            
            with PdfPages(output_path) as pdf:
                # Page 1: Summary
                self._create_summary_page(result, pdf)
                
                # Page 2: Visualizations
                self._create_visualization_page(result, pdf)
                
                # Page 3: Defect details
                if result.get('classifications') and len(result['classifications']) > 0:
                    self._create_details_page(result, pdf)
            
            # Verify file was created
            if not os.path.exists(output_path):
                raise Exception("PDF file was not created")
                
        except Exception as e:
            raise Exception(f"Failed to generate PDF: {str(e)}")
        finally:
            plt.close('all')
    
    def _create_summary_page(self, result: Dict, pdf: PdfPages):
        """Create summary page"""
        try:
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor('white')
            
            # Title
            fig.text(0.5, 0.95, 'PCB Inspection Report', 
                    ha='center', fontsize=20, fontweight='bold')
            
            # Date
            fig.text(0.5, 0.90, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    ha='center', fontsize=10)
            
            # Summary box
            summary_text = f"""
INSPECTION SUMMARY
{'='*60}

Total Defects Detected: {result['num_defects']}
Processing Time: {result['processing_time']:.2f}s

Alignment Information:
  - Feature Matches: {result['alignment_info']['num_matches']}
  - RANSAC Inliers: {result['alignment_info']['num_inliers']}

Defect Classification:
"""
            
            # Add class distribution
            class_dist = {}
            for defect in result.get('classifications', []):
                label = defect['predicted_label']
                class_dist[label] = class_dist.get(label, 0) + 1
            
            for class_name in self.class_names:
                count = class_dist.get(class_name, 0)
                summary_text += f"  - {class_name}: {count}\n"
            
            fig.text(0.1, 0.75, summary_text, fontsize=10, family='monospace',
                    verticalalignment='top')
            
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            plt.close('all')
            raise Exception(f"Error creating summary page: {str(e)}")
    
    def _create_visualization_page(self, result: Dict, pdf: PdfPages):
        """Create visualization page"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            
            # Test image
            axes[0, 0].imshow(result['images']['test'], cmap='gray')
            axes[0, 0].set_title('Original Test Image', fontsize=10, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Difference map
            axes[0, 1].imshow(result['images']['difference_map'], cmap='hot')
            axes[0, 1].set_title('Difference Map', fontsize=10, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Defect mask
            axes[1, 0].imshow(result['images']['mask'], cmap='gray')
            axes[1, 0].set_title(f'Defect Mask ({result["num_defects"]} defects)', 
                               fontsize=10, fontweight='bold')
            axes[1, 0].axis('off')
            
            # Annotated result
            annotated_rgb = cv2.cvtColor(result['images']['annotated'], cv2.COLOR_BGR2RGB)
            axes[1, 1].imshow(annotated_rgb)
            axes[1, 1].set_title('Classified Defects', fontsize=10, fontweight='bold')
            axes[1, 1].axis('off')
            
            plt.suptitle('Processing Pipeline Visualization', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            plt.close('all')
            raise Exception(f"Error creating visualization page: {str(e)}")
    
    def _create_details_page(self, result: Dict, pdf: PdfPages):
        """Create defect details page"""
        try:
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor('white')
            
            fig.text(0.5, 0.95, 'Defect Details', 
                    ha='center', fontsize=16, fontweight='bold')
            
            details_text = "DETECTED DEFECTS\n" + "="*60 + "\n\n"
            
            for i, defect in enumerate(result.get('classifications', [])):
                details_text += f"Defect #{i+1}:\n"
                details_text += f"  Type: {defect['predicted_label']}\n"
                details_text += f"  Confidence: {defect['confidence']*100:.1f}%\n"
                details_text += f"  Location (x, y): {defect['centroid']}\n"
                details_text += f"  Area: {defect['area']} pixels\n"
                details_text += f"  Bounding Box: {defect['bbox']}\n"
                details_text += "\n"
            
            fig.text(0.1, 0.85, details_text, fontsize=9, family='monospace',
                    verticalalignment='top')
            
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            plt.close('all')
            raise Exception(f"Error creating details page: {str(e)}")