"""
Contour extraction and ROI detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


class ContourExtractor:
    """
    Extracts contours and bounding boxes from binary masks
    """
    
    def __init__(self, min_area: int = 50,
                 max_area: int = 50000,
                 approximation: str = 'simple'):
        """
        Initialize contour extractor
        
        Args:
            min_area: Minimum contour area
            max_area: Maximum contour area
            approximation: Contour approximation method
        """
        self.min_area = min_area
        self.max_area = max_area
        self.approximation = approximation
    
    def extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Extract contours from binary mask
        
        Args:
            mask: Binary mask
            
        Returns:
            List of contours
        """
        # Find contours
        if self.approximation == 'simple':
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
        
        # Filter by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def get_bounding_boxes(self, contours: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Get bounding boxes for contours
        
        Args:
            contours: List of contours
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
        return bboxes
    
    def get_contour_properties(self, contours: List[np.ndarray]) -> List[Dict]:
        """
        Get properties for each contour
        
        Args:
            contours: List of contours
            
        Returns:
            List of property dictionaries
        """
        properties = []
        
        for i, contour in enumerate(contours):
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Area
            area = cv2.contourArea(contour)
            
            # Perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            properties.append({
                'id': i,
                'bbox': (x, y, w, h),
                'area': area,
                'perimeter': perimeter,
                'centroid': (cx, cy),
                'aspect_ratio': w / h if h != 0 else 0
            })
        
        return properties
    
    def extract_rois(self, image: np.ndarray, 
                     contours: List[np.ndarray],
                     padding: int = 5) -> List[np.ndarray]:
        """
        Extract ROI images from contours
        
        Args:
            image: Source image
            contours: List of contours
            padding: Padding around bounding box
            
        Returns:
            List of ROI images
        """
        rois = []
        h, w = image.shape[:2]
        
        for contour in contours:
            x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
            
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bbox_w + padding)
            y2 = min(h, y + bbox_h + padding)
            
            # Extract ROI
            roi = image[y1:y2, x1:x2]
            rois.append(roi)
        
        return rois
    
    def draw_contours_and_boxes(self, image: np.ndarray,
                                contours: List[np.ndarray],
                                color: Tuple[int, int, int] = (0, 255, 0),
                                thickness: int = 2) -> np.ndarray:
        """
        Draw contours and bounding boxes on image
        
        Args:
            image: Input image
            contours: List of contours
            color: Drawing color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn contours and boxes
        """
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()
        
        # Draw contours
        cv2.drawContours(output, contours, -1, color, thickness)
        
        # Draw bounding boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), thickness)
        
        return output
    
    def annotate_defects(self, image: np.ndarray,
                         properties: List[Dict]) -> np.ndarray:
        """
        Annotate image with defect information
        
        Args:
            image: Input image
            properties: List of contour properties
            
        Returns:
            Annotated image
        """
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()
        
        for prop in properties:
            x, y, w, h = prop['bbox']
            
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw centroid
            cx, cy = prop['centroid']
            cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
            
            # Add text label
            label = f"ID:{prop['id']} A:{int(prop['area'])}"
            cv2.putText(output, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return output