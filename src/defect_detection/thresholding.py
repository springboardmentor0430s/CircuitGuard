"""
Thresholding techniques for defect segmentation
"""

import cv2
import numpy as np
from typing import Tuple


class DefectThresholder:
    """
    Applies thresholding to difference maps to create binary masks
    """
    
    def __init__(self, method: str = 'otsu',
                 min_threshold: int = 10,
                 max_threshold: int = 255):
        """
        Initialize thresholder
        
        Args:
            method: Thresholding method ('otsu', 'adaptive', 'binary')
            min_threshold: Minimum threshold value
            max_threshold: Maximum threshold value
        """
        self.method = method.lower()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    def apply_otsu_threshold(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Apply Otsu's thresholding
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (binary mask, threshold value used)
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Apply Otsu's thresholding
        threshold_value, binary_mask = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return binary_mask, int(threshold_value)
    
    def apply_adaptive_threshold(self, image: np.ndarray,
                                 block_size: int = 11,
                                 C: int = 2) -> np.ndarray:
        """
        Apply adaptive thresholding
        
        Args:
            image: Input grayscale image
            block_size: Size of neighborhood area
            C: Constant subtracted from mean
            
        Returns:
            Binary mask
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Apply adaptive thresholding
        binary_mask = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
        
        return binary_mask
    
    def apply_binary_threshold(self, image: np.ndarray,
                               threshold_value: int = 50) -> np.ndarray:
        """
        Apply simple binary thresholding
        
        Args:
            image: Input grayscale image
            threshold_value: Threshold value
            
        Returns:
            Binary mask
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Apply binary thresholding
        _, binary_mask = cv2.threshold(
            image, threshold_value, 255, cv2.THRESH_BINARY
        )
        
        return binary_mask
    
    def threshold_difference_map(self, diff_map: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Apply thresholding to difference map
        
        Args:
            diff_map: Difference map (grayscale uint8)
            
        Returns:
            Tuple of (binary mask, threshold info dict)
        """
        info = {'method': self.method}
        
        if self.method == 'otsu':
            binary_mask, threshold_value = self.apply_otsu_threshold(diff_map)
            info['threshold_value'] = threshold_value
        elif self.method == 'adaptive':
            binary_mask = self.apply_adaptive_threshold(diff_map)
            info['block_size'] = 11
        elif self.method == 'binary':
            binary_mask = self.apply_binary_threshold(diff_map, self.min_threshold)
            info['threshold_value'] = self.min_threshold
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return binary_mask, info
    
    def post_process_mask(self, mask: np.ndarray,
                         min_area: int = 50) -> np.ndarray:
        """
        Post-process binary mask to remove noise
        
        Args:
            mask: Binary mask
            min_area: Minimum area for connected components
            
        Returns:
            Cleaned binary mask
        """
        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # Create clean mask
        clean_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                clean_mask[labels == i] = 255
        
        return clean_mask