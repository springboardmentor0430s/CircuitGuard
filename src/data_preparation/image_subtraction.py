"""
Image subtraction for defect detection
"""

import cv2
import numpy as np
from typing import Tuple


class ImageSubtractor:
    """
    Performs image subtraction between template and test images
    """
    
    def __init__(self, method: str = 'absolute',
                 blur_kernel: Tuple[int, int] = (5, 5),
                 gaussian_sigma: float = 1.5):
        """
        Initialize subtractor
        
        Args:
            method: Subtraction method ('absolute', 'squared')
            blur_kernel: Gaussian blur kernel size
            gaussian_sigma: Gaussian blur sigma
        """
        self.method = method.lower()
        self.blur_kernel = blur_kernel
        self.gaussian_sigma = gaussian_sigma
    
    def subtract_images(self, template: np.ndarray, 
                       test_image: np.ndarray) -> np.ndarray:
        """
        Subtract template from test image
        
        Args:
            template: Template/reference image
            test_image: Test image (should be aligned)
            
        Returns:
            Difference map
        """
        # Ensure images have same shape
        if template.shape != test_image.shape:
            raise ValueError(f"Image shapes don't match: {template.shape} vs {test_image.shape}")
        
        # Convert to float for accurate subtraction
        template_float = template.astype(np.float32)
        test_float = test_image.astype(np.float32)
        
        # Perform subtraction
        if self.method == 'absolute':
            diff = cv2.absdiff(template_float, test_float)
        elif self.method == 'squared':
            diff = (template_float - test_float) ** 2
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return diff
    
    def preprocess_difference(self, diff: np.ndarray) -> np.ndarray:
        """
        Preprocess difference map with blurring
        
        Args:
            diff: Raw difference map
            
        Returns:
            Preprocessed difference map
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(diff, self.blur_kernel, self.gaussian_sigma)
        
        return blurred
    
    def normalize_difference(self, diff: np.ndarray) -> np.ndarray:
        """
        Normalize difference map to [0, 255] range
        
        Args:
            diff: Difference map
            
        Returns:
            Normalized difference map as uint8
        """
        # Normalize to 0-255 range
        normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    
    def compute_difference_map(self, template: np.ndarray,
                              test_image: np.ndarray,
                              preprocess: bool = True) -> np.ndarray:
        """
        Complete difference map computation pipeline
        
        Args:
            template: Template image
            test_image: Test image (aligned)
            preprocess: Whether to apply preprocessing
            
        Returns:
            Difference map (uint8)
        """
        # Subtract images
        diff = self.subtract_images(template, test_image)
        
        # Preprocess if requested
        if preprocess:
            diff = self.preprocess_difference(diff)
        
        # Normalize to uint8
        diff_normalized = self.normalize_difference(diff)
        
        return diff_normalized