"""
Image preprocessing utilities
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Handles image preprocessing operations
    """
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None,
                 grayscale: bool = True,
                 normalize: bool = True):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target size (height, width) for resizing
            grayscale: Whether to convert to grayscale
            normalize: Whether to normalize pixel values
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
    
    def load_image(self, image_path: str, 
                   grayscale: Optional[bool] = None) -> np.ndarray:
        """
        Load image from file
        
        Args:
            image_path: Path to image file
            grayscale: Override grayscale setting
            
        Returns:
            Loaded image
        """
        use_gray = grayscale if grayscale is not None else self.grayscale
        
        if use_gray:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    def resize_image(self, image: np.ndarray, 
                     size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            size: Target size (height, width), uses self.target_size if None
            
        Returns:
            Resized image
        """
        target = size if size is not None else self.target_size
        
        if target is None:
            return image
        
        return cv2.resize(image, (target[1], target[0]), 
                         interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert normalized image back to [0, 255] range
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image
        """
        return (image * 255.0).astype(np.uint8)
    
    def apply_gaussian_blur(self, image: np.ndarray, 
                           kernel_size: Tuple[int, int] = (5, 5),
                           sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur to image
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed image
        """
        # Load image
        image = self.load_image(image_path)
        
        # Resize if needed
        if self.target_size is not None:
            image = self.resize_image(image)
        
        # Normalize if needed
        if self.normalize:
            image = self.normalize_image(image)
        
        return image
    
    def get_image_stats(self, image: np.ndarray) -> dict:
        """
        Get statistical information about image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'std': float(np.std(image))
        }
        return stats