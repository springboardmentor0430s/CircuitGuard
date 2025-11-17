"""
Morphological operations for mask refinement
"""

import cv2
import numpy as np
from typing import Tuple


class MorphologicalProcessor:
    """
    Applies morphological operations to refine binary masks
    """
    
    def __init__(self, erosion_kernel: Tuple[int, int] = (3, 3),
                 dilation_kernel: Tuple[int, int] = (5, 5),
                 erosion_iterations: int = 1,
                 dilation_iterations: int = 2):
        """
        Initialize morphological processor
        
        Args:
            erosion_kernel: Kernel size for erosion
            dilation_kernel: Kernel size for dilation
            erosion_iterations: Number of erosion iterations
            dilation_iterations: Number of dilation iterations
        """
        self.erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, erosion_kernel
        )
        self.dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, dilation_kernel
        )
        self.erosion_iterations = erosion_iterations
        self.dilation_iterations = dilation_iterations
    
    def erode(self, mask: np.ndarray, iterations: int = None) -> np.ndarray:
        """
        Apply erosion to mask
        
        Args:
            mask: Binary mask
            iterations: Number of iterations (uses default if None)
            
        Returns:
            Eroded mask
        """
        iters = iterations if iterations is not None else self.erosion_iterations
        return cv2.erode(mask, self.erosion_kernel, iterations=iters)
    
    def dilate(self, mask: np.ndarray, iterations: int = None) -> np.ndarray:
        """
        Apply dilation to mask
        
        Args:
            mask: Binary mask
            iterations: Number of iterations (uses default if None)
            
        Returns:
            Dilated mask
        """
        iters = iterations if iterations is not None else self.dilation_iterations
        return cv2.dilate(mask, self.dilation_kernel, iterations=iters)
    
    def opening(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening (erosion followed by dilation)
        Removes small noise
        
        Args:
            mask: Binary mask
            
        Returns:
            Opened mask
        """
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.erosion_kernel)
    
    def closing(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological closing (dilation followed by erosion)
        Fills small holes
        
        Args:
            mask: Binary mask
            
        Returns:
            Closed mask
        """
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.dilation_kernel)
    
    def remove_small_objects(self, mask: np.ndarray, min_area: int = 50) -> np.ndarray:
        """
        Remove small connected components
        
        Args:
            mask: Binary mask
            min_area: Minimum area threshold
            
        Returns:
            Cleaned mask
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # Create output mask
        output = np.zeros_like(mask)
        
        # Keep only components larger than threshold
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                output[labels == i] = 255
        
        return output
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in binary mask
        
        Args:
            mask: Binary mask
            
        Returns:
            Mask with holes filled
        """
        # Copy the mask
        im_floodfill = mask.copy()
        
        # Get mask shape
        h, w = mask.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, flood_mask, (0, 0), 255)
        
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        
        # Combine with original mask
        filled = mask | im_floodfill_inv
        
        return filled
    
    def refine_mask(self, mask: np.ndarray, min_area: int = 50) -> np.ndarray:
        """
        Complete mask refinement pipeline
        
        Args:
            mask: Binary mask
            min_area: Minimum area for connected components
            
        Returns:
            Refined mask
        """
        # Apply opening to remove noise
        refined = self.opening(mask)
        
        # Apply closing to fill gaps
        refined = self.closing(refined)
        
        # Remove small objects
        refined = self.remove_small_objects(refined, min_area)
        
        # Fill holes
        refined = self.fill_holes(refined)
        
        return refined