"""
Image alignment using feature matching
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class ImageAligner:
    """
    Aligns test images to template images using feature matching
    """
    
    def __init__(self, method: str = 'orb',
                 max_features: int = 5000,
                 match_threshold: float = 0.75,
                 ransac_threshold: float = 5.0):
        """
        Initialize aligner
        
        Args:
            method: Feature detection method ('orb', 'sift', 'ecc')
            max_features: Maximum number of features to detect
            match_threshold: Ratio threshold for feature matching
            ransac_threshold: RANSAC reprojection threshold
        """
        self.method = method.lower()
        self.max_features = max_features
        self.match_threshold = match_threshold
        self.ransac_threshold = ransac_threshold
        
        # Initialize feature detector
        if self.method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=max_features)
        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple:
        """
        Detect keypoints and compute descriptors
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, 
                       desc2: np.ndarray) -> list:
        """
        Match features between two images
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            
        Returns:
            List of good matches
        """
        # Create matcher
        if self.method == 'orb':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:  # SIFT
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Match descriptors
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test (Lowe's ratio test)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def estimate_transform(self, kp1: list, kp2: list, 
                          matches: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate transformation matrix between images
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Good matches
            
        Returns:
            Tuple of (homography matrix, mask)
        """
        if len(matches) < 4:
            raise ValueError("Not enough matches to compute homography")
        
        # Extract matched keypoint locations
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                     self.ransac_threshold)
        
        return H, mask
    
    def align_image(self, template: np.ndarray, 
                    test_image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Align test image to template
        
        Args:
            template: Reference template image (grayscale)
            test_image: Test image to align (grayscale)
            
        Returns:
            Tuple of (aligned image, alignment info dict)
        """
        # Ensure images are grayscale uint8
        if template.dtype != np.uint8:
            template = (template * 255).astype(np.uint8)
        if test_image.dtype != np.uint8:
            test_image = (test_image * 255).astype(np.uint8)
        
        # Detect features
        kp1, desc1 = self.detect_and_compute(template)
        kp2, desc2 = self.detect_and_compute(test_image)
        
        if desc1 is None or desc2 is None:
            raise ValueError("Failed to detect features")
        
        # Match features
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < 4:
            raise ValueError(f"Not enough matches found: {len(matches)}")
        
        # Estimate transformation
        H, mask = self.estimate_transform(kp1, kp2, matches)
        
        if H is None:
            raise ValueError("Failed to compute homography")
        
        # Warp test image to align with template
        h, w = template.shape[:2]
        aligned = cv2.warpPerspective(test_image, H, (w, h))
        
        # Compute alignment info
        inliers = np.sum(mask) if mask is not None else 0
        alignment_info = {
            'num_keypoints_template': len(kp1),
            'num_keypoints_test': len(kp2),
            'num_matches': len(matches),
            'num_inliers': int(inliers),
            'homography': H.tolist()
        }
        
        return aligned, alignment_info
    
    def visualize_matches(self, template: np.ndarray, 
                         test_image: np.ndarray,
                         max_display: int = 50) -> np.ndarray:
        """
        Visualize feature matches between images
        
        Args:
            template: Template image
            test_image: Test image
            max_display: Maximum matches to display
            
        Returns:
            Visualization image
        """
        # Ensure images are uint8
        if template.dtype != np.uint8:
            template = (template * 255).astype(np.uint8)
        if test_image.dtype != np.uint8:
            test_image = (test_image * 255).astype(np.uint8)
        
        # Detect and match
        kp1, desc1 = self.detect_and_compute(template)
        kp2, desc2 = self.detect_and_compute(test_image)
        matches = self.match_features(desc1, desc2)
        
        # Draw matches
        matches_img = cv2.drawMatches(
            template, kp1, test_image, kp2, 
            matches[:max_display], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return matches_img