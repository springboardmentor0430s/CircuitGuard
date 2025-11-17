"""
Evaluation metrics for PCB defect detection
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class AlignmentMetrics:
    """
    Metrics for evaluating image alignment quality
    """
    
    @staticmethod
    def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Mean Squared Error between two images
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            MSE value
        """
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Peak Signal-to-Noise Ratio
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            PSNR value in dB
        """
        mse = AlignmentMetrics.compute_mse(img1, img2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (simplified version)
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            SSIM value [0, 1]
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
    
    @staticmethod
    def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Normalized Cross-Correlation
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            NCC value [-1, 1]
        """
        img1_flat = img1.flatten().astype(float)
        img2_flat = img2.flatten().astype(float)
        
        img1_norm = (img1_flat - np.mean(img1_flat)) / (np.std(img1_flat) + 1e-10)
        img2_norm = (img2_flat - np.mean(img2_flat)) / (np.std(img2_flat) + 1e-10)
        
        ncc = np.mean(img1_norm * img2_norm)
        return float(ncc)
    
    @staticmethod
    def evaluate_alignment(template: np.ndarray, 
                          aligned: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive alignment evaluation
        
        Args:
            template: Template image
            aligned: Aligned image
            
        Returns:
            Dictionary of metrics
        """
        return {
            'mse': AlignmentMetrics.compute_mse(template, aligned),
            'psnr': AlignmentMetrics.compute_psnr(template, aligned),
            'ssim': AlignmentMetrics.compute_ssim(template, aligned),
            'ncc': AlignmentMetrics.compute_ncc(template, aligned)
        }


class DefectDetectionMetrics:
    """
    Metrics for evaluating defect detection performance
    """
    
    @staticmethod
    def compute_iou(bbox1: Tuple[int, int, int, int], 
                    bbox2: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union for two bounding boxes
        
        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)
            
        Returns:
            IoU value [0, 1]
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to (x1, y1, x2, y2) format
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def compute_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Dice coefficient between two binary masks
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            Dice coefficient [0, 1]
        """
        intersection = np.sum(mask1 * mask2)
        dice = (2.0 * intersection) / (np.sum(mask1) + np.sum(mask2) + 1e-10)
        return float(dice)
    
    @staticmethod
    def compute_pixel_accuracy(pred_mask: np.ndarray, 
                              true_mask: np.ndarray) -> float:
        """
        Compute pixel-wise accuracy
        
        Args:
            pred_mask: Predicted binary mask
            true_mask: Ground truth binary mask
            
        Returns:
            Accuracy [0, 1]
        """
        correct = np.sum(pred_mask == true_mask)
        total = pred_mask.size
        return correct / total
    
    @staticmethod
    def compute_precision_recall(pred_mask: np.ndarray,
                                 true_mask: np.ndarray) -> Tuple[float, float]:
        """
        Compute precision and recall
        
        Args:
            pred_mask: Predicted binary mask (0 or 255)
            true_mask: Ground truth binary mask (0 or 255)
            
        Returns:
            Tuple of (precision, recall)
        """
        pred_binary = (pred_mask > 0).astype(int).flatten()
        true_binary = (true_mask > 0).astype(int).flatten()
        
        tp = np.sum((pred_binary == 1) & (true_binary == 1))
        fp = np.sum((pred_binary == 1) & (true_binary == 0))
        fn = np.sum((pred_binary == 0) & (true_binary == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        return float(precision), float(recall)
    
    @staticmethod
    def compute_f1_score(precision: float, recall: float) -> float:
        """
        Compute F1 score from precision and recall
        
        Args:
            precision: Precision value
            recall: Recall value
            
        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def evaluate_detection(pred_mask: np.ndarray,
                          true_mask: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive detection evaluation
        
        Args:
            pred_mask: Predicted binary mask
            true_mask: Ground truth binary mask
            
        Returns:
            Dictionary of metrics
        """
        precision, recall = DefectDetectionMetrics.compute_precision_recall(
            pred_mask, true_mask
        )
        f1 = DefectDetectionMetrics.compute_f1_score(precision, recall)
        
        return {
            'pixel_accuracy': DefectDetectionMetrics.compute_pixel_accuracy(pred_mask, true_mask),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'dice_coefficient': DefectDetectionMetrics.compute_dice_coefficient(pred_mask, true_mask)
        }


class ImageQualityMetrics:
    """
    General image quality metrics
    """
    
    @staticmethod
    def compute_snr(image: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio
        
        Args:
            image: Input image
            
        Returns:
            SNR value in dB
        """
        mean = np.mean(image)
        std = np.std(image)
        
        if std == 0:
            return float('inf')
        
        snr = 20 * np.log10(mean / std)
        return snr
    
    @staticmethod
    def compute_contrast(image: np.ndarray) -> float:
        """
        Compute image contrast (RMS contrast)
        
        Args:
            image: Input image
            
        Returns:
            Contrast value
        """
        return float(np.std(image))
    
    @staticmethod
    def compute_entropy(image: np.ndarray) -> float:
        """
        Compute image entropy
        
        Args:
            image: Input image
            
        Returns:
            Entropy value
        """
        histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        histogram = histogram / histogram.sum()
        histogram = histogram[histogram > 0]
        entropy = -np.sum(histogram * np.log2(histogram))
        return float(entropy)
    
    @staticmethod
    def compute_sharpness(image: np.ndarray) -> float:
        """
        Compute image sharpness using Laplacian variance
        
        Args:
            image: Input image
            
        Returns:
            Sharpness value
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return float(laplacian.var())
    
    @staticmethod
    def evaluate_quality(image: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive image quality evaluation
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of quality metrics
        """
        return {
            'snr': ImageQualityMetrics.compute_snr(image),
            'contrast': ImageQualityMetrics.compute_contrast(image),
            'entropy': ImageQualityMetrics.compute_entropy(image),
            'sharpness': ImageQualityMetrics.compute_sharpness(image),
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image))
        }


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics dictionary
    
    Args:
        metrics: Dictionary of metric values
        title: Title for the metrics
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"  {key:.<40} {value:.4f}")
    print(f"{'='*50}\n")