# config.py
class Config:
    # Alignment parameters
    ALIGNMENT_METHOD = "enhanced"  # "simple", "feature", "enhanced"
    MIN_MATCHES = 10
    
    # Subtraction parameters
    USE_ADAPTIVE_THRESHOLD = True
    ADAPTIVE_BLOCK_SIZE = 11
    ADAPTIVE_C = 2
    
    # Morphological parameters
    OPENING_KERNEL = (2, 2)
    CLOSING_KERNEL = (3, 3)
    
    # Contour parameters
    MIN_DEFECT_AREA = 15
    MAX_DEFECT_AREA = 500
    MIN_ASPECT_RATIO = 0.2
    NMS_OVERLAP_THRESH = 0.3