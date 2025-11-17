# CircuitGuard-PCB - Technical Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Pipeline Overview](#pipeline-overview)
4. [Module Descriptions](#module-descriptions)
5. [Algorithm Details](#algorithm-details)
6. [Performance Metrics](#performance-metrics)
7. [API Reference](#api-reference)

---

## System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Web Interface (Streamlit)                │
│  - Single Inspection  - Batch Processing  - History  - Export│
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│                    Backend API Layer                         │
│  - PCBInspectionBackend  - BatchProcessor  - ReportGenerator │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴──────────┐
        ▼                      ▼
┌───────────────┐      ┌──────────────────┐
│ Defect        │      │ Deep Learning    │
│ Detection     │      │ Classification   │
│ Pipeline      │      │ (EfficientNet-B4)│
└───────────────┘      └──────────────────┘
        │                      │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │   Data Processing    │
        │   & Visualization    │
        └──────────────────────┘
```

---

## Technology Stack

### Core Technologies
- **Python 3.11+**: Main programming language
- **PyTorch 2.0+**: Deep learning framework
- **OpenCV 4.8+**: Computer vision operations
- **Streamlit 1.28+**: Web application framework

### Deep Learning
- **Model**: EfficientNet-B4 (pretrained on ImageNet)
- **Input**: 128×128 grayscale images
- **Output**: 6-class defect classification
- **Accuracy**: ≥97% on test set

### Image Processing
- **Feature Detection**: ORB (Oriented FAST and Rotated BRIEF)
- **Alignment**: RANSAC-based homography estimation
- **Thresholding**: Otsu's method
- **Morphology**: Erosion and dilation operations

### Data Management
- **Storage**: Local file system
- **Format**: JPG/PNG images, CSV/JSON metadata
- **Organization**: Train/Val/Test splits (70/15/15)

---

## Pipeline Overview

### Complete Processing Flow
```
Input: Template (Reference) + Test (Inspection) Image
    │
    ├──> 1. Image Preprocessing
    │       - Grayscale conversion
    │       - Noise reduction
    │
    ├──> 2. Image Alignment
    │       - ORB feature detection
    │       - Feature matching
    │       - RANSAC homography
    │       - Perspective transformation
    │
    ├──> 3. Defect Detection
    │       - Image subtraction
    │       - Difference map computation
    │       - Otsu thresholding
    │       - Morphological refinement
    │       - Contour extraction
    │
    ├──> 4. ROI Extraction
    │       - Bounding box computation
    │       - Region of Interest extraction
    │       - Size normalization (128×128)
    │
    ├──> 5. Classification
    │       - EfficientNet-B4 inference
    │       - Softmax probability
    │       - Class prediction
    │
    └──> 6. Output Generation
            - Annotated images
            - Classification results
            - Confidence scores
            - Export files
```

---

## Module Descriptions

### 1. Data Preparation (`src/data_preparation/`)

#### `preprocessing.py`
- **Purpose**: Image loading and basic preprocessing
- **Key Functions**:
  - `load_image()`: Load and convert images
  - `resize_image()`: Resize with aspect ratio preservation
  - `normalize_image()`: Pixel value normalization

#### `image_alignment.py`
- **Purpose**: Align test images to template
- **Algorithm**: ORB + RANSAC
- **Parameters**:
  - Max features: 5000
  - Match ratio: 0.75
  - RANSAC threshold: 5.0

#### `image_subtraction.py`
- **Purpose**: Compute difference between aligned images
- **Methods**:
  - Absolute difference
  - Weighted subtraction
  - Adaptive subtraction

### 2. Defect Detection (`src/defect_detection/`)

#### `thresholding.py`
- **Purpose**: Binary segmentation of defects
- **Methods**:
  - Otsu's thresholding (primary)
  - Adaptive thresholding (alternative)
  - Multi-level thresholding

#### `morphological_ops.py`
- **Purpose**: Noise removal and defect refinement
- **Operations**:
  - Erosion (3×3 kernel)
  - Dilation (5×5 kernel)
  - Opening and closing
  - Connected component analysis

#### `contour_extraction.py`
- **Purpose**: Extract defect regions
- **Features**:
  - Contour detection
  - Area filtering (50-50,000 px²)
  - Bounding box computation
  - ROI extraction with padding

### 3. Deep Learning Model (`src/model/`)

#### `model.py`
- **Architecture**: EfficientNet-B4
- **Modifications**:
  - Grayscale input (1 channel)
  - Custom classifier head
  - Dropout (0.3) for regularization
  - 6-class output

#### `dataset.py`
- **Data Loading**: PyTorch DataLoader
- **Augmentation**:
  - Horizontal/vertical flips
  - Rotation (±15°)
  - Brightness/contrast adjustment
  - Gaussian blur

#### `trainer.py`
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Cross-entropy
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience=10 epochs

### 4. Web Application (`src/web_app/`)

#### `backend.py`
- **Class**: `PCBInspectionBackend`
- **Methods**:
  - `process_image_pair()`: Main inference
  - `get_class_statistics()`: Defect counting
  - `format_results_for_display()`: UI formatting

#### `batch_processor.py`
- **Class**: `BatchProcessor`
- **Features**:
  - Multi-image processing
  - Progress tracking
  - Result aggregation
  - Statistics computation

---

## Algorithm Details

### 1. ORB Feature Detection

**Steps**:
1. Detect FAST corners
2. Compute BRIEF descriptors with rotation
3. Match using Hamming distance
4. Apply Lowe's ratio test (0.75)

**Advantages**:
- Rotation invariant
- Scale invariant
- Fast computation
- Patent-free

### 2. RANSAC Homography

**Process**:
1. Randomly sample 4 point correspondences
2. Compute homography matrix
3. Count inliers (threshold: 5.0 pixels)
4. Repeat for N iterations
5. Select best homography

**Parameters**:
- Iterations: 5000
- Reprojection threshold: 5.0
- Confidence: 0.995

### 3. Otsu's Thresholding

**Algorithm**:
1. Compute histogram of difference map
2. For each threshold t:
   - Calculate between-class variance
3. Select t that maximizes variance

**Formula**:
```
σ²(t) = ω₀(t)ω₁(t)[μ₀(t) - μ₁(t)]²
```

### 4. EfficientNet-B4 Architecture

**Specifications**:
- Depth: 380 layers
- Width multiplier: 1.4
- Resolution: 380×380 (adapted to 128×128)
- Parameters: ~19M

**Compound Scaling**:
```
depth = α^φ
width = β^φ
resolution = γ^φ
α × β² × γ² ≈ 2
```

---

## Performance Metrics

### Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ≥97.0% |
| Precision | 0.97+ |
| Recall | 0.96+ |
| F1-Score | 0.96+ |

### Per-Class Performance

| Defect Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Mousebite | 0.98 | 0.97 | 0.97 |
| Open | 0.96 | 0.95 | 0.96 |
| Short | 0.97 | 0.98 | 0.97 |
| Spur | 0.95 | 0.94 | 0.95 |
| Copper | 0.96 | 0.97 | 0.96 |
| Pin-hole | 0.97 | 0.96 | 0.96 |

### Processing Speed

| Operation | Time (avg) |
|-----------|------------|
| Image Loading | 0.05s |
| Alignment | 0.3s |
| Detection | 0.2s |
| Classification | 0.1s |
| **Total** | **~0.65s** |

**Throughput**: ~1.5 images/second (CPU)

---

## API Reference

### Backend API

#### `PCBInspectionBackend`
```python
backend = PCBInspectionBackend()

# Process single image pair
result = backend.process_image_pair(
    template_image: np.ndarray,  # Grayscale template
    test_image: np.ndarray       # Grayscale test image
) -> Dict

# Result structure:
{
    'success': bool,
    'num_defects': int,
    'classifications': List[Dict],
    'images': Dict[str, np.ndarray],
    'alignment_info': Dict,
    'processing_time': float
}
```

#### `DefectPredictor`
```python
predictor = DefectPredictor(model, config, device)

# Detect defects
detection_result = predictor.detect_defects(
    template_path: str,
    test_path: str
) -> Dict

# Classify single defect
class_id, confidence, probs = predictor.classify_defect(
    roi: np.ndarray
) -> Tuple[int, float, np.ndarray]
```

### Configuration

**Format**: YAML (`configs/config.yaml`)
```yaml
# Key parameters
alignment:
  method: 'orb'
  max_features: 5000

thresholding:
  method: 'otsu'

contours:
  min_area: 50
  max_area: 50000

model:
  architecture: 'efficientnet_b4'
  num_classes: 6
  dropout: 0.3

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 50
```

---

## Error Handling

### Common Errors

1. **Alignment Failure**
   - Cause: Insufficient feature matches
   - Solution: Increase max_features or use different images

2. **Memory Error**
   - Cause: Large batch size or image size
   - Solution: Reduce batch_size or resize images

3. **CUDA Out of Memory**
   - Cause: GPU memory exhausted
   - Solution: Use CPU or reduce batch_size

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| E001 | Alignment failed | Check image quality |
| E002 | Model loading failed | Verify checkpoint path |
| E003 | Invalid image format | Convert to JPG/PNG |
| E004 | Out of memory | Reduce batch size |

---

## Security Considerations

1. **Input Validation**
   - File type checking
   - Size limitations (200MB max)
   - Format verification

2. **Path Traversal Prevention**
   - Sanitized file paths
   - Restricted directory access

3. **Resource Limits**
   - Processing timeout (60s)
   - Memory limits
   - Concurrent request limits

---

## Troubleshooting

### Issue: Slow Processing

**Solutions**:
1. Enable GPU acceleration
2. Reduce image resolution
3. Decrease max_features parameter
4. Use batch processing

### Issue: Poor Alignment

**Solutions**:
1. Ensure good lighting in images
2. Check for sufficient texture
3. Increase max_features
4. Try SIFT instead of ORB

### Issue: Low Classification Accuracy

**Solutions**:
1. Retrain model with more data
2. Adjust confidence threshold
3. Review defect ROI extraction
4. Fine-tune model hyperparameters

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025 | Initial release |
| 1.1.0 | 2025 | Added batch processing |
| 1.2.0 | 2025 | Enhanced export features |

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Maintained By**: CircuitGuard-PCB Team