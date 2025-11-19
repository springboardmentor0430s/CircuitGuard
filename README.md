# CircuitGuard — Smart PCB Defect Detection Dashboard
CircuitGuard is an AI-powered quality inspection system that automatically detects and classifies **defects in Printed Circuit Boards (PCBs)** by comparing a defect-free *template image* with a *defective image*.
It integrates a deep learning model (EfficientNet-B4) with an interactive Streamlit dashboard and a professional report generator.

## Problem Statement

Traditional PCB inspection in manufacturing is **manual, time-consuming, and error-prone**.
Even minor defects like spurs, mousebites, or open circuits can cause device failure.
CircuitGuard aims to automate this process using **AI-based visual defect detection** to enhance accuracy, speed, and reliability.

## Objective

- Automate the defect detection process in PCB manufacturing.
- Detect and classify defects such as *Open, Short, Spur, Mousebite, Pin-hole,* and *Spurious Copper*.
- Generate detailed visual and analytical reports with heatmaps and defect summaries.
- Provide user control for filtering detections based on area, confidence, and blur sensitivity.

## Tools & Frameworks

| Category | Technology |
|-----------|-------------|
| **Frontend** | Streamlit, HTML/CSS (Custom Theme) |
| **Backend** | Python |
| **Model / ML Framework** | PyTorch, Torchvision |
| **Image Processing** | OpenCV, NumPy |
| **Visualization** | Matplotlib |
| **Report Generation** | FPDF |
| **IDE / Environment** | VS Code, Conda / venv |
| **Dataset** | DeepPCB Dataset (open-source) |

## Logical Flow

1. **Upload Images** — A reference PCB and a defective PCB
2. **Difference Extraction** — Identify changed pixels using `cv2.absdiff()`
3. **Mask Creation** — Thresholding + blur for defect isolation
4. **ROI Extraction** — Extract bounding boxes (potential defect regions)
5. **Classification** — EfficientNet-B4 predicts defect type
6. **Filtering** — Based on area, confidence, and blur sensitivity
7. **Visualization** — Overlay image + heatmap output
8. **Report Generation** — Detailed PDF report with summary and visuals

## Machine Learning Model

### Model:
**EfficientNet-B4** (Transfer Learning)

### Dataset:
**DeepPCB Dataset** (38K labeled PCB images)

Defect Classes:
- Open
- Short
- Spur
- Mousebite
- Pin-hole
- Spurious Copper

### Training Parameters

| Parameter | Value |
|------------|--------|
| Epochs | 10 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Input Size | 128×128 |
| Loss Function | CrossEntropyLoss |

### Evaluation Metrics

| Metric | Score |
|---------|-------|
| Accuracy | 95.6% |
| Precision | 93.8% |
| Recall | 94.1% |
| F1 Score | 93.9% |

## User Interface (Streamlit Dashboard)

### Features

- Upload template and test PCB images
- Adjust confidence threshold, area, and blur filters
- Visualize results in **Overlay**, **Heatmap**, or **Both**
- View detailed defect table with coordinates
- Generate a professional PDF report

### Parameter Controls

| Parameter | Function |
|------------|-----------|
| **Confidence Threshold** | Filters low-confidence detections |
| **Min / Max Area** | Controls ROI size |
| **Blur Strength** | Reduces background noise |
| **View Mode** | Overlay, Heatmap, Both |

## PDF Report Details

- **Header:** CircuitGuard branding + analysis metadata
- **Content Sections:**
  - System parameters
  - Defect summary table
  - Coordinates of detected ROIs
  - Overlay and heatmap images
- **Footer:** Auto-generated timestamp & branding

## Project Structure

circuitguard/
│
├── app.py # Main Streamlit App
├── theme.css # Custom UI Theme
├── requirements.txt # Python dependencies
│
├── models/
│ └── efficientnet_b4_best.pth
│
├── src/
│ ├── model.py
│ ├── overlay_visualizer.py
│ └── report_generator.py
│
├── outputs
│ ├── overlay_output.jpg
│ ├── heatmap_output.jpg
│ └── CircuitGuard_Report.pdf

└── README.md

## Installation & Usage

### 1️. Install Dependencies
```bash
pip install -r requirements.txt
2. Run the application
--streamlit run app.py

3. UPload images
Upload Template PCB (defect-free) Upload Test PCB (with potential defects)

4. view and export
see the exported pdf with classified defects
