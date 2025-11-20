# CircuitGuard: AI-Powered PCB Defect Detection

**CircuitGuard** is a complete, end-to-end application that leverages a hybrid computer vision and deep learning approach to automatically detect, classify, and report on manufacturing defects in Printed Circuit Boards (PCBs).

This project achieved a **99.65% test accuracy** with its trained `EfficientNet-B4` model, successfully exceeding the project's 97% target.

---


## 1. Project Overview

The manual visual inspection of PCBs is a slow, expensive, and error-prone bottleneck in electronics manufacturing. This project, **CircuitGuard**, was built to solve this problem by creating an automated, intelligent inspection tool.

The system works by:
1.  **Detecting** potential defects by comparing a test image against a defect-free "template" image using OpenCV's image subtraction methods.
2.  **Classifying** each detected defect using a deep learning model (`EfficientNet-B4`) trained on over 10,000 labeled examples.
3.  **Reporting** the findings through an interactive web dashboard with detailed charts, tables, and a downloadable, multi-page PDF report.

## 2. Key Features

* **High-Accuracy AI Model:** Achieved **99.65% test accuracy** on 6 defect classes.
* **Interactive Web Dashboard:** A user-friendly UI built with Streamlit, featuring a clean dashboard layout.
* **Adjustable Detection Parameters:** Sidebar sliders allow the user to fine-tune the `Difference Threshold`, `Minimum Defect Area`, and `AI Confidence Threshold` to optimize results.
* **Rich Data Visualization:** Generates a full dashboard of results, including:
    * An annotated Result Image with Bounding Boxes
    * A Defect Summary with counts (e.g., "Short: 3")
    * An Analysis Breakdown (Raw Difference & Cleaned Mask images)
    * Advanced Charts (Bar, Pie, Heatmap, and Scatter plots) for in-depth analysis.
* **Professional PDF Reporting:** Generates a comprehensive, multi-page **PDF report** using **ReportLab**, complete with a cover page, headers, footers, charts, and detailed defect logs.
* **Multiple Export Options:** Download the annotated image, a simple CSV log, or the full PDF report.

## 3. Tech Stack

| Area | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Building the interactive web application UI. |
| **Backend** | Python, OpenCV, NumPy | Image processing (subtraction, thresholding, contours). |
| **AI Model** | PyTorch, `timm` | Building, training, and running the `EfficientNet-B4` model. |
| **Data Analysis**| Pandas, Matplotlib, Seaborn | Creating the UI charts (bar, pie, heatmap, scatter). |
| **PDF Reporting**| **ReportLab** | Generating professional, multi-page PDF reports. |

---

## 4. Final Model Performance (99.65% Accuracy)

The model was trained for 30 epochs using advanced augmentation and a learning rate scheduler, which successfully solved initial overfitting and pushed the performance far beyond the 83% baseline.

| ![Training Plots](outputs/training_plots.png) | ![Confusion Matrix](outputs/confusion_matrix.png) |
| :---: | :---: |
| **Figure 1:** Model Accuracy/Loss vs. Epochs. The test accuracy (blue) successfully surpassed the 97% target. | **Figure 2:** Confusion Matrix. The strong diagonal shows near-perfect classification performance on the test set. |

---

## 5. How to Run This Project

### 1. Clone the Repository
```bash
git clone <github-repo-link>
cd CircuitGuard

2. Install Dependencies
pip install -r requirements.txt

3. Run the Application
streamlit run app.py
4. How to Use
Open the local URL (e.g., http://localhost:8501) in your browser.
Upload a "Template Image" (a defect-free PCB).
Upload a "Test Image" (a PCB you want to inspect).
Adjust the "Detection Parameters" in the sidebar as needed.
Click the "Analyze for Defects" button.
Review the full dashboard of results and download your desired report.

6. Project Structure
CircuitGuard/
│
├── data/
│ ├── raw/                # 1,500+ original images and annotations
│ └── processed/          # 10,000+ cropped images for training
│
├── Models/
│ └── best_model.pth      # The final 99.65% accuracy model
│
├── outputs/
│ ├── training_plots.png  # Accuracy/Loss graphs
│ └── confusion_matrix.png # Final confusion matrix
│
├── scripts/
│ ├── organize_raw_data.py # Script to sort the downloaded dataset
│ ├── prepare_dataset.py  # Script to create the training dataset
│ ├── train_model.py      # Script to train the AI model
│ └── evaluate_model.py   # Script to generate evaluation plots
│
├── app.py                # The main Streamlit application
├── pdf_generator.py      # Module for creating the PDF report
├── requirements.txt      # All Python dependencies

└── README.md             # This documentation file
