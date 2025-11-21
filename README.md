CircuitGuard – PCB Defect Detection System

Deep Learning–based Automated Visual Inspection Tool

📌 Project Overview

CircuitGuard is an automated PCB defect detection system built using deep learning and image processing.
It compares a template PCB with a test PCB and identifies manufacturing defects such as:

.Missing Hole
.Mousebite
.Open Circuit
.Short
.Spur
.Spurious Copper
.And marks Non-Defect regions

The system performs difference detection, extracts regions of interest (ROIs), classifies each region using a trained EfficientNet-B4 model, and generates a fully automated PDF inspection report.

📁 Directory Structure

CircuitGuard_Project/
│
├── app/
│   ├── app.py              
│
├── backend/
│   ├── backend.py           
│
├── training/
│   ├── training.py          
│   ├── best_model.pth        
│        
│──testing/           
│   ├── test_results
├── datasets/
│   ├── PCBData
│   ├── PCBData_Paired
│             
├── matrices/
│   ├── confusion_matrix.png     
│
├── preprocessing/
│    ├── output  
│    │      ├── combined_rois 
│    │      ├── diff_images
│    │      ├── mask_images
│    │      ├── ROIs
│    │      ├── vis_images
│    ├── output_dataset
│    │      ├── test 
│    │      ├── train
│    │      ├── val
│    │
│    ├──preprocessing.py
│    ├──splitting.py
│    ├──txtfiles.py
│    
│
└── README.md


⚙️ Features

.Upload Template and Test PCB images
.Automatic defect detection using CNN
.Visual ROI annotation with color-coded boxes
.Statistical summaries
.Defect position table (coordinates)
.Bar & Pie chart visualization
.Automatically generated high-quality PDF report
.Fully responsive Streamlit UI



📦 Dependencies

| Package | Purpose |
|--------|---------|
| Streamlit | Web UI for uploading images & generating reports |
| PyTorch | Deep learning model loading & inference |
| Torchvision | EfficientNet-B4 model weights |
| NumPy | Matrix & image operations |
| OpenCV | Preprocessing, mask generation, ROI extraction |
| Pillow | Image conversion & saving |
| Matplotlib | Bar graph & pie chart generation |
| FPDF2 | Creates downloadable PDF report |
| Scikit-Learn | Metrics & model evaluation |



🧠 How to Run the Application
1. Install Dependencies
pip install -r streamlit torch torchvision numpy opencv-python Pillow matplotlib fpdf2scikit-learn .txt

2. Run the Streamlit App - python -m streamlit run "C:\Users\laksh\OneDrive\Desktop\coding\Circuitguard_Project\app\app.py"

3. Upload Images
.Upload Template Image
.Upload Test Image
.Click Analyze & Generate Report


🧪 Model Training (Optional)

If you want to retrain the model:

.python training/training.py
.Training file includes:
.Dataset loading
.Preprocessing
.Train/test split
.EfficientNet-B4 finetuning
.Saving best model


📄 PDF Report Output

The generated report includes:

1.Summary
2.Defect Count Table
3.Bar Graph
4.Pie Chart
5.Annotated PCB Image
6.Defect Position Table
7.Insights & Observations

📧 Contact

For any doubts or improvements:
K.Lakshmi Sravika
Email:Lakshmisravika2807@gmail.com