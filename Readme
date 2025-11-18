🖥️ PCB Defect Detection & Classification using EfficientNet-B4
📌 Project Overview

This project implements an automated deep learning–based system for detecting and classifying defects in Printed Circuit Boards (PCBs). 
It compares a reference image with a test PCB image, identifies defects using image processing and classifies them using EfficientNet-B4, delivering visual analytics and downloadable PDF reports.

🧠 Key Features
Feature	                             Description
---------------------------------------------------------------------------------------
Automated defect localization	     Difference-based segmentation + contour detection
Accurate defect classification	     EfficientNet-B4 deep learning model
Bounding box visualization	         Annotated output images with colored labels
Analytics Dashboard	                 Pie, Bar & Scatter plots for defect analysis
PDF Report Generator	             Multi-page industry-style downloadable report
Confidence-based filtering	         Adjustable via Streamlit UI
Combined results	                 Heatmap & annotated comparison view


🏗️ System Architecture
Upload PCB Images (Reference + Test)
              ↓
Image Preprocessing (Diff + Threshold + Morphology)
              ↓
Contour Detection & ROI Extraction
              ↓
EfficientNet-B4 Classification (PyTorch)
              ↓
Defect Log + Charts Dashboard
              ↓
PDF Export + Annotated Image Download


🚀 Tech Stack
Category	              Tools / Frameworks
----------------------------------------------------
Deep Learning	         PyTorch, EfficientNet-B4
Frontend UI	             Streamlit
Image Processing	     OpenCV
Visualization	         Matplotlib, Seaborn
Reporting	             ReportLab
Data Handling	         Pandas, NumPy
Deployment	             Local GPU / CPU execution


🧾 Dataset & Defect Classes
Class	       Description
--------------------------------------------
Copper	      Excess copper region
Mousebite	  Rough edge indentation
Noise	      Random false anomalies
Open	      Trace breaks / discontinuity
Pin-hole	  Small circular missing copper
Short	      Unwanted copper bridging
Spur	      Copper protrusion


📊 Performance Metrics
Overall Metrics
Metric	                Value
-------------------------------
Test Accuracy	       97.81%
Precision	           0.968
Recall	               0.968
F1-Score	           0.968


Per-Class Performance
Defect Type	  Precision	  Recall	F1-Score	Support
--------------------------------------------------------
Copper	      0.986	      0.997	    0.991	    290
Mousebite	  0.983	      0.964	    0.973	    359
Open	      0.972	      0.990	    0.981	    384
Pin-hole	  0.990	      0.996	    0.993	    285
Short	      0.986	      0.979	    0.983	    288
Spur	      0.984	      0.975	    0.979	    317


🧠 Model Information
Parameter	           Value
------------------------------------------------------
Architecture	      EfficientNet-B4
Training Epochs	      30
Batch Size	          32
Learning Rate	      1e-4
Input Size	          128×128
Dataset Split	      70% Train / 10% Val / 20% Test


📥 Installation & Execution
Prerequisites
pip install -r requirements.txt

Run the Application
streamlit run app.py

Model Training
python model_train.py


📦 Output Deliverables
Output Type	               Description
-----------------------------------------------------------------
Annotated Image	          Labeled bounding boxes with confidence
Heatmap	                  Severity visualization
Analytics Charts	      Pie, Bar & Scatter plots
PDF Report	              Multi-page generated summary


🔮 Future Enhancements
Planned Feature	                  Benefit
------------------------------------------------------------------
YOLO-v8 segmentation	         Pixel-level defect shapes
Cloud deployment	             Real-time factory integration
ONNX / TensorRT optimization	 Faster inference on edge devices
Support for Multi-layer PCB	     Industrial-grade inspection


