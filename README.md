🖥️ PCB Defect Detection & Classification using EfficientNet-B4
📌 Project Overview

This project implements an automated deep learning–based system for detecting and classifying defects in Printed Circuit Boards (PCBs). 
It compares a reference image with a test PCB image, identifies defects using image processing and classifies them using EfficientNet-B4, delivering visual analytics and downloadable PDF reports.

🌟 What is PCB Defect Detector?
The PCB Defect Detector is a powerful end-to-end AI application that revolutionizes traditional PCB inspection. By integrating advanced computer vision, deep learning (EfficientNet-B4), and an interactive Streamlit user interface, it automates the complete defect detection process—from image preprocessing to defect classification and real-time PDF reporting.

The system compares a defect-free reference PCB image against a test PCB image, identifies deviations, isolates defective regions, and classifies them into categories such as mousebite, open, short, copper, noise, pin-hole, and spur. It delivers results in real-time (2–5 seconds) with high accuracy and visual markup annotations.

Whether used in academic research, automated quality control, prototype validation, or industrial manufacturing lines, this system brings precision, speed, and traceable analytics to PCB inspection workflows.

🎯 The Problem We’re Solving
Manual visual inspection of PCBs is still common in many manufacturing environments, but it suffers from major limitations:
Slow & labor-intensive — each board must be inspected manually by trained technicians
Prone to human fatigue & inconsistency
Low scalability — cannot sustain high-volume production lines
High error rate — subtle defects may be overlooked
Poor documentation — lacks structured quality analysis and traceability
Increased production cost due to rework and scrap caused by overlooked defects

These limitations create major bottlenecks and risks in modern electronics production, especially with today’s dense circuit layouts.

✅ Our Solution
The PCB Defect Detector overcomes these challenges with a fully automated AI-driven system featuring:

🚀 Real-Time Automated Defect Detection
Processes and analyzes uploaded PCB images instantly, detecting multiple defect types with high accuracy and reliability.

🧠 Deep Learning-Powered Classification
Built using EfficientNet-B4 (PyTorch) trained on thousands of PCB defect samples for robust pattern recognition, even under noisy conditions.

🖼 Visual Markings & Region-Based Analysis
Highlights defect locations using bounding boxes and color codes, offering instant spatial insight.

📊 Data-Driven Insights & Analytics
Generates charts like bar graphs, pie charts, and confidence-area scatter plots for deeper statistical analysis.

📄 Auto-Generated Professional PDF Report
Creates a polished inspection document including:
Input and processed images
Defect log table
Analytical charts
Summaries and metrics

Perfect for audits and documentation workflows.

🌐 User-Friendly Streamlit Web Application
No technical expertise required — simple upload interface, slider-based controls, and single-click export support.

🛠 Technology Stack
Backend / Model | PyTorch, EfficientNet-B4
Computer Vision | OpenCV, Numpy
UI / Frontend | Streamlit
Visualization | Matplotlib, Seaborn
Data Processing | Pandas
Report Generation | ReportLab PDF Engine


🧪 Supported PCB Defect Types
Copper
Mousebite
Noise
Open
Pin-hole
Short
Spur

🏁 How It Works (Pipeline Overview)
Upload reference and test PCB images
Preprocessing using difference imaging + thresholding
Contour detection and ROI extraction
EfficientNet-B4 classification on each defect region
Visual annotation overlay and heatmap creation
Statistical analysis and plotting
Auto-generate structured PDF report
Download results instantly


