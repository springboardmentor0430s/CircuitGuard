#  CircuitGuard – PCB Defect Detection System

**Image Processing • MobileNetV2 • Streamlit Web App**

CircuitGuard is an AI-powered PCB defect detection and classification system.  
It combines OpenCV-based image subtraction with a MobileNetV2 deep learning model to identify, localize, and label PCB defects through an intuitive web interface.

---

##  Features
- MobileNetV2 classifier  
- ROI-level classification  
- Softmax confidence scores  
- Supports 6 PCB defect types:
  - Mousebite  
  - Open  
  - Pin Hole  
  - Short  
  - Spur  
  - Spurious Copper  

---

###  **Image Processing**
- ORB-based image alignment  
- Image subtraction + Otsu thresholding  
- Morphology for noise removal  
- Contour detection & ROI extraction  

---

### **Web App (Streamlit)**
- Upload template + test images  
- Real-time defect detection  
- Annotated output with bounding boxes  
- ROI-level detailed view  
- Download annotated image  
- Auto-generated PDF report (summary + charts + logs)  

---

##  Dataset

Used **only during training**, not required for inference.

**DeepPCB Dataset**  
🔗 https://www.dropbox.com/scl/fi/4vrtqn7t001yl41oucflu/PCB_DATASET.zip
  
---

##  Workflow Overview

1️⃣ Image Alignment (ORB + Homography)  
2️⃣ Subtraction & Thresholding  
3️⃣ ROI Extraction  
4️⃣ MobileNetV2 Classification  
5️⃣ Annotation + PDF Report  

---

##  Installation

### 1. Clone Repository
```sh
git clone https://github.com/Akshaya-41/circuitGuard
cd circuitGuard 
```
### 2. Install Dependencies
```sh
pip install -r requirements.txt
```
### 3. Run Web App
```sh
streamlit run app_integrated.py
```
## 🛠 Tech Stack
- **Image Processing:** OpenCV, NumPy  
- **Deep Learning:** PyTorch, MobileNetV2  
- **Frontend:** Streamlit  
- **Reporting:** Matplotlib, ReportLab  
- **Others:** Pillow

##  Results Summary
**Model:** MobileNetV2 | **Device:** CPU | **Input Size:** 128×128  
**Best Validation Accuracy:** 99.24% | **Test Accuracy:** 99.09%  
**Per-Class Accuracy:** mousebite 98.06%, open 99.30%, pin_hole 98.88%, short 100%, spur 98.48%, spurious_copper 100%  
**Average FP Rate:** 0.18% | **Average FN Rate:** 0.91%  
**Inference Time (per PCB):** ~5s (alignment 0.93s, detection 0.03s, ROI 0.01s, classification 4.06s)  



