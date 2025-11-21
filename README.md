📘 CircuitGuard — PCB Defect Detector

An automated PCB (Printed Circuit Board) defect detection system built using Streamlit, OpenCV, and ReportLab.
The application compares a template (golden) PCB image with a test PCB, detects manufacturing defects, classifies them, and generates a detailed PDF report containing visuals, metrics, charts, and logs.


---

🚀 Features

✔ Upload Template and Test PCB images
✔ Automatic alignment, subtraction, contour detection, and defect classification
✔ Displays:

Annotated PCB output

Difference map

Defect mask
✔ Pie chart & bar chart visualization
✔ Auto-generated Executive Summary
✔ Processing logs
✔ Export options:

Annotated Image (PNG)

Logs (CSV)

Full PDF Report



---

📁 Project Structure

CircuitGuard_PCB_Project/
│
├── backend/
│   ├── inference.py     # Core algorithm: detection, subtraction, classification
│
├── app.py               # Main Streamlit UI (your provided code)
├── README.md            # You are reading this file
└── requirements.txt     # Dependencies


---

🛠 Installation

1️⃣ Create a virtual environment

python -m venv .venv

2️⃣ Activate it

Windows PowerShell:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1

3️⃣ Install dependencies

pip install -r requirements.txt


---

▶ Run the Application

streamlit run app.py

The browser will open automatically at:

http://localhost:8501


---

📥 Usage

1. Upload Template Image (clean PCB)


2. Upload Test Image (defective PCB)


3. Click Run Inference


4. View:

Annotated PCB

Difference & Mask images

Defect charts

Executive summary

Logs



5. Export results using:

Download Annotated Image

Download Logs (CSV)

Download PDF Report





---

📄 Generated PDF Includes

Header + timestamp

Executive summary

Metrics table

Defect distribution charts

Processing logs

Annotated PCB image



---

⚙ Module 7: Testing, Evaluation, and Exporting Results

✔ Tasks Completed

Added export buttons for:

Annotated PCB image

Processing logs (CSV)

Full PDF report


Integrated chart export

Tested with multiple PCB image pairs

Code structured for speed optimization


✔ Deliverables

Fully functional Streamlit web app

Annotated output images

CSV log download

PDF downloadable report



---

📦 Dependencies

Add this in requirements.txt:

streamlit
opencv-python
numpy
pillow
matplotlib
reportlab


---

🧩 Backend Overview (inference.py)

Your run_inference_on_pair() function should return:

annotated_image
difference_map
mask_image
logs
stats

Where:

logs → list of text messages

stats → dictionary example:


{
  "Open Circuit": 4,
  "Short Circuit": 2,
  "Mousebite": 1,
  "Spur": 1
}


---

🧪 Testing Checklist

[ ] Verify alignment works with rotated/shifted PCBs

[ ] Confirm classification labels are correct

[ ] Test with 10+ sample PCB pairs

[ ] Check PDF charts scale correctly

[ ] Validate log export

[ ] Monitor Streamlit 
 