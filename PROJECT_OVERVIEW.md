## CircuitGuard - PCB Defect Detection System

CircuitGuard is an end-to-end system for PCB defect detection and reporting. It consists of a React-based frontend and a Flask backend that runs a computer-vision pipeline with optional classification using a PyTorch EfficientNet model. The system supports visual inspection, statistical analysis, and PDF report generation.

### Tech Stack
- Frontend: React 19, Axios, Recharts, react-scripts, lucide-react
- Backend: Python (Flask, Flask-CORS), OpenCV, NumPy, PyTorch, ReportLab, scikit-learn, timm

### Repository Structure
```
.
├─ backend/
│  ├─ api_server.py            # Flask app with /api/process and /api/process-pdf
│  ├─ main.py                  # CircuitGuardPipeline and CLI for batch processing
│  ├─ pdf_report_utils.py      # ReportLab PDF generation utilities
│  ├─ requirements.txt         # Backend Python dependencies
│  ├─ model/                   # EfficientNet model definition, training, evaluation
│  ├─ src/
│  │  ├─ preprocessing/        # alignment, binary defect detection, contour extraction
│  │  ├─ inference/            # prediction, integration examples
│  │  └─ data/                 # dataset utilities
│  └─ scripts/                 # training and evaluation scripts
├─ frontend/
│  ├─ package.json             # React app, proxy to backend @ :5000
│  └─ src/                     # App.js talks to backend API
```


## Install the .pth file from here and place it inside backend/model
```
https://drive.google.com/file/d/1TsArrsEhypOJEkUwJpmPD_lvAg5xteXe/view?usp=sharing
```

### High-level Architecture

```mermaid
flowchart LR

User[User] --> UI[React Frontend (App.js)]

subgraph Frontend [Frontend: React]
  UI -->|POST /api/process| Axios
  UI -->|POST /api/process-pdf| Axios
  UI -.->|GET /api/defect-types| Axios
end

Axios -->|HTTP| API[(Flask API)]

subgraph Backend [Backend: Flask + CV Pipeline]
  API -->|calls| Pipeline[CircuitGuardPipeline (main.py)]
  Pipeline --> Align[src.preprocessing.alignment]
  Pipeline --> Binary[src.preprocessing.binary_defect_detection]
  Pipeline --> Contours[src.preprocessing.contour_detection]
  Pipeline --> Classifier[model.efficientnet.PCBDefectClassifier]
  Pipeline --> Utils[pdf_report_utils.generate_pdf_report]
end

Classifier -->|PyTorch| Model[(best_model.pth)]

Pipeline --> Results[(Annotated images, defects, stats)]
API -->|JSON/Base64 images| UI
API -->|PDF stream| UI
```

### Data Flow
1. Frontend uploads two images (template and test) via `multipart/form-data`.
2. Backend `api_server.py` saves temporary copies and invokes `CircuitGuardPipeline.process_image_pair`.
3. Pipeline:
   - Aligns images
   - Performs binary defect detection (XOR-based)
   - Extracts contours and ROIs
   - Optionally classifies ROIs with EfficientNet if `model/best_model.pth` is present
   - Produces annotated result image and metrics
4. API returns:
   - `/api/process`: JSON with defects, stats, and base64-encoded images
   - `/api/process-pdf`: Generated PDF report (download attachment)

### API Endpoints
- GET `/api/health`: health check
- POST `/api/process`: accepts `template`, `test` files; returns JSON
- POST `/api/process-pdf`: accepts `template`, `test` files; returns PDF
- GET `/api/defect-types`: static list of supported defect classes

Example response fields from `/api/process`:
- `images`: `{ template, test, defect_mask, result }` as data URLs
- `defects`: array of `{ id, class_name, class_id, confidence, bbox, area, center }`
- `frequency_analysis`: `{ class_name: { count, percentage } }`
- `confidence_stats`: `{ average, min, max, high_confidence_count, medium_confidence_count, low_confidence_count }`

### Frontend Behavior
- Uses `axios` to call backend. Base URL defaults to:
  - `REACT_APP_API_URL` if set, else `http://localhost:5000`
- `package.json` includes `"proxy": "http://localhost:5000"` for local dev
- Visualizes annotated images and renders charts (Recharts) for frequency and confidence distributions
- Provides downloads for:
  - PDF Report (via `/api/process-pdf`)
  - Annotated image (from response)
  - CSV log of detections (generated in-browser)

### Backend Behavior
- Flask server (`backend/api_server.py`), CORS enabled
- Uses `CircuitGuardPipeline` from `backend/main.py`
- Optional classification depends on presence of `backend/model/best_model.pth`
- PDF generated using `pdf_report_utils.generate_pdf_report`

### Setup and Running
- See `backend/SETUP.md` for backend environment and server instructions
- See `frontend/SETUP.md` for frontend setup and development server

### Notes and Operational Considerations
- GPU optional. PyTorch runs on CPU if CUDA not available.
- Ensure images are reasonable resolution to avoid heavy processing time.
- For production, run Flask behind a proper WSGI server and configure CORS, logging, and error handling accordingly.


