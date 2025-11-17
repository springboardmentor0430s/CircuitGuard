## Backend Setup (Flask + CV Pipeline)

This guide walks through creating a Python environment, installing dependencies, and running the API server.

### Prerequisites
- Python 3.9+ recommended
- Windows: PowerShell or cmd
- Optional: CUDA-capable GPU with compatible PyTorch build (CPU works too)

### 1) Create and activate a virtual environment
```bash
cd backend
python -m venv .venv

# Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1

# Windows cmd (alternative)
.\\.venv\\Scripts\\activate.bat
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you need GPU-accelerated PyTorch, install the appropriate build from the official site instructions.

### 3) Run the API server
```bash
python api_server.py
```
The server runs at `http://localhost:5000` with endpoints:
- `/api/health`
- `/api/process` (POST with `template`, `test`)
- `/api/process-pdf` (POST with `template`, `test`)

### 4) Model weights (optional classification)
- Place a trained checkpoint at `backend/model/best_model.pth` to enable classification.
- Without it, the pipeline still performs detection and returns results, but class names default to `unknown`.

### 5) Troubleshooting
- OpenCV import errors: ensure `opencv-python` installed and correct Python version used.
- Torch not found: verify `pip show torch` in the same venv.
- Large images/slow performance: consider resizing inputs or using GPU PyTorch.


