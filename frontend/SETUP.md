## Frontend Setup (React)

This guide covers installing dependencies and running the React development server.

### Prerequisites
- Node.js 16+ and npm

### 1) Install dependencies
```bash
cd frontend
npm install
```

### 2) Configure API URL (optional)
By default, the app uses:
- `REACT_APP_API_URL` if defined
- otherwise `http://localhost:5000`

For local development, `package.json` also defines:
```json
"proxy": "http://localhost:5000"
```
So you can typically just run the backend on port 5000 and use `npm start`.

To override:
```bash
# PowerShell example
$env:REACT_APP_API_URL="http://localhost:5000"
npm start
```

### 3) Start the dev server
```bash
npm start
```
Open `http://localhost:3000` in your browser.

### 4) Usage
1. Upload two images:
   - Template image (defect-free reference)
   - Test image (to inspect)
2. Click “Detect Defects” to process.
3. Download:
   - Report (PDF) via backend `/api/process-pdf`
   - Annotated image (from results)
   - Logs (CSV) produced client-side

### 5) Build for production
```bash
npm run build
```
Generates a production build in `frontend/build`.


