
# PCB DEFECT DETECTION SYSTEM
What it Does:

PCB Vision is an automated quality control system that uses artificial intelligence and computer vision to detect manufacturing defects in printed circuit boards. The system analyzes high-resolution images of PCBs, identifying six critical defect types—including short circuits, open circuits, mouse bites, and spurious copper—with 99.1% accuracy. It processes each board in under 3 seconds, generates detailed inspection reports, and provides real-time quality analytics through an intuitive web dashboard.

Who It's For:

This solution is designed for electronics manufacturers, PCB fabrication facilities, and quality assurance teams across the electronics industry. It serves both large-scale production facilities seeking to automate their quality control processes and small to medium manufacturers looking for cost-effective alternatives to expensive commercial AOI systems. Quality managers, production engineers, and manufacturing operations directors will benefit from its consistent inspection capabilities, reduced labor costs, and comprehensive defect documentation.


# Acknowledgements

 **Acknowledgements**

We extend our sincere gratitude to our project mentor, **Mr. Avinash**, for his invaluable guidance, expert insights, and continuous support throughout the development of this project. His mentorship and encouragement were pivotal in navigating challenges and achieving our goals.

We would also like to thank **Infosys Springboard** for providing a robust learning platform, structured curriculum, and essential resources that helped us build and apply our skills in machine learning and computer vision.

Our appreciation goes to the contributors of the **DeepPCB dataset** and the open-source community behind libraries such as **PyTorch, OpenCV, and Streamlit**, which were fundamental to the implementation of this system.

Lastly, we thank our peers and colleagues for their collaboration, feedback, and motivation throughout this journey. This accomplishment is a reflection of shared effort and collective dedication.

# Tech Stack

**Client-Side Technologies**
- **React** - Component-based UI library for building interactive user interfaces
- **Redux** - State management for predictable application state container
- **TailwindCSS** - Utility-first CSS framework for rapid UI development

**Server-Side Technologies**
- **Node.js** - JavaScript runtime environment for server-side execution
- **Express.js** - Web application framework for building RESTful APIs and server logic


# API Reference

## Base URL
```
http://localhost:5000/api/v1
```

## Endpoints

### 1. Health Check
**GET /health**  
Check API server status  
**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

### 2. Analyze PCB Images
**POST /analyze**  
Upload and analyze PCB images for defects  
**Body:** `multipart/form-data`  
**Parameters:**
- `template_image`: File (required)
- `test_image`: File (required)  
- `sensitivity`: number (optional, default: 0.8)

**Response:**
```json
{
  "analysis_id": "PCB-ANALYSIS-123456789",
  "status": "completed",
  "overall_score": 96,
  "total_defects": 7,
  "defects": [...]
}
```

### 3. Get Results
**GET /results/{analysis_id}**  
Retrieve specific analysis results

### 4. Export Report
**GET /export/{analysis_id}**  
Export report in CSV, JSON, or PDF format  
**Query:** `?format=csv|json|pdf`

### 5. System Metrics
**GET /metrics**  
Get system performance and statistics

---

## Error Responses
**400 Bad Request:**
```json
{
  "error": "Invalid input",
  "message": "Error description"
}
```

**500 Server Error:**
```json
{
  "error": "Internal server error",
  "message": "Error description"
}
```
# Color Reference

## Primary Colors
- **Primary Blue**: `#2563eb` - Main brand color, buttons, primary actions
- **Primary Green**: `#10b981` - Success states, completed actions
- **Primary Red**: `#ef4444` - Error states, critical alerts
- **Primary Yellow**: `#f59e0b` - Warning states, pending actions

## Neutral Colors
- **Dark Gray**: `#1f2937` - Text, headings
- **Medium Gray**: `#6b7280` - Secondary text, labels
- **Light Gray**: `#e5e7eb` - Borders, dividers
- **Background**: `#f8fafc` - Page background

## Status Colors
- **Defect Critical**: `#dc2626` - Critical defects
- **Defect High**: `#ea580c` - High priority defects
- **Defect Medium**: `#d97706` - Medium priority defects
- **Defect Low**: `#65a30d` - Low priority defects

## Chart Colors
- **Dataset 1**: `#3b82f6` - Open Circuit defects
- **Dataset 2**: `#8b5cf6` - Short Circuit defects
- **Dataset 3**: `#06b6d4` - Mouse Bite defects
- **Dataset 4**: `#10b981` - Spurious Copper defects
- **Dataset 5**: `#f59e0b` - Spur defects
- **Dataset 6**: `#ef4444` - Pin Hole defects

## Usage Examples

```css
/* Primary Actions */
.primary-button {
  background-color: #2563eb;
  color: white;
}

/* Success States */
.success-alert {
  background-color: #10b981;
  color: white;
}

/* Defect Indicators */
.critical-defect {
  border-color: #dc2626;
  color: #dc2626;
}
```
# Deployment

To deploy this project run

```bash
  npm run deploy
  npm run build 
```


# Key Features

- **High Accuracy**: 99.1% defect detection using AI
- **Fast Processing**: Under 3 seconds per PCB analysis
- **Multiple Defects**: Detects 6+ types of PCB defects
- **Web Dashboard**: User-friendly interface for easy operation
- **Automated Reports**: Export results in CSV, PDF & JSON formats
- **Real-time Results**: Instant defect visualization and classification
- **Scalable System**: Handles high-volume manufacturing needs
- **Cost Effective**: Reduces manual inspection costs and errors

