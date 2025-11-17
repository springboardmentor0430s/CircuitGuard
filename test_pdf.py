import sys
print('Python version:', sys.version)
print('Python path:', sys.executable)

import reportlab
print('ReportLab version:', reportlab.__version__)

from src.web_app.pdf_report import create_pdf_report
import os

# Test data
test_result = {
    'summary': {
        'total_defects': 2,
        'average_confidence': 95.5,
        'alignment_matches': 100,
        'alignment_inliers': 90
    },
    'processing_time': '1.5s',
    'defect_details': [
        {'id': 1, 'type': 'mousebite', 'confidence (%)': 95.5, 'location': '(100, 100)', 'area (px²)': 50},
        {'id': 2, 'type': 'short', 'confidence (%)': 97.8, 'location': '(200, 200)', 'area (px²)': 75}
    ],
    'class_distribution': {
        'mousebite': 1,
        'short': 1
    }
}

# Create temp directory if it doesn't exist
os.makedirs('temp', exist_ok=True)

# Test PDF generation
test_path = 'temp/test_report.pdf'
print(f'Trying to create PDF at {test_path}')

try:
    create_pdf_report(test_result, test_path, ['mousebite', 'short', 'open', 'spur'])
    print(f'PDF creation successful! File exists: {os.path.exists(test_path)}')
    print(f'File size: {os.path.getsize(test_path)} bytes')
except Exception as e:
    print(f'Error creating PDF: {str(e)}')
    import traceback
    traceback.print_exc()