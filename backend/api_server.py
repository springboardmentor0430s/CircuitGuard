from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import tempfile
import sys
 

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import CircuitGuardPipeline
from pdf_report_utils import (
    generate_pdf_report,
)

app = Flask(__name__)
CORS(app)

# Initialize the pipeline
pipeline = CircuitGuardPipeline()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'CircuitGuard API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/process', methods=['POST'])
def process_images():
    """Process template and test images for defect detection"""
    try:
        # Check if files are present
        if 'template' not in request.files or 'test' not in request.files:
            return jsonify({'error': 'Both template and test images are required'}), 400
        
        template_file = request.files['template']
        test_file = request.files['test']
        
        if template_file.filename == '' or test_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_template:
            template_file.save(temp_template.name)
            template_path = temp_template.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_test:
            test_file.save(temp_test.name)
            test_path = temp_test.name
        
        try:
            # Process images
            results = pipeline.process_image_pair(template_path, test_path)
            
            if results is None:
                return jsonify({'error': 'Failed to process images'}), 500
            
            # Prepare response data
            response_data = {
                'success': True,
                'defect_count': results['defect_count'],
                'defects': [],
                'processing_time': datetime.now().isoformat(),
                'images': {
                    'template': encode_image_to_base64(results['template']),
                    'test': encode_image_to_base64(results['test']),
                    'defect_mask': encode_image_to_base64(results['defect_mask']),
                    'result': encode_image_to_base64(results['result'])
                }
            }
            
            # Process defects data
            for i, classification in enumerate(results['classifications']):
                if i < len(results['bounding_boxes']):
                    x, y, w, h = results['bounding_boxes'][i]
                    defect_data = {
                        'id': i + 1,
                        'class_name': classification['class_name'],
                        'class_id': classification['class_id'],
                        'confidence': classification.get('confidence', 0.0),
                        'bbox': {
                            'x': int(x),
                            'y': int(y),
                            'width': int(w),
                            'height': int(h)
                        },
                        'area': int(w * h),
                        'center': {
                            'x': int(x + w/2),
                            'y': int(y + h/2)
                        }
                    }
                    response_data['defects'].append(defect_data)
            
            # Add frequency analysis
            response_data['frequency_analysis'] = get_frequency_analysis(response_data['defects'])
            
            # Add confidence statistics
            response_data['confidence_stats'] = get_confidence_stats(response_data['defects'])
            
            return jsonify(response_data)
            
        finally:
            # Clean up temporary files
            if os.path.exists(template_path):
                os.unlink(template_path)
            if os.path.exists(test_path):
                os.unlink(test_path)
                
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/process-pdf', methods=['POST'])
def process_images_pdf():
    """Process images and return PDF report"""
    try:
        start_time = datetime.now()
        
        # Check if files are present
        if 'template' not in request.files or 'test' not in request.files:
            return jsonify({'error': 'Both template and test images are required'}), 400
        
        template_file = request.files['template']
        test_file = request.files['test']
        
        if template_file.filename == '' or test_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_template:
            template_file.save(temp_template.name)
            template_path = temp_template.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_test:
            test_file.save(temp_test.name)
            test_path = temp_test.name
        
        try:
            # Process images
            results = pipeline.process_image_pair(template_path, test_path)
            
            if results is None:
                return jsonify({'error': 'Failed to process images'}), 500
            
            # Prepare defects data
            defects = []
            for i, classification in enumerate(results['classifications']):
                if i < len(results['bounding_boxes']):
                    x, y, w, h = results['bounding_boxes'][i]
                    defect_data = {
                        'id': i + 1,
                        'class_name': classification['class_name'],
                        'class_id': classification['class_id'],
                        'confidence': classification.get('confidence', 0.0),
                        'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                        'area': int(w * h),
                        'center': {'x': int(x + w/2), 'y': int(y + h/2)}
                    }
                    defects.append(defect_data)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Generate PDF
            pdf_buffer = generate_pdf_report(
                results=results,
                defects=defects,
                defect_count=results['defect_count'],
                frequency_analysis=get_frequency_analysis(defects),
                confidence_stats=get_confidence_stats(defects),
                processing_time=processing_time
            )
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'CircuitGuard_Report_{timestamp}.pdf'
            
            return send_file(
                pdf_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
            
        finally:
            # Clean up temporary files
            if os.path.exists(template_path):
                os.unlink(template_path)
            if os.path.exists(test_path):
                os.unlink(test_path)
                
    except Exception as e:
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500


def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    if image is None:
        return None
    
    # Convert BGR to RGB for web display
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', image_rgb)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"

def get_frequency_analysis(defects):
    """Calculate frequency analysis of defects"""
    if not defects:
        return {}
    
    frequency = {}
    for defect in defects:
        class_name = defect['class_name']
        frequency[class_name] = frequency.get(class_name, 0) + 1
    
    # Calculate percentages
    total = len(defects)
    frequency_percentages = {}
    for class_name, count in frequency.items():
        frequency_percentages[class_name] = {
            'count': count,
            'percentage': round((count / total) * 100, 2)
        }
    
    return frequency_percentages

def get_confidence_stats(defects):
    """Calculate confidence statistics"""
    if not defects:
        return {}
    
    confidences = [defect.get('confidence', 0) for defect in defects]
    
    return {
        'average': round(sum(confidences) / len(confidences), 3),
        'min': round(min(confidences), 3),
        'max': round(max(confidences), 3),
        'high_confidence_count': sum(1 for c in confidences if c > 0.8),
        'medium_confidence_count': sum(1 for c in confidences if 0.5 <= c <= 0.8),
        'low_confidence_count': sum(1 for c in confidences if c < 0.5)
    }

@app.route('/api/defect-types', methods=['GET'])
def get_defect_types():
    """Get available defect types"""
    return jsonify({
        'defect_types': [
            {'id': 0, 'name': 'missing_hole', 'display_name': 'Missing Hole'},
            {'id': 1, 'name': 'mouse_bite', 'display_name': 'Mouse Bite'},
            {'id': 2, 'name': 'open_circuit', 'display_name': 'Open Circuit'},
            {'id': 3, 'name': 'short', 'display_name': 'Short'},
            {'id': 4, 'name': 'spur', 'display_name': 'Spur'},
            {'id': 5, 'name': 'spurious_copper', 'display_name': 'Spurious Copper'}
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting CircuitGuard API Server...")
    print("📡 API will be available at: http://localhost:5000")
    print("🔗 Health check: http://localhost:5000/api/health")
    print("📄 PDF endpoint: http://localhost:5000/api/process-pdf")
    app.run(debug=True, host='0.0.0.0', port=5000)