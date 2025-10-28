from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import tempfile
import sys
import gc


# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import CircuitGuardPipeline

app = Flask(__name__)
# limit request size to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# allow origins from env or default
allowed_origins = os.environ.get('FRONTEND_ORIGINS', 'https://circuit-guard.vercel.app,http://localhost:3000')
print(f"🔒 Allowed CORS origins: {allowed_origins}")

# Split origins string and clean
origins = [o.strip() for o in allowed_origins.split(',') if o.strip()]

# Enable CORS with proper config
CORS(app, 
     origins=origins,
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type'],
     supports_credentials=True,
     max_age=3600)

# lazy pipeline (do not load model at import)
_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = CircuitGuardPipeline()  # heavy init deferred
    return _pipeline

# max image dimension (pixels) to downscale uploads
MAX_IMAGE_DIM = int(os.environ.get('MAX_IMAGE_DIM', 1024))

def save_and_resize(file_storage, max_dim=MAX_IMAGE_DIM, jpeg_quality=85):
    """Read uploaded file, downscale if large, save to temp file, return path"""
    # read bytes and decode to image
    data = file_storage.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a valid image")
    h, w = img.shape[:2]
    scale = max(h, w) / float(max_dim)
    if scale > 1.0:
        new_w = int(w / scale)
        new_h = int(h / scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(tmp.name, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    return tmp.name

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
        # Check files
        if 'template' not in request.files or 'test' not in request.files:
            return jsonify({'error': 'Both template and test images are required'}), 400

        template_file = request.files['template']
        test_file = request.files['test']

        if template_file.filename == '' or test_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400

        # Save & downscale uploads to reduce memory use
        template_path = None
        test_path = None
        try:
            template_path = save_and_resize(template_file)
            test_path = save_and_resize(test_file)

            # instantiate pipeline lazily
            pipeline = get_pipeline()

            # run pipeline inside try to catch MemoryError
            try:
                results = pipeline.process_image_pair(template_path, test_path)
            except MemoryError:
                # free memory and return 503 so Render doesn't keep failing
                gc.collect()
                return jsonify({'error': 'Server out of memory, try smaller images or upgrade instance'}), 503

            if results is None:
                return jsonify({'error': 'Failed to process images'}), 500

            # Prepare response (same as before) ...
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

            response_data['frequency_analysis'] = get_frequency_analysis(response_data['defects'])
            response_data['confidence_stats'] = get_confidence_stats(response_data['defects'])

            return jsonify(response_data)

        finally:
            # cleanup temp files ASAP and free memory
            try:
                if template_path and os.path.exists(template_path):
                    os.unlink(template_path)
                if test_path and os.path.exists(test_path):
                    os.unlink(test_path)
            except Exception:
                pass
            gc.collect()

    except Exception as e:
        # if MemoryError bubbled here
        if isinstance(e, MemoryError):
            gc.collect()
            return jsonify({'error': 'Server out of memory, try smaller images or upgrade instance'}), 503
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

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
    app.run(debug=True, host='0.0.0.0', port=5000)
