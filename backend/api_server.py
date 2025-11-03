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
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend
from io import BytesIO

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import CircuitGuardPipeline

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

def create_defect_type_bar_chart(frequency_analysis, width=400, height=250):
    """Create a bar chart for defect type distribution"""
    drawing = Drawing(width, height)
    
    if not frequency_analysis:
        return drawing
    
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 50
    chart.width = width - 100
    chart.height = height - 100
    
    # Prepare data
    defect_types = list(frequency_analysis.keys())
    counts = [frequency_analysis[dt]['count'] for dt in defect_types]
    
    chart.data = [counts]
    chart.categoryAxis.categoryNames = [dt.replace('_', ' ').title() for dt in defect_types]
    
    # Styling
    chart.bars[0].fillColor = colors.HexColor('#667eea')
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(counts) * 1.2 if counts else 10
    chart.valueAxis.valueStep = max(1, max(counts) // 5) if counts else 1
    
    chart.categoryAxis.labels.boxAnchor = 'ne'
    chart.categoryAxis.labels.dx = -5
    chart.categoryAxis.labels.dy = -5
    chart.categoryAxis.labels.angle = 30
    chart.categoryAxis.labels.fontSize = 9
    
    chart.valueAxis.labels.fontSize = 9
    
    drawing.add(chart)
    return drawing

def create_confidence_distribution_chart(confidence_stats, width=400, height=250):
    """Create a bar chart for confidence level distribution"""
    drawing = Drawing(width, height)
    
    if not confidence_stats:
        return drawing
    
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 50
    chart.width = width - 100
    chart.height = height - 100
    
    # Data: High, Medium, Low confidence counts
    data = [
        confidence_stats.get('high_confidence_count', 0),
        confidence_stats.get('medium_confidence_count', 0),
        confidence_stats.get('low_confidence_count', 0)
    ]
    
    chart.data = [data]
    chart.categoryAxis.categoryNames = ['High (≥80%)', 'Medium (50-80%)', 'Low (<50%)']
    
    # Color coding
    chart.bars[0].fillColor = colors.HexColor('#10b981')
    chart.bars[1].fillColor = colors.HexColor('#f59e0b')
    chart.bars[2].fillColor = colors.HexColor('#ef4444')
    
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(data) * 1.2 if max(data) > 0 else 10
    chart.valueAxis.valueStep = max(1, max(data) // 5) if max(data) > 0 else 1
    
    chart.categoryAxis.labels.fontSize = 9
    chart.categoryAxis.labels.boxAnchor = 'ne'
    chart.categoryAxis.labels.dx = -5
    chart.categoryAxis.labels.dy = -5
    chart.categoryAxis.labels.angle = 20
    
    chart.valueAxis.labels.fontSize = 9
    
    drawing.add(chart)
    return drawing

def create_defect_size_distribution_chart(defects, width=400, height=250):
    """Create a bar chart showing defect size categories"""
    drawing = Drawing(width, height)
    
    if not defects:
        return drawing
    
    # Categorize defects by size (area in pixels)
    size_categories = {
        'Small (<100px²)': 0,
        'Medium (100-500px²)': 0,
        'Large (500-1000px²)': 0,
        'Very Large (>1000px²)': 0
    }
    
    for defect in defects:
        area = defect.get('area', 0)
        if area < 100:
            size_categories['Small (<100px²)'] += 1
        elif area < 500:
            size_categories['Medium (100-500px²)'] += 1
        elif area < 1000:
            size_categories['Large (500-1000px²)'] += 1
        else:
            size_categories['Very Large (>1000px²)'] += 1
    
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 50
    chart.width = width - 100
    chart.height = height - 100
    
    counts = list(size_categories.values())
    chart.data = [counts]
    chart.categoryAxis.categoryNames = list(size_categories.keys())
    
    # Gradient colors
    chart.bars[0].fillColor = colors.HexColor('#8b5cf6')
    
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(counts) * 1.2 if max(counts) > 0 else 10
    chart.valueAxis.valueStep = max(1, max(counts) // 5) if max(counts) > 0 else 1
    
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.boxAnchor = 'ne'
    chart.categoryAxis.labels.dx = -5
    chart.categoryAxis.labels.dy = -5
    chart.categoryAxis.labels.angle = 25
    
    chart.valueAxis.labels.fontSize = 9
    
    drawing.add(chart)
    return drawing

def create_defect_type_pie_chart(frequency_analysis, width=400, height=250):
    """Create a pie chart for defect type distribution"""
    drawing = Drawing(width, height)
    
    if not frequency_analysis:
        return drawing
    
    pie = Pie()
    pie.x = 150
    pie.y = 50
    pie.width = 150
    pie.height = 150
    
    # Prepare data
    defect_types = list(frequency_analysis.keys())
    counts = [frequency_analysis[dt]['count'] for dt in defect_types]
    
    pie.data = counts
    pie.labels = [dt.replace('_', ' ').title() for dt in defect_types]
    
    # Color scheme
    colors_list = [
        colors.HexColor('#667eea'),
        colors.HexColor('#764ba2'),
        colors.HexColor('#f59e0b'),
        colors.HexColor('#10b981'),
        colors.HexColor('#ef4444'),
        colors.HexColor('#8b5cf6')
    ]
    
    for i in range(len(counts)):
        pie.slices[i].fillColor = colors_list[i % len(colors_list)]
    
    pie.slices.strokeWidth = 1
    pie.slices.strokeColor = colors.white
    
    # Add legend
    legend = Legend()
    legend.x = 20
    legend.y = height - 40
    legend.dx = 8
    legend.dy = 8
    legend.fontName = 'Helvetica'
    legend.fontSize = 8
    legend.boxAnchor = 'nw'
    legend.columnMaximum = 6
    legend.strokeWidth = 0
    legend.strokeColor = colors.white
    legend.deltax = 70
    legend.deltay = 10
    legend.autoXPadding = 5
    legend.yGap = 0
    legend.dxTextSpace = 5
    legend.alignment = 'right'
    legend.dividerLines = 1|2|4
    legend.dividerOffsY = 4.5
    legend.subCols.rpad = 30
    
    legend.colorNamePairs = [(pie.slices[i].fillColor, pie.labels[i]) for i in range(len(counts))]
    
    drawing.add(pie)
    drawing.add(legend)
    
    return drawing

def generate_pdf_report(results, defects, defect_count, frequency_analysis, confidence_stats, processing_time):
    """Generate PDF report with images and analysis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a73e8'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1a73e8'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("CircuitGuard Defect Detection Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Summary Information
    story.append(Paragraph("Report Summary", heading_style))
    quality_status = 'PASS - No Defects' if defect_count == 0 else f'FAIL - {defect_count} Defects Found'
    summary_data = [
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Processing Time:', f'{processing_time:.2f} seconds'],
        ['Total Defects Found:', str(defect_count)],
        ['Quality Status:', quality_status]
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Save images to temporary files for PDF
    temp_images = []
    try:
        # Annotated Result Image
        story.append(Paragraph("Annotated Result", heading_style))
        result_img_path = save_cv2_image_temp(results['result'])
        temp_images.append(result_img_path)
        story.append(Image(result_img_path, width=5*inch, height=4.75*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # Charts Section (only if defects exist)
        if defect_count > 0:
            story.append(PageBreak())
            story.append(Paragraph("Statistical Analysis", heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Defect Type Distribution - Bar Chart
            story.append(Paragraph("<b>Defect Type Distribution (Bar Chart)</b>", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            bar_chart = create_defect_type_bar_chart(frequency_analysis)
            story.append(bar_chart)
            story.append(Spacer(1, 0.3*inch))
            
            # Defect Type Distribution - Pie Chart
            story.append(Paragraph("<b>Defect Type Distribution (Pie Chart)</b>", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            pie_chart = create_defect_type_pie_chart(frequency_analysis)
            story.append(pie_chart)
            story.append(Spacer(1, 0.3*inch))
            
            # Confidence Distribution
            story.append(PageBreak())
            story.append(Paragraph("<b>Confidence Level Distribution</b>", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            conf_chart = create_confidence_distribution_chart(confidence_stats)
            story.append(conf_chart)
            story.append(Spacer(1, 0.3*inch))
            
            # Defect Size Distribution
            story.append(Paragraph("<b>Defect Size Distribution</b>", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            size_chart = create_defect_size_distribution_chart(defects)
            story.append(size_chart)
            story.append(Spacer(1, 0.3*inch))

            # Defect Heatmap
            story.append(PageBreak())
            story.append(Paragraph("<b>Defect Location Heatmap</b>", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            heatmap = build_defect_heatmap(results['test'].shape, defects)
            if heatmap is not None:
                heatmap_path = save_cv2_image_temp(heatmap)
                temp_images.append(heatmap_path)
                story.append(Image(heatmap_path, width=5*inch, height=4*inch))
                story.append(Spacer(1, 0.3*inch))
        
        # Frequency Analysis Table
        if frequency_analysis and defect_count > 0:
            story.append(Paragraph("Defect Type Distribution Table", heading_style))
            freq_data = [['Defect Type', 'Count', 'Percentage']]
            for defect_type, data in frequency_analysis.items():
                freq_data.append([
                    defect_type.replace('_', ' ').title(),
                    str(data['count']),
                    f"{data['percentage']}%"
                ])
            
            freq_table = Table(freq_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
            freq_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ]))
            story.append(freq_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Confidence Statistics
        if confidence_stats and defect_count > 0:
            story.append(Paragraph("Confidence Statistics", heading_style))
            conf_data = [
                ['Metric', 'Value'],
                ['Average Confidence', f"{confidence_stats['average']*100:.1f}%"],
                ['Minimum Confidence', f"{confidence_stats['min']*100:.1f}%"],
                ['Maximum Confidence', f"{confidence_stats['max']*100:.1f}%"],
                ['High Confidence (≥80%)', str(confidence_stats['high_confidence_count'])],
                ['Medium Confidence (50-80%)', str(confidence_stats['medium_confidence_count'])],
                ['Low Confidence (<50%)', str(confidence_stats['low_confidence_count'])]
            ]
            
            conf_table = Table(conf_data, colWidths=[3*inch, 2*inch])
            conf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ]))
            story.append(conf_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Detailed Defects List
        if defects:
            story.append(Paragraph("Detailed Defect List", heading_style))
            defect_data = [['ID', 'Type', 'Confidence', 'Position (x,y)', 'Size (w×h)', 'Area (px²)']]
            
            for defect in defects:
                defect_data.append([
                    str(defect['id']),
                    defect['class_name'].replace('_', ' ').title(),
                    f"{defect['confidence']*100:.1f}%",
                    f"({defect['bbox']['x']}, {defect['bbox']['y']})",
                    f"{defect['bbox']['width']}×{defect['bbox']['height']}",
                    str(defect['area'])
                ])
            
            defect_table = Table(defect_data, colWidths=[0.4*inch, 1.5*inch, 0.9*inch, 1.1*inch, 0.9*inch, 0.8*inch])
            defect_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ]))
            story.append(defect_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Reference Images Section
        story.append(PageBreak())
        story.append(Paragraph("Reference Images", heading_style))
        
        # Template Image
        story.append(Paragraph("<b>Template Image (Defect-free Reference)</b>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        template_img_path = save_cv2_image_temp(results['template'])
        temp_images.append(template_img_path)
        story.append(Image(template_img_path, width=4*inch, height=3*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # Test Image
        story.append(Paragraph("<b>Test Image (Inspected PCB)</b>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        test_img_path = save_cv2_image_temp(results['test'])
        temp_images.append(test_img_path)
        story.append(Image(test_img_path, width=4*inch, height=3*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # Defect Mask
        story.append(PageBreak())
        story.append(Paragraph("<b>Defect Detection Mask</b>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        mask_img_path = save_cv2_image_temp(results['defect_mask'])
        temp_images.append(mask_img_path)
        story.append(Image(mask_img_path, width=4*inch, height=3*inch))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
        
    finally:
        # Clean up temporary image files
        for img_path in temp_images:
            if os.path.exists(img_path):
                os.unlink(img_path)

def build_defect_heatmap(base_shape, defects_list):
    """Generate a heatmap visualization of defect locations"""
    h, w = base_shape[:2]
    if h <= 0 or w <= 0:
        return None
    heat = np.zeros((h, w), dtype=np.float32)
    for d in defects_list:
        cx = int(d.get('center', {}).get('x', d.get('bbox', {}).get('x', 0)))
        cy = int(d.get('center', {}).get('y', d.get('bbox', {}).get('y', 0)))
        cv2.circle(heat, (max(0, min(w-1, cx)), max(0, min(h-1, cy))), 20, 1.0, -1)
    if np.max(heat) > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), 15)
        heat = (255 * (heat / np.max(heat))).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    # Light background for nicer look
    bg = np.full_like(heat_color, 245)
    overlay = cv2.addWeighted(bg, 0.65, heat_color, 0.35, 0)
    return overlay

def save_cv2_image_temp(cv2_image):
    """Save OpenCV image to temporary file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_file.name, cv2_image)
    return temp_file.name

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