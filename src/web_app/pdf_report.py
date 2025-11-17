"""
Comprehensive PDF Report Generator for CircuitGuard-PCB
Includes all visualizations, statistical analysis, risk assessment, and recommendations
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph, 
                                Spacer, PageBreak, Image, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime
from PIL import Image as PILImage
import io
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class BorderedCanvas(canvas.Canvas):
    """Custom canvas with simple black border and headers/footers"""
    
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []
        
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        page_count = len(self.pages)
        for i, page in enumerate(self.pages):
            self.__dict__.update(page)
            self.draw_border_and_header(i + 1, page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_border_and_header(self, page_num, page_count):
        """Draw simple black border and header/footer on each page"""
        page_width, page_height = letter
        
        # Simple single black border
        self.setStrokeColor(colors.black)
        self.setLineWidth(2)
        self.rect(0.5*inch, 0.5*inch, page_width - 1*inch, page_height - 1*inch)
        
        # Header section background
        self.setFillColor(colors.HexColor('#1f77b4'))
        self.rect(0.5*inch, page_height - 1*inch, page_width - 1*inch, 0.4*inch, fill=1, stroke=0)
        
        # Header text
        self.setFillColor(colors.white)
        self.setFont("Helvetica-Bold", 14)
        self.drawString(0.7*inch, page_height - 0.8*inch, "CircuitGuard-PCB")
        
        self.setFont("Helvetica", 10)
        self.drawRightString(page_width - 0.7*inch, page_height - 0.8*inch, 
                            f"Inspection Report")
        
        # Footer
        self.setFillColor(colors.black)
        self.setFont("Helvetica", 8)
        self.drawString(0.7*inch, 0.6*inch, 
                       f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.drawCentredString(page_width/2, 0.6*inch, 
                              "Automated PCB Defect Detection & Classification System")
        self.drawRightString(page_width - 0.7*inch, 0.6*inch, 
                            f"Page {page_num} of {page_count}")


def generate_additional_charts(formatted_data, temp_dir='temp'):
    """Generate additional visualization charts for the report"""
    
    chart_paths = {}
    
    if not formatted_data['defect_details']:
        return chart_paths
    
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    defects = formatted_data['defect_details']
    
    # Extract data
    confidences = [d['confidence (%)'] for d in defects]
    areas = [d['area (px²)'] for d in defects]
    types = [d['type'] for d in defects]
    
    # 1. Box Plot - Confidence by Type
    try:
        plt.figure(figsize=(8, 5))
        type_conf_dict = {}
        for d in defects:
            if d['type'] not in type_conf_dict:
                type_conf_dict[d['type']] = []
            type_conf_dict[d['type']].append(d['confidence (%)'])
        
        plt.boxplot(type_conf_dict.values(), labels=[t.title() for t in type_conf_dict.keys()])
        plt.title('Confidence Distribution by Defect Type', fontsize=14, fontweight='bold')
        plt.ylabel('Confidence (%)')
        plt.xlabel('Defect Type')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        box_path = os.path.join(temp_dir, f'boxplot_{timestamp}.png')
        plt.savefig(box_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths['boxplot'] = box_path
    except Exception as e:
        print(f"Error generating box plot: {e}")
    
    # 2. Violin Plot - Size Distribution by Type
    try:
        plt.figure(figsize=(8, 5))
        type_size_dict = {}
        for d in defects:
            if d['type'] not in type_size_dict:
                type_size_dict[d['type']] = []
            type_size_dict[d['type']].append(d['area (px²)'])
        
        positions = range(1, len(type_size_dict) + 1)
        parts = plt.violinplot(type_size_dict.values(), positions=positions, showmeans=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        plt.xticks(positions, [t.title() for t in type_size_dict.keys()], rotation=45, ha='right')
        plt.title('Defect Size Distribution by Type', fontsize=14, fontweight='bold')
        plt.ylabel('Area (pixels²)')
        plt.xlabel('Defect Type')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        violin_path = os.path.join(temp_dir, f'violin_{timestamp}.png')
        plt.savefig(violin_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths['violin'] = violin_path
    except Exception as e:
        print(f"Error generating violin plot: {e}")
    
    # 3. Scatter Plot - Area vs Confidence
    try:
        plt.figure(figsize=(8, 5))
        
        unique_types = list(set(types))
        colors_map = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        
        for i, dtype in enumerate(unique_types):
            dtype_indices = [j for j, t in enumerate(types) if t == dtype]
            dtype_areas = [areas[j] for j in dtype_indices]
            dtype_confs = [confidences[j] for j in dtype_indices]
            plt.scatter(dtype_areas, dtype_confs, label=dtype.title(), 
                       alpha=0.6, s=100, c=[colors_map[i]])
        
        plt.title('Defect Area vs Confidence', fontsize=14, fontweight='bold')
        plt.xlabel('Area (pixels²)')
        plt.ylabel('Confidence (%)')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        scatter_path = os.path.join(temp_dir, f'scatter_{timestamp}.png')
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths['scatter'] = scatter_path
    except Exception as e:
        print(f"Error generating scatter plot: {e}")
    
    # 4. Correlation Heatmap
    try:
        # Extract locations
        locations_x = []
        locations_y = []
        for d in defects:
            loc_str = d['location'].strip('()').split(',')
            locations_x.append(int(loc_str[0]))
            locations_y.append(int(loc_str[1]))
        
        # Create correlation matrix
        data_matrix = np.array([locations_x, locations_y, confidences, areas]).T
        corr_matrix = np.corrcoef(data_matrix.T)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='coolwarm', 
                   xticklabels=['X Position', 'Y Position', 'Confidence', 'Area'],
                   yticklabels=['X Position', 'Y Position', 'Confidence', 'Area'],
                   center=0,
                   vmin=-1, vmax=1,
                   square=True,
                   linewidths=1)
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        heatmap_path = os.path.join(temp_dir, f'heatmap_{timestamp}.png')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths['heatmap'] = heatmap_path
    except Exception as e:
        print(f"Error generating heatmap: {e}")
    
    # 5. Defect Location Map
    try:
        plt.figure(figsize=(8, 6))
        
        # Create 2D histogram (heatmap of locations)
        plt.hist2d(locations_x, locations_y, bins=20, cmap='hot', alpha=0.8)
        plt.colorbar(label='Defect Density')
        
        # Overlay scatter points
        for i, dtype in enumerate(unique_types):
            dtype_indices = [j for j, t in enumerate(types) if t == dtype]
            dtype_x = [locations_x[j] for j in dtype_indices]
            dtype_y = [locations_y[j] for j in dtype_indices]
            plt.scatter(dtype_x, dtype_y, label=dtype.title(), 
                       alpha=0.7, s=80, edgecolors='white', linewidths=1.5)
        
        plt.title('Spatial Distribution of Defects', fontsize=14, fontweight='bold')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.legend(loc='best')
        plt.gca().invert_yaxis()  # Match image coordinates
        plt.tight_layout()
        
        location_path = os.path.join(temp_dir, f'location_{timestamp}.png')
        plt.savefig(location_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths['location_map'] = location_path
    except Exception as e:
        print(f"Error generating location map: {e}")
    
    # 6. Priority Matrix
    try:
        plt.figure(figsize=(8, 6))
        
        # Calculate priority score (high area + low confidence = high priority)
        max_area = max(areas) if areas else 1
        priorities = [(100 - c) * (a / max_area) for c, a in zip(confidences, areas)]
        
        scatter = plt.scatter(areas, confidences, c=priorities, s=150, 
                            cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidths=1)
        plt.colorbar(scatter, label='Priority Score')
        
        # Add quadrant lines
        plt.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=np.median(areas), color='gray', linestyle='--', alpha=0.5)
        
        # Label quadrants
        plt.text(0.95, 0.95, 'High Priority\n(Large, Low Conf)', 
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        plt.text(0.05, 0.05, 'Low Priority\n(Small, High Conf)', 
                transform=plt.gca().transAxes, ha='left', va='bottom',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        plt.title('Defect Priority Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Area (pixels²)')
        plt.ylabel('Confidence (%)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        priority_path = os.path.join(temp_dir, f'priority_{timestamp}.png')
        plt.savefig(priority_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths['priority_matrix'] = priority_path
    except Exception as e:
        print(f"Error generating priority matrix: {e}")
    
    return chart_paths


def calculate_quality_score(formatted_data):
    """Calculate overall quality score based on defects"""
    
    if formatted_data['summary']['total_defects'] == 0:
        return 100, "Excellent"
    
    defects = formatted_data['defect_details']
    
    # Base score
    score = 100
    
    # Deduct points based on number of defects
    defect_penalty = min(formatted_data['summary']['total_defects'] * 5, 50)
    score -= defect_penalty
    
    # Adjust based on average confidence (higher confidence defects = more penalty)
    avg_conf = np.mean([d['confidence (%)'] for d in defects])
    if avg_conf >= 95:
        score -= 10
    elif avg_conf >= 90:
        score -= 5
    
    # Adjust based on defect sizes
    avg_area = np.mean([d['area (px²)'] for d in defects])
    if avg_area > 100:
        score -= 10
    elif avg_area > 50:
        score -= 5
    
    score = max(0, score)
    
    if score >= 90:
        grade = "Excellent"
    elif score >= 80:
        grade = "Good"
    elif score >= 70:
        grade = "Acceptable"
    elif score >= 60:
        grade = "Poor"
    else:
        grade = "Critical"
    
    return score, grade


def assess_risk_level(formatted_data):
    """Assess overall risk level"""
    
    if formatted_data['summary']['total_defects'] == 0:
        return "LOW", "green"
    
    defects = formatted_data['defect_details']
    total = len(defects)
    
    # Count critical defects (high confidence, large area)
    critical_count = sum(1 for d in defects 
                        if d['confidence (%)'] >= 90 and d['area (px²)'] > 50)
    
    # Risk assessment
    if total >= 10 or critical_count >= 3:
        return "HIGH", "red"
    elif total >= 5 or critical_count >= 1:
        return "MEDIUM", "orange"
    else:
        return "LOW", "green"


def create_pdf_report(formatted_data, output_path, class_names, result_images=None, chart_paths=None):
    """
    Create comprehensive PDF inspection report with all visualizations
    """
    
    # Create PDF with custom canvas and proper margins
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1.2*inch,
        bottomMargin=0.9*inch
    )
    
    # Container for PDF elements
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        spaceBefore=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=15,
        spaceBefore=20,
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=colors.HexColor('#1f77b4'),
        borderPadding=8,
        backColor=colors.HexColor('#f0f8ff')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        spaceBefore=4
    )
    
    # Generate additional charts
    extra_charts = generate_additional_charts(formatted_data)
    if chart_paths:
        chart_paths.update(extra_charts)
    else:
        chart_paths = extra_charts
    
    # Calculate quality metrics
    quality_score, quality_grade = calculate_quality_score(formatted_data)
    risk_level, risk_color = assess_risk_level(formatted_data)
    
    # ===================== COVER PAGE =====================
    story.append(Spacer(1, 1*inch))
    
    # Title
    story.append(Paragraph("PCB INSPECTION REPORT", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Report info box
    report_info = [
        ['Report Type:', 'Automated Defect Detection & Classification'],
        ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['System:', 'CircuitGuard-PCB v1.0'],
        ['Model:', 'EfficientNet-B4'],
        ['Quality Score:', f'{quality_score}/100 ({quality_grade})'],
        ['Risk Level:', risk_level],
        ['Status:', 'PASSED' if formatted_data['summary']['total_defects'] == 0 else 'FAILED']
    ]
    
    info_table = Table(report_info, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f2ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1f77b4')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 0.5*inch))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    
    summary_text = f"""
    <b>Total Defects Detected:</b> {formatted_data['summary']['total_defects']}<br/>
    <b>Average Confidence:</b> {formatted_data['summary']['average_confidence']}<br/>
    <b>Processing Time:</b> {formatted_data['processing_time']}<br/>
    <b>Image Alignment Quality:</b> {formatted_data['summary']['alignment_matches']} matches, 
    {formatted_data['summary']['alignment_inliers']} inliers<br/>
    <b>Quality Score:</b> {quality_score}/100 ({quality_grade})<br/>
    <b>Risk Assessment:</b> <font color='{risk_color}'>{risk_level} RISK</font><br/>
    """
    
    if formatted_data['summary']['total_defects'] == 0:
        summary_text += "<br/><b>Result:</b> <font color='green'>✓ PCB PASSED - No defects detected</font>"
    else:
        summary_text += f"<br/><b>Result:</b> <font color='red'>✗ PCB FAILED - {formatted_data['summary']['total_defects']} defect(s) require attention</font>"
    
    story.append(Paragraph(summary_text, body_style))
    
    story.append(PageBreak())
    
    # ===================== QUALITY ASSESSMENT =====================
    story.append(Paragraph("QUALITY ASSESSMENT", heading_style))
    
    assessment_data = [
        ['Metric', 'Value', 'Score', 'Status'],
        ['Defect Count', str(formatted_data['summary']['total_defects']), 
         '✓' if formatted_data['summary']['total_defects'] == 0 else '✗',
         'PASS' if formatted_data['summary']['total_defects'] == 0 else 'FAIL'],
        ['Detection Confidence', formatted_data['summary']['average_confidence'], 
         '✓', 'HIGH'],
        ['Overall Quality', f"{quality_score}/100", quality_grade, 
         'PASS' if quality_score >= 70 else 'FAIL'],
        ['Risk Level', risk_level, '⚠' if risk_level != 'LOW' else '✓',
         'ACCEPTABLE' if risk_level == 'LOW' else 'REVIEW REQUIRED']
    ]
    
    assessment_table = Table(assessment_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
    assessment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0fff0')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(assessment_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Quality explanation
    quality_explanation = f"""
    <b>Quality Score Calculation:</b><br/>
    The quality score of {quality_score}/100 is calculated based on multiple factors including 
    defect count, severity, confidence levels, and defect sizes. A score above 90 indicates 
    excellent quality, 80-90 is good, 70-80 is acceptable, and below 70 requires attention.<br/><br/>
    <b>Risk Assessment:</b> {risk_level} - Based on the number and severity of detected defects.
    """
    story.append(Paragraph(quality_explanation, body_style))
    
    story.append(PageBreak())
    
    # ===================== DETAILED INSPECTION RESULTS =====================
    story.append(Paragraph("DETAILED INSPECTION RESULTS", heading_style))
    
    # Defect Distribution
    if formatted_data['class_distribution']:
        story.append(Paragraph("Defect Distribution by Type", subheading_style))
        
        dist_data = [['Defect Type', 'Count', 'Percentage', 'Severity']]
        total = sum(formatted_data['class_distribution'].values())
        
        # Define severity levels for each defect type
        severity_map = {
            'short': 'CRITICAL',
            'open': 'CRITICAL',
            'mousebite': 'HIGH',
            'spur': 'MEDIUM',
            'copper': 'MEDIUM',
            'pinhole': 'LOW'
        }
        
        for defect_type, count in sorted(formatted_data['class_distribution'].items(), 
                                        key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            severity = severity_map.get(defect_type, 'MEDIUM')
            dist_data.append([defect_type.title(), str(count), f"{percentage:.1f}%", severity])
        
        dist_table = Table(dist_data, colWidths=[2.2*inch, 1.2*inch, 1.2*inch, 1.4*inch])
        dist_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(dist_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Statistical Summary
    if formatted_data['defect_details']:
        story.append(Paragraph("Statistical Summary", subheading_style))
        
        confidences = [d['confidence (%)'] for d in formatted_data['defect_details']]
        areas = [d['area (px²)'] for d in formatted_data['defect_details']]
        
        stats_data = [
            ['Metric', 'Minimum', 'Maximum', 'Average', 'Median'],
            ['Confidence (%)', f"{min(confidences):.1f}", f"{max(confidences):.1f}",
             f"{np.mean(confidences):.1f}", f"{np.median(confidences):.1f}"],
            ['Area (px²)', f"{min(areas):.0f}", f"{max(areas):.0f}",
             f"{np.mean(areas):.0f}", f"{np.median(areas):.0f}"]
        ]
        
        stats_table = Table(stats_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff7f0e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff5f0')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(stats_table)
    
    # ===================== VISUAL INSPECTION RESULTS =====================
    if result_images:
        story.append(PageBreak())
        story.append(Paragraph("VISUAL INSPECTION RESULTS", heading_style))
        
        def add_image_to_story(cv_image, caption, max_width=5*inch, max_height=3.5*inch):
            """Convert OpenCV image and add to story with caption"""
            try:
                if len(cv_image.shape) == 2:
                    pil_img = PILImage.fromarray(cv_image)
                elif len(cv_image.shape) == 3:
                    import cv2
                    if cv_image.shape[2] == 3:
                        pil_img = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                    else:
                        pil_img = PILImage.fromarray(cv_image)
                
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                img_width, img_height = pil_img.size
                aspect = img_height / float(img_width)
                
                if img_width > max_width / inch:
                    display_width = max_width
                    display_height = display_width * aspect
                else:
                    display_width = img_width
                    display_height = img_height
                
                if display_height > max_height:
                    display_height = max_height
                    display_width = display_height / aspect
                
                img = Image(img_buffer, width=display_width, height=display_height)
                
                story.append(Paragraph(caption, subheading_style))
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
                
            except Exception as e:
                print(f"Error adding image {caption}: {str(e)}")
        
        if 'test' in result_images:
            add_image_to_story(result_images['test'], "1. Test PCB Image (Original)")
        
        if 'template' in result_images:
            add_image_to_story(result_images['template'], "2. Template Image (Reference)")
        
        story.append(PageBreak())
        
        if 'aligned' in result_images:
            add_image_to_story(result_images['aligned'], "3. Aligned Test Image")
        
        if 'difference_map' in result_images:
            add_image_to_story(result_images['difference_map'], "4. Difference Map")
        
        story.append(PageBreak())
        
        if 'mask' in result_images:
            add_image_to_story(result_images['mask'], "5. Defect Mask (Binary)")
        
        if 'annotated' in result_images:
            add_image_to_story(result_images['annotated'], "6. Final Annotated Result")
    
    # ===================== COMPREHENSIVE STATISTICAL ANALYSIS =====================
    if chart_paths and formatted_data['defect_details']:
        story.append(PageBreak())
        story.append(Paragraph("COMPREHENSIVE STATISTICAL ANALYSIS", heading_style))
        
        # Distribution charts
        if 'distribution' in chart_paths and os.path.exists(chart_paths['distribution']):
            story.append(Paragraph("Defect Type Distribution", subheading_style))
            story.append(Image(chart_paths['distribution'], width=5*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
        
        if 'confidence' in chart_paths and os.path.exists(chart_paths['confidence']):
            story.append(Paragraph("Confidence Score Distribution", subheading_style))
            story.append(Image(chart_paths['confidence'], width=5*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
        
        story.append(PageBreak())
        
        # Box plot
        if 'boxplot' in chart_paths and os.path.exists(chart_paths['boxplot']):
            story.append(Paragraph("Confidence Comparison by Type (Box Plot)", subheading_style))
            story.append(Paragraph("Shows median, quartiles, and outliers for each defect type.", body_style))
            story.append(Image(chart_paths['boxplot'], width=5*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # Violin plot
        if 'violin' in chart_paths and os.path.exists(chart_paths['violin']):
            story.append(Paragraph("Size Distribution by Type (Violin Plot)", subheading_style))
            story.append(Paragraph("Displays the full distribution shape of defect sizes.", body_style))
            story.append(Image(chart_paths['violin'], width=5*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
        
        story.append(PageBreak())
        
        # Scatter plot
        if 'scatter' in chart_paths and os.path.exists(chart_paths['scatter']):
            story.append(Paragraph("Area vs Confidence Correlation", subheading_style))
            story.append(Paragraph("Analyzes the relationship between defect size and detection confidence.", body_style))
            story.append(Image(chart_paths['scatter'], width=5*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # Location map
        if 'location_map' in chart_paths and os.path.exists(chart_paths['location_map']):
            story.append(Paragraph("Spatial Distribution Map", subheading_style))
            story.append(Paragraph("Heatmap showing concentration and location of defects on PCB.", body_style))
            story.append(Image(chart_paths['location_map'], width=5*inch, height=3.5*inch))
            story.append(Spacer(1, 0.2*inch))
        
        story.append(PageBreak())
        
        # Correlation heatmap
        if 'heatmap' in chart_paths and os.path.exists(chart_paths['heatmap']):
            story.append(Paragraph("Feature Correlation Matrix", subheading_style))
            story.append(Paragraph("Shows correlations between position, confidence, and size.", body_style))
            story.append(Image(chart_paths['heatmap'], width=4.5*inch, height=4*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # Priority matrix
        if 'priority_matrix' in chart_paths and os.path.exists(chart_paths['priority_matrix']):
            story.append(Paragraph("Defect Priority Matrix", subheading_style))
            story.append(Paragraph("Prioritizes defects based on size and confidence for efficient remediation.", body_style))
            story.append(Image(chart_paths['priority_matrix'], width=5*inch, height=3.5*inch))
    
    # ===================== DETAILED DEFECT INFORMATION =====================
    if formatted_data['defect_details']:
        story.append(PageBreak())
        story.append(Paragraph("DETAILED DEFECT INFORMATION", heading_style))
        
        defect_table_data = [['ID', 'Type', 'Confidence', 'Location', 'Area (px²)', 'Priority']]
        
        # Calculate priorities
        areas = [d['area (px²)'] for d in formatted_data['defect_details']]
        max_area = max(areas) if areas else 1
        
        for defect in formatted_data['defect_details'][:50]:
            conf = defect['confidence (%)']
            area = defect['area (px²)']
            
            # Priority calculation
            if conf >= 90 and area > 50:
                priority = "HIGH"
            elif conf >= 85 or area > 30:
                priority = "MEDIUM"
            else:
                priority = "LOW"
            
            defect_table_data.append([
                str(defect['id']),
                defect['type'].title(),
                f"{conf:.1f}%",
                defect['location'],
                str(area),
                priority
            ])
        
        defect_table = Table(defect_table_data, colWidths=[0.4*inch, 1.2*inch, 0.9*inch, 1.3*inch, 0.9*inch, 0.8*inch])
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d62728')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff0f0')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        story.append(defect_table)
        
        if len(formatted_data['defect_details']) > 50:
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(f"<i>Note: Showing first 50 of {len(formatted_data['defect_details'])} defects. "
                                 "See complete data in CSV export.</i>", body_style))
    
    # ===================== DEFECT CLASSIFICATION GUIDE =====================
    story.append(PageBreak())
    story.append(Paragraph("DEFECT CLASSIFICATION REFERENCE", heading_style))
    
    defect_guide = [
        ['Defect Type', 'Description', 'Typical Causes', 'Severity'],
        ['Mousebite', 'Small perforations or incomplete separation', 'Incomplete routing, dull tools', 'HIGH'],
        ['Open Circuit', 'Broken or incomplete traces', 'Etching issues, mechanical damage', 'CRITICAL'],
        ['Short Circuit', 'Unintended connections between traces', 'Over-etching, copper residue', 'CRITICAL'],
        ['Spur', 'Unwanted copper extensions', 'Etching irregularities', 'MEDIUM'],
        ['Copper Issue', 'Missing or excess copper', 'Plating problems, contamination', 'MEDIUM'],
        ['Pin-hole', 'Microscopic holes in copper', 'Plating defects, contamination', 'LOW']
    ]
    
    guide_table = Table(defect_guide, colWidths=[1.1*inch, 2*inch, 2*inch, 0.9*inch])
    guide_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff7f0e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff8f0')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    story.append(guide_table)
    
    # ===================== RECOMMENDATIONS =====================
    story.append(PageBreak())
    story.append(Paragraph("RECOMMENDATIONS & NEXT STEPS", heading_style))
    
    if formatted_data['summary']['total_defects'] == 0:
        recommendations = """
        <b>Quality Status: PASSED ✓</b><br/><br/>
        The inspected PCB shows no detectable defects and meets quality standards.<br/><br/>
        <b>Recommended Actions:</b><br/>
        • Proceed with the next manufacturing stage<br/>
        • Continue regular quality monitoring<br/>
        • Archive this report for quality records<br/>
        • Maintain current manufacturing parameters<br/>
        • Consider this batch as reference for quality standards<br/>
        """
    else:
        # Priority-based recommendations
        high_priority = sum(1 for d in formatted_data['defect_details'] 
                          if d['confidence (%)'] >= 90 and d['area (px²)'] > 50)
        
        recommendations = f"""
        <b>Quality Status: REQUIRES ATTENTION ✗</b><br/><br/>
        The inspection detected {formatted_data['summary']['total_defects']} defect(s), 
        including {high_priority} high-priority defects that require immediate review.<br/><br/>
        <b>Immediate Actions Required (Priority Order):</b><br/>
        1. <b>HIGH PRIORITY:</b> Review and address {high_priority} critical defects first<br/>
        2. Assess defect severity and impact on functionality<br/>
        3. Determine if rework is feasible and cost-effective<br/>
        4. Document root cause analysis for each defect type<br/>
        5. Isolate affected units to prevent downstream issues<br/><br/>
        <b>Preventive Measures:</b><br/>
        • Review manufacturing process parameters<br/>
        • Check equipment calibration and maintenance schedules<br/>
        • Verify material quality and handling procedures<br/>
        • Implement additional in-process quality controls<br/>
        • Train personnel on identified defect patterns<br/>
        • Update work instructions based on findings<br/><br/>
        <b>Follow-up Actions:</b><br/>
        • Schedule re-inspection after corrective actions<br/>
        • Update quality management system records<br/>
        • Monitor next {min(10, formatted_data['summary']['total_defects'] * 2)} units closely<br/>
        • Conduct process capability study if defects persist<br/>
        • Implement preventive measures for future production<br/>
        """
    
    story.append(Paragraph(recommendations, body_style))
    
    # ===================== CONCLUSION =====================
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("CONCLUSION", heading_style))
    
    if formatted_data['summary']['total_defects'] == 0:
        conclusion_text = f"""
        This automated inspection has verified that the PCB meets all quality standards with no detectable 
        defects. The board has successfully passed the inspection process and is approved for use in the 
        next stage of manufacturing or deployment.<br/><br/>
        
        <b>Quality Score:</b> {quality_score}/100 ({quality_grade})<br/>
        <b>Risk Level:</b> <font color='{risk_color}'>{risk_level}</font><br/>
        <b>Final Verdict:</b> <font color='green'><b>APPROVED FOR PRODUCTION</b></font><br/><br/>
        
        The high confidence scores and successful image alignment confirm the reliability of this inspection 
        result. Regular monitoring should continue to maintain this quality standard.
        """
    else:
        conclusion_text = f"""
        This automated inspection has identified {formatted_data['summary']['total_defects']} defect(s) 
        on the PCB that require attention. The defects have been classified with an average confidence of 
        {formatted_data['summary']['average_confidence']}, indicating reliable detection.<br/><br/>
        
        <b>Quality Score:</b> {quality_score}/100 ({quality_grade})<br/>
        <b>Risk Level:</b> <font color='{risk_color}'>{risk_level}</font><br/>
        <b>Final Verdict:</b> <font color='red'><b>REQUIRES CORRECTIVE ACTION</b></font><br/><br/>
        
        Before proceeding to production, the quality assurance team should review the detected defects, 
        assess their impact on functionality, and determine the appropriate corrective measures. 
        Re-inspection is recommended after implementing corrections. The priority matrix and detailed 
        defect information should guide the remediation sequence.
        """
    
    story.append(Paragraph(conclusion_text, body_style))
    
    story.append(Spacer(1, 0.4*inch))
    
    # ===================== REPORT CERTIFICATION =====================
    footer_text = """
    <b>Report Certification</b><br/>
    This comprehensive report was automatically generated by CircuitGuard-PCB, an AI-powered PCB defect 
    detection system using EfficientNet-B4 deep learning architecture. All measurements, classifications, 
    and statistical analyses are performed with high precision computer vision algorithms.<br/><br/>
    <b>System Information:</b><br/>
    • Model: EfficientNet-B4 Deep Learning Architecture<br/>
    • Detection Method: Template Matching + Deep Learning Classification<br/>
    • Confidence Threshold: Optimized for maximum accuracy<br/>
    • Processing: Automated image alignment and defect localization<br/>
    • Analysis: Statistical, spatial, and priority-based assessment<br/><br/>
    <b>Report Includes:</b><br/>
    • Visual inspection results with 6 analysis stages<br/>
    • Comprehensive statistical analysis with 6+ visualization types<br/>
    • Quality scoring and risk assessment<br/>
    • Priority-based defect ranking<br/>
    • Actionable recommendations and remediation guidelines<br/><br/>
    <i>For questions or concerns about this report, please contact your quality assurance team.</i>
    """
    
    story.append(Paragraph(footer_text, body_style))
    
    # Build PDF
    doc.build(story, canvasmaker=BorderedCanvas)
    
    # Clean up temporary chart files
    for chart_path in extra_charts.values():
        try:
            if os.path.exists(chart_path):
                os.remove(chart_path)
        except:
            pass
    
    print(f"✅ Comprehensive PDF report generated: {output_path}")