from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import os
import numpy as np
import cv2
import zipfile
from backend import process_pair_and_predict, load_model
import csv
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import io

# Flask setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# Load model once at startup
print("Loading model...")
model = load_model()
print("Model loaded successfully ✅")

@app.route("/")
def index():
    return render_template("index.html")

def generate_pdf_report(data, output_path):
    """Generate comprehensive PDF report"""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#333333'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title Page
    story.append(Paragraph("PCB Defect Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Quality Status Box
    status_color = colors.green if data['quality_metrics']['pass_fail'] == 'PASS' else (
        colors.orange if data['quality_metrics']['pass_fail'] == 'REVIEW' else colors.red
    )
    
    status_data = [
        ['Quality Status', data['quality_metrics']['pass_fail']],
        ['Severity Score', f"{data['severity_score']}/100"]
    ]
    
    status_table = Table(status_data, colWidths=[3*inch, 2*inch])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#333333')),
        ('TEXTCOLOR', (1, 0), (1, 0), status_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(status_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(data['summary'], styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Conclusion
    story.append(Paragraph("Conclusion", heading_style))
    story.append(Paragraph(data['conclusion'], styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Key Metrics Table
    story.append(Paragraph("Key Metrics", heading_style))
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Defects Found', str(data['total_defects'])],
        ['Average Confidence', f"{data['avg_confidence']}%"],
        ['Area Affected', f"{data['affected_percentage']}%"],
        ['Defect Density', f"{data['defect_density']} per 1000px²"],
        ['Defect-Free Area', f"{data['quality_metrics']['defect_free_percentage']}%"],
        ['Confidence Level', data['quality_metrics']['confidence_level']],
        ['PCB Dimensions', f"{data['pcb_dimensions']['width']} × {data['pcb_dimensions']['height']} px"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Critical Defects Warning
    if data['critical_defects']:
        story.append(Paragraph(f"⚠️ CRITICAL DEFECTS DETECTED: {len(data['critical_defects'])} high-priority defects require immediate attention!", 
                             ParagraphStyle('Warning', parent=styles['Normal'], textColor=colors.red, fontSize=12, fontName='Helvetica-Bold')))
        story.append(Spacer(1, 0.2*inch))
    
    # Defect Distribution
    if data['defect_labels']:
        story.append(Paragraph("Defect Type Distribution", heading_style))
        defect_dist_data = [['Defect Type', 'Count']]
        for label, count in zip(data['defect_labels'], data['defect_counts']):
            defect_dist_data.append([label, str(count)])
        
        defect_table = Table(defect_dist_data, colWidths=[3*inch, 2*inch])
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(defect_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Page Break
    story.append(PageBreak())
    
    # Detailed Defect Breakdown
    if data['defect_details']:
        story.append(Paragraph("Detailed Defect Breakdown", heading_style))
        detail_data = [['ID', 'Type', 'Confidence', 'Area (px²)', 'Severity']]
        for defect in data['defect_details'][:20]:  # Limit to first 20
            detail_data.append([
                f"#{defect['id']}",
                defect['type'],
                f"{defect['confidence']}%",
                str(defect['area']),
                defect['severity']
            ])
        
        detail_table = Table(detail_data, colWidths=[0.6*inch, 1.8*inch, 1*inch, 1*inch, 1*inch])
        detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        story.append(detail_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Repair Recommendations
    if data['repair_recommendations']:
        story.append(PageBreak())
        story.append(Paragraph("Repair Recommendations", heading_style))
        repair_data = [['Defect Type', 'Recommended Action', 'Priority', 'Est. Time']]
        for rec in data['repair_recommendations']:
            repair_data.append([
                rec['defect'],
                rec['action'],
                rec['priority'],
                rec['estimated_time']
            ])
        
        repair_table = Table(repair_data, colWidths=[1.5*inch, 2.5*inch, 0.8*inch, 0.8*inch])
        repair_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(repair_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Add Images
    story.append(PageBreak())
    story.append(Paragraph("Visual Analysis", heading_style))
    
    # Annotated Image
    if os.path.exists(data['annotated_path']):
        story.append(Paragraph("Annotated Result", styles['Heading3']))
        img = RLImage(data['annotated_path'], width=5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    # Difference Map
    if os.path.exists(data['diff_path']):
        story.append(Paragraph("Difference Map", styles['Heading3']))
        img = RLImage(data['diff_path'], width=5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)

@app.route("/process", methods=["POST"])
def process_images():
    if "template" not in request.files or "test" not in request.files:
        return "Please upload both template and test images.", 400

    template_file = request.files["template"]
    test_file = request.files["test"]
    threshold = int(request.form.get("threshold", 10))
    min_area = int(request.form.get("min_area", 100))
    confidence_filter = float(request.form.get("confidence_filter", 0))

    if template_file.filename == "" or test_file.filename == "":
        return "Please upload both template and test images.", 400

    # Secure filenames
    template_name = secure_filename(template_file.filename)
    test_name = secure_filename(test_file.filename)
    template_path = os.path.join(app.config["UPLOAD_FOLDER"], template_name)
    test_path = os.path.join(app.config["UPLOAD_FOLDER"], test_name)
    template_file.save(template_path)
    test_file.save(test_path)

    # Convert to PIL
    pil_temp = Image.open(template_path).convert("RGB")
    pil_test = Image.open(test_path).convert("RGB")

    # Run backend processing
    annotated_pil, csv_bytes, crop_files, mask_np, diff_np = process_pair_and_predict(
        pil_test, pil_temp,
        model=model,
        threshold=threshold,
        min_area=min_area,
        crop_prefix="flask_"
    )

    # Save annotated image
    annotated_path = os.path.join(app.config["OUTPUT_FOLDER"], "annotated.jpg")
    annotated_pil.save(annotated_path)

    # Save CSV
    csv_path = os.path.join(app.config["OUTPUT_FOLDER"], "detections.csv")
    with open(csv_path, "wb") as f:
        f.write(csv_bytes.getvalue())

    # Save diff & mask
    diff_path = os.path.join(app.config["OUTPUT_FOLDER"], "diff.jpg")
    mask_path = os.path.join(app.config["OUTPUT_FOLDER"], "mask.jpg")
    cv2.imwrite(diff_path, diff_np)
    cv2.imwrite(mask_path, mask_np)

    # Process detected defects for visualization
    class_counts = {}
    scatter_points = []
    confidence_distribution = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    defect_details = []
    total_area_affected = 0
    
    for i, (name, pil_crop, pred_label, conf_pct) in enumerate(crop_files):
        if conf_pct < confidence_filter:
            continue
        class_counts[pred_label] = class_counts.get(pred_label, 0) + 1
        
        # Calculate defect area (approximate from crop size)
        defect_area = pil_crop.width * pil_crop.height
        total_area_affected += defect_area
        
        # Scatter position (placeholder)
        scatter_points.append({
            "x": np.random.randint(0, 100), 
            "y": np.random.randint(0, 100),
            "type": pred_label,
            "confidence": conf_pct
        })
        
        # Confidence distribution
        if conf_pct < 20:
            confidence_distribution["0-20"] += 1
        elif conf_pct < 40:
            confidence_distribution["20-40"] += 1
        elif conf_pct < 60:
            confidence_distribution["40-60"] += 1
        elif conf_pct < 80:
            confidence_distribution["60-80"] += 1
        else:
            confidence_distribution["80-100"] += 1
        
        # Detailed defect info
        defect_details.append({
            "id": i + 1,
            "type": pred_label,
            "confidence": round(conf_pct, 2),
            "area": defect_area,
            "severity": "Critical" if conf_pct > 80 else ("High" if conf_pct > 60 else "Medium")
        })

    # Compute damage severity
    total_defects = sum(class_counts.values())
    avg_conf = np.mean([c[3] for c in crop_files]) if crop_files else 0
    severity = min(100, int(total_defects * 10 + avg_conf * 0.5))
    
    # Calculate PCB dimensions and affected percentage
    pcb_total_area = pil_test.width * pil_test.height
    affected_percentage = min(100, (total_area_affected / pcb_total_area) * 100) if pcb_total_area > 0 else 0
    
    # Defect density (defects per 1000 pixels²)
    defect_density = (total_defects / pcb_total_area) * 1000 if pcb_total_area > 0 else 0

    # Auto summary and conclusion
    if severity < 30:
        summary = "Only minor solder or pad irregularities detected, PCB is in healthy state."
        conclusion = "PCB is safe for production use with minimal rework required."
    elif severity < 70:
        summary = "Moderate level of defects found such as solder bridges or missing vias."
        conclusion = "PCB should undergo manual inspection before approval."
    else:
        summary = "High density of major defects including track breaks and severe pad damage."
        conclusion = "PCB is not suitable for deployment, requires full rework or replacement."

    # Generate repair recommendations based on defect types
    repair_recommendations = []
    if "solder_bridge" in class_counts:
        repair_recommendations.append({
            "defect": "Solder Bridge",
            "action": "Use desoldering wick or solder sucker to remove excess solder",
            "priority": "High",
            "estimated_time": f"{class_counts['solder_bridge'] * 2} minutes"
        })
    if "missing_hole" in class_counts or "mouse_bite" in class_counts:
        repair_recommendations.append({
            "defect": "Missing Hole / Mouse Bite",
            "action": "Drill new holes or replace PCB section",
            "priority": "Critical",
            "estimated_time": f"{class_counts.get('missing_hole', 0) * 5} minutes"
        })
    if "open_circuit" in class_counts or "short" in class_counts:
        repair_recommendations.append({
            "defect": "Circuit Issue",
            "action": "Trace and repair using conductive ink or jumper wire",
            "priority": "Critical",
            "estimated_time": f"{class_counts.get('open_circuit', 0) * 10} minutes"
        })
    if "spurious_copper" in class_counts:
        repair_recommendations.append({
            "defect": "Spurious Copper",
            "action": "Carefully remove excess copper with precision knife",
            "priority": "Medium",
            "estimated_time": f"{class_counts['spurious_copper'] * 3} minutes"
        })
    
    # Quality metrics
    quality_metrics = {
        "pass_fail": "PASS",  # Always PASS by default
        "defect_free_percentage": max(0, 100 - affected_percentage),
        "confidence_level": "High" if avg_conf > 70 else ("Medium" if avg_conf > 40 else "Low")
    }
    
    # Critical defects (high confidence + serious types)
    critical_defects = [d for d in defect_details if d["confidence"] > 80 and 
                    d["type"] in ["open_circuit", "short", "missing_hole"]]
    
    # Generate comprehensive log CSV
    log_csv_path = os.path.join(app.config["OUTPUT_FOLDER"], "analysis_log.csv")
    with open(log_csv_path, 'w', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        
        # Header
        writer.writerow(['PCB Defect Analysis Log'])
        writer.writerow(['Timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])
        
        # Summary Information
        writer.writerow(['=== SUMMARY ==='])
        writer.writerow(['Quality Status', quality_metrics['pass_fail']])
        writer.writerow(['Severity Score', f"{severity}/100"])
        writer.writerow(['Total Defects Found', total_defects])
        writer.writerow(['Average Confidence', f"{round(avg_conf, 2)}%"])
        writer.writerow(['Area Affected', f"{round(affected_percentage, 2)}%"])
        writer.writerow(['Defect Density', f"{round(defect_density, 4)} per 1000px²"])
        writer.writerow(['Defect-Free Area', f"{quality_metrics['defect_free_percentage']}%"])
        writer.writerow(['Confidence Level', quality_metrics['confidence_level']])
        writer.writerow(['PCB Dimensions', f"{pil_test.width} × {pil_test.height} px"])
        writer.writerow([])
        
        # Executive Summary
        writer.writerow(['=== EXECUTIVE SUMMARY ==='])
        writer.writerow([summary])
        writer.writerow([])
        
        # Conclusion
        writer.writerow(['=== CONCLUSION ==='])
        writer.writerow([conclusion])
        writer.writerow([])
        
        # Defect Type Distribution
        if class_counts:
            writer.writerow(['=== DEFECT TYPE DISTRIBUTION ==='])
            writer.writerow(['Defect Type', 'Count'])
            for label, count in class_counts.items():
                writer.writerow([label, count])
            writer.writerow([])
        
        # Confidence Distribution
        writer.writerow(['=== CONFIDENCE DISTRIBUTION ==='])
        writer.writerow(['Range', 'Count'])
        for range_label, count in confidence_distribution.items():
            writer.writerow([f"{range_label}%", count])
        writer.writerow([])
        
        # Critical Defects
        if critical_defects:
            writer.writerow(['=== CRITICAL DEFECTS ==='])
            writer.writerow([f'{len(critical_defects)} high-priority defects require immediate attention!'])
            writer.writerow(['ID', 'Type', 'Confidence', 'Area (px²)', 'Severity'])
            for defect in critical_defects:
                writer.writerow([
                    f"#{defect['id']}", 
                    defect['type'], 
                    f"{defect['confidence']}%", 
                    defect['area'], 
                    defect['severity']
                ])
            writer.writerow([])
        
        # Detailed Defect Breakdown
        if defect_details:
            writer.writerow(['=== DETAILED DEFECT BREAKDOWN ==='])
            writer.writerow(['ID', 'Type', 'Confidence (%)', 'Area (px²)', 'Severity'])
            for defect in defect_details:
                writer.writerow([
                    f"#{defect['id']}", 
                    defect['type'], 
                    defect['confidence'], 
                    defect['area'], 
                    defect['severity']
                ])
            writer.writerow([])
        
        # Repair Recommendations
        if repair_recommendations:
            writer.writerow(['=== REPAIR RECOMMENDATIONS ==='])
            writer.writerow(['Defect Type', 'Recommended Action', 'Priority', 'Estimated Time'])
            for rec in repair_recommendations:
                writer.writerow([
                    rec['defect'], 
                    rec['action'], 
                    rec['priority'], 
                    rec['estimated_time']
                ])
            writer.writerow([])
        
        # Processing Parameters
        writer.writerow(['=== PROCESSING PARAMETERS ==='])
        writer.writerow(['Threshold', threshold])
        writer.writerow(['Minimum Area', min_area])
        writer.writerow(['Confidence Filter', confidence_filter])

    # Create a ZIP file for report (images + CSV + log)
    zip_path = os.path.join(app.config["OUTPUT_FOLDER"], "full_report.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(annotated_path, "annotated.jpg")
        zipf.write(csv_path, "detections.csv")
        zipf.write(diff_path, "diff.jpg")
        zipf.write(mask_path, "mask.jpg")
        zipf.write(log_csv_path, "analysis_log.csv")
    
    # Generate PDF Report
    pdf_path = os.path.join(app.config["OUTPUT_FOLDER"], "analysis_report.pdf")
    pdf_data = {
        'severity_score': severity,
        'summary': summary,
        'conclusion': conclusion,
        'defect_labels': list(class_counts.keys()),
        'defect_counts': list(class_counts.values()),
        'defect_details': defect_details,
        'confidence_distribution': confidence_distribution,
        'total_defects': total_defects,
        'avg_confidence': round(avg_conf, 2),
        'affected_percentage': round(affected_percentage, 2),
        'defect_density': round(defect_density, 4),
        'repair_recommendations': repair_recommendations,
        'quality_metrics': quality_metrics,
        'critical_defects': critical_defects,
        'pcb_dimensions': {"width": pil_test.width, "height": pil_test.height},
        'annotated_path': annotated_path,
        'diff_path': diff_path,
        'mask_path': mask_path
    }
    
    generate_pdf_report(pdf_data, pdf_path)

    return render_template(
        "results.html",
        annotated_img="/outputs/annotated.jpg",
        diff_img="/outputs/diff.jpg",
        mask_img="/outputs/mask.jpg",
        report_zip="/outputs/full_report.zip",
        report_pdf="/outputs/analysis_report.pdf",
        log_csv="/outputs/analysis_log.csv",
        severity_score=severity,
        summary=summary,
        conclusion=conclusion,
        defect_labels=list(class_counts.keys()),
        defect_counts=list(class_counts.values()),
        scatter_points=scatter_points,
        # New additions
        defect_details=defect_details,
        confidence_distribution=confidence_distribution,
        total_defects=total_defects,
        avg_confidence=round(avg_conf, 2),
        affected_percentage=round(affected_percentage, 2),
        defect_density=round(defect_density, 4),
        repair_recommendations=repair_recommendations,
        quality_metrics=quality_metrics,
        critical_defects=critical_defects,
        pcb_dimensions={"width": pil_test.width, "height": pil_test.height}
    )

@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
