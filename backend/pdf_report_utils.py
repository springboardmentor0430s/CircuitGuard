from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.widgets.markers import makeMarker
from io import BytesIO
import tempfile
import numpy as np
import cv2
from datetime import datetime
import os


def create_header_box(width=7*inch, height=0.2*inch):
    """Create a decorative header box"""
    drawing = Drawing(width, height)
    
    # Gradient-like effect with rectangles
    colors_gradient = [
        colors.HexColor('#1a73e8'),
        colors.HexColor('#4285f4'),
        colors.HexColor('#669df6')
    ]
    
    segment_width = width / len(colors_gradient)
    for i, color in enumerate(colors_gradient):
        rect = Rect(i * segment_width, 0, segment_width, height, 
                   fillColor=color, strokeColor=None)
        drawing.add(rect)
    
    return drawing


def create_defect_type_bar_chart(frequency_analysis, width=450, height=280):
    """Create an enhanced bar chart for defect type distribution"""
    drawing = Drawing(width, height)
    
    if not frequency_analysis:
        return drawing
    
    chart = VerticalBarChart()
    chart.x = 60
    chart.y = 60
    chart.width = width - 120
    chart.height = height - 120
    
    # Prepare data
    defect_types = list(frequency_analysis.keys())
    counts = [frequency_analysis[dt]['count'] for dt in defect_types]
    
    chart.data = [counts]
    chart.categoryAxis.categoryNames = [dt.replace('_', ' ').title() for dt in defect_types]
    
    # Enhanced styling with gradient colors
    color_palette = [
        colors.HexColor('#667eea'),
        colors.HexColor('#764ba2'),
        colors.HexColor('#f093fb'),
        colors.HexColor('#4facfe'),
        colors.HexColor('#00f2fe'),
        colors.HexColor('#43e97b')
    ]
    
    for i in range(len(counts)):
        chart.bars[i].fillColor = color_palette[i % len(color_palette)]
    
    chart.bars.strokeColor = colors.white
    chart.bars.strokeWidth = 1
    
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(counts) * 1.2 if counts else 10
    chart.valueAxis.valueStep = max(1, max(counts) // 5) if counts else 1
    
    chart.categoryAxis.labels.boxAnchor = 'ne'
    chart.categoryAxis.labels.dx = -5
    chart.categoryAxis.labels.dy = -5
    chart.categoryAxis.labels.angle = 30
    chart.categoryAxis.labels.fontSize = 9
    chart.categoryAxis.labels.fontName = 'Helvetica-Bold'
    
    chart.valueAxis.labels.fontSize = 10
    chart.valueAxis.labels.fontName = 'Helvetica'
    chart.valueAxis.strokeWidth = 2
    chart.valueAxis.strokeColor = colors.HexColor('#cccccc')
    
    # Add title
    title = String(width/2, height - 25, 'Defect Type Distribution',
                  fontSize=12, fontName='Helvetica-Bold',
                  textAnchor='middle', fillColor=colors.HexColor('#1a73e8'))
    drawing.add(title)
    
    drawing.add(chart)
    return drawing


def create_confidence_distribution_chart(confidence_stats, width=450, height=280):
    """Create an enhanced bar chart for confidence level distribution"""
    drawing = Drawing(width, height)
    
    if not confidence_stats:
        return drawing
    
    chart = VerticalBarChart()
    chart.x = 60
    chart.y = 60
    chart.width = width - 120
    chart.height = height - 120
    
    # Data: High, Medium, Low confidence counts
    data = [
        confidence_stats.get('high_confidence_count', 0),
        confidence_stats.get('medium_confidence_count', 0),
        confidence_stats.get('low_confidence_count', 0)
    ]
    
    chart.data = [data]
    chart.categoryAxis.categoryNames = ['High\n(≥80%)', 'Medium\n(50-80%)', 'Low\n(<50%)']
    
    # Color coding with enhanced visuals
    chart.bars[0].fillColor = colors.HexColor('#10b981')
    chart.bars[1].fillColor = colors.HexColor('#f59e0b')
    chart.bars[2].fillColor = colors.HexColor('#ef4444')
    
    chart.bars.strokeColor = colors.white
    chart.bars.strokeWidth = 2
    
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(data) * 1.2 if max(data) > 0 else 10
    chart.valueAxis.valueStep = max(1, max(data) // 5) if max(data) > 0 else 1
    
    chart.categoryAxis.labels.fontSize = 9
    chart.categoryAxis.labels.fontName = 'Helvetica-Bold'
    chart.categoryAxis.labels.boxAnchor = 'n'
    
    chart.valueAxis.labels.fontSize = 10
    chart.valueAxis.strokeWidth = 2
    chart.valueAxis.strokeColor = colors.HexColor('#cccccc')
    
    # Add title
    title = String(width/2, height - 25, 'Confidence Level Distribution',
                  fontSize=12, fontName='Helvetica-Bold',
                  textAnchor='middle', fillColor=colors.HexColor('#1a73e8'))
    drawing.add(title)
    
    drawing.add(chart)
    return drawing


def create_defect_size_distribution_chart(defects, width=450, height=280):
    """Create an enhanced bar chart showing defect size categories"""
    drawing = Drawing(width, height)
    
    if not defects:
        return drawing
    
    # Categorize defects by size (area in pixels)
    size_categories = {
        'Small\n(<100px²)': 0,
        'Medium\n(100-500px²)': 0,
        'Large\n(500-1000px²)': 0,
        'Very Large\n(>1000px²)': 0
    }
    
    for defect in defects:
        area = defect.get('area', 0)
        if area < 100:
            size_categories['Small\n(<100px²)'] += 1
        elif area < 500:
            size_categories['Medium\n(100-500px²)'] += 1
        elif area < 1000:
            size_categories['Large\n(500-1000px²)'] += 1
        else:
            size_categories['Very Large\n(>1000px²)'] += 1
    
    chart = VerticalBarChart()
    chart.x = 60
    chart.y = 60
    chart.width = width - 120
    chart.height = height - 120
    
    counts = list(size_categories.values())
    chart.data = [counts]
    chart.categoryAxis.categoryNames = list(size_categories.keys())
    
    # Gradient colors from small to large
    colors_gradient = [
        colors.HexColor('#a78bfa'),
        colors.HexColor('#8b5cf6'),
        colors.HexColor('#7c3aed'),
        colors.HexColor('#6d28d9')
    ]
    
    for i in range(len(counts)):
        chart.bars[i].fillColor = colors_gradient[i]
    
    chart.bars.strokeColor = colors.white
    chart.bars.strokeWidth = 2
    
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(counts) * 1.2 if max(counts) > 0 else 10
    chart.valueAxis.valueStep = max(1, max(counts) // 5) if max(counts) > 0 else 1
    
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.fontName = 'Helvetica-Bold'
    chart.categoryAxis.labels.boxAnchor = 'n'
    
    chart.valueAxis.labels.fontSize = 10
    chart.valueAxis.strokeWidth = 2
    chart.valueAxis.strokeColor = colors.HexColor('#cccccc')
    
    # Add title
    title = String(width/2, height - 25, 'Defect Size Distribution',
                  fontSize=12, fontName='Helvetica-Bold',
                  textAnchor='middle', fillColor=colors.HexColor('#1a73e8'))
    drawing.add(title)
    
    drawing.add(chart)
    return drawing


def create_defect_scatter_plot(defects, image_width, image_height, width=450, height=350):
    """Create a scatter plot showing defect locations with size representation"""
    drawing = Drawing(width, height)
    
    if not defects or image_width <= 0 or image_height <= 0:
        return drawing
    
    # Plot dimensions
    plot_x = 60
    plot_y = 60
    plot_width = width - 120
    plot_height = height - 120
    
    # Background
    bg_rect = Rect(plot_x, plot_y, plot_width, plot_height,
                   fillColor=colors.HexColor('#f8f9fa'),
                   strokeColor=colors.HexColor('#dee2e6'),
                   strokeWidth=2)
    drawing.add(bg_rect)
    
    # Grid lines
    grid_color = colors.HexColor('#e9ecef')
    for i in range(5):
        # Horizontal lines
        y_pos = plot_y + (plot_height / 4) * i
        drawing.add(Rect(plot_x, y_pos, plot_width, 0.5,
                        fillColor=grid_color, strokeColor=None))
        # Vertical lines
        x_pos = plot_x + (plot_width / 4) * i
        drawing.add(Rect(x_pos, plot_y, 0.5, plot_height,
                        fillColor=grid_color, strokeColor=None))
    
    # Color mapping for defect types
    defect_colors = {
        'missing_hole': colors.HexColor('#ef4444'),
        'mouse_bite': colors.HexColor('#f59e0b'),
        'open_circuit': colors.HexColor('#10b981'),
        'short': colors.HexColor('#3b82f6'),
        'spur': colors.HexColor('#8b5cf6'),
        'spurious_copper': colors.HexColor('#ec4899')
    }
    
    # Plot defects
    for defect in defects:
        center = defect.get('center', {})
        x_norm = center.get('x', 0) / image_width
        y_norm = 1 - (center.get('y', 0) / image_height)  # Invert Y axis
        
        x_plot = plot_x + x_norm * plot_width
        y_plot = plot_y + y_norm * plot_height
        
        # Size based on area (scaled)
        area = defect.get('area', 100)
        marker_size = min(12, max(4, np.sqrt(area) / 5))
        
        # Color based on defect type
        defect_type = defect.get('class_name', 'unknown')
        marker_color = defect_colors.get(defect_type, colors.HexColor('#6b7280'))
        
        # Draw marker (circle)
        from reportlab.graphics.shapes import Circle
        marker = Circle(x_plot, y_plot, marker_size,
                       fillColor=marker_color,
                       strokeColor=colors.white,
                       strokeWidth=1)
        drawing.add(marker)
    
    # Axes labels
    x_label = String(plot_x + plot_width/2, plot_y - 35,
                    'X Position (pixels)',
                    fontSize=10, fontName='Helvetica-Bold',
                    textAnchor='middle', fillColor=colors.HexColor('#374151'))
    drawing.add(x_label)
    
    y_label = String(plot_x - 45, plot_y + plot_height/2,
                    'Y Position',
                    fontSize=10, fontName='Helvetica-Bold',
                    textAnchor='middle', fillColor=colors.HexColor('#374151'))
    
    # Title
    title = String(width/2, height - 25,
                  'Defect Spatial Distribution',
                  fontSize=12, fontName='Helvetica-Bold',
                  textAnchor='middle', fillColor=colors.HexColor('#1a73e8'))
    drawing.add(title)
    
    # Add legend
    legend_y = height - 60
    legend_x = plot_x + plot_width - 55
    
    legend_title = String(legend_x, legend_y + 15, 'Defect Types',
                         fontSize=8, fontName='Helvetica-Bold',
                         textAnchor='start', fillColor=colors.HexColor('#374151'))
    drawing.add(legend_title)
    
    unique_types = list(set(d.get('class_name', '') for d in defects))[:6]
    for i, defect_type in enumerate(unique_types):
        y_pos = legend_y - i * 12
        marker_color = defect_colors.get(defect_type, colors.HexColor('#6b7280'))
        
        # Legend marker
        from reportlab.graphics.shapes import Circle
        legend_marker = Circle(legend_x, y_pos, 3,
                              fillColor=marker_color,
                              strokeColor=colors.white,
                              strokeWidth=0.5)
        drawing.add(legend_marker)
        
        # Legend text
        legend_text = String(legend_x + 8, y_pos - 2,
                           defect_type.replace('_', ' ').title(),
                           fontSize=7, fontName='Helvetica',
                           textAnchor='start', fillColor=colors.HexColor('#374151'))
        drawing.add(legend_text)
    
    drawing.add(y_label)
    return drawing


def create_defect_type_pie_chart(frequency_analysis, width=450, height=280):
    """Create an enhanced pie chart for defect type distribution"""
    drawing = Drawing(width, height)
    
    if not frequency_analysis:
        return drawing
    
    pie = Pie()
    pie.x = width/2 - 80
    pie.y = height/2 - 80
    pie.width = 160
    pie.height = 160
    
    # Prepare data
    defect_types = list(frequency_analysis.keys())
    counts = [frequency_analysis[dt]['count'] for dt in defect_types]
    
    pie.data = counts
    pie.labels = [f"{dt.replace('_', ' ').title()}\n({frequency_analysis[dt]['percentage']}%)" 
                  for dt in defect_types]
    
    # Enhanced color scheme
    colors_list = [
        colors.HexColor('#667eea'),
        colors.HexColor('#764ba2'),
        colors.HexColor('#f093fb'),
        colors.HexColor('#4facfe'),
        colors.HexColor('#00f2fe'),
        colors.HexColor('#43e97b')
    ]
    
    for i in range(len(counts)):
        pie.slices[i].fillColor = colors_list[i % len(colors_list)]
        pie.slices[i].strokeWidth = 2
        pie.slices[i].strokeColor = colors.white
        pie.slices[i].popout = 5 if i == 0 else 0  # Pop out largest slice
    
    pie.slices.fontSize = 8
    pie.slices.fontName = 'Helvetica-Bold'
    pie.slices.labelRadius = 1.2
    
    # Add title
    title = String(width/2, height - 25, 'Defect Type Distribution',
                  fontSize=12, fontName='Helvetica-Bold',
                  textAnchor='middle', fillColor=colors.HexColor('#1a73e8'))
    drawing.add(title)
    
    drawing.add(pie)
    return drawing


def generate_pdf_report(results, defects, defect_count, frequency_analysis, confidence_stats, processing_time):
    """Generate enhanced PDF report with images and detailed analysis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.4*inch, bottomMargin=0.4*inch,
                           leftMargin=0.6*inch, rightMargin=0.6*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1a73e8'),
        spaceAfter=10,
        spaceBefore=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#6b7280'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1a73e8'),
        spaceAfter=12,
        spaceBefore=16,
        fontName='Helvetica-Bold',
        borderWidth=0,
        borderColor=colors.HexColor('#1a73e8'),
        borderPadding=5,
        backColor=colors.HexColor('#f0f7ff')
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#374151'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#374151'),
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # Cover Page
    story.append(Spacer(1, 1.5*inch))
    story.append(create_header_box())
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("CircuitGuard", title_style))
    story.append(Paragraph("AI based PCB Defect Detection & Analysis Report (99.73% accuracy)", subtitle_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Status Box
    status_color = colors.HexColor('#10b981') if defect_count == 0 else colors.HexColor('#ef4444')
    quality_status = 'PASS - No Defects Detected' if defect_count == 0 else f'FAIL - {defect_count} Defect(s) Detected'
    
    status_data = [[Paragraph(f"<b>Quality Status: {quality_status}</b>", 
                             ParagraphStyle('status', parent=body_style, 
                                          textColor=colors.white, fontSize=14, alignment=TA_CENTER))]]
    status_table = Table(status_data, colWidths=[6*inch])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), status_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('ROUNDEDCORNERS', [10, 10, 10, 10]),
    ]))
    story.append(status_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    summary_text = f"""
    This report presents the results of automated PCB defect detection performed on 
    {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}. The analysis was completed in 
    {processing_time:.2f} seconds using advanced computer vision and deep learning techniques.
    """
    if defect_count > 0:
        summary_text += f""" A total of {defect_count} defect(s) were identified across 
        {len(frequency_analysis)} different categories. The average confidence level of 
        detections is {confidence_stats.get('average', 0)*100:.1f}%."""
    else:
        summary_text += " The PCB passed inspection with no defects detected."
    
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Key Metrics Summary
    story.append(Paragraph("Key Metrics", section_style))
    metrics_data = [
        ['Metric', 'Value', 'Status'],
        ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '✓'],
        ['Processing Time', f'{processing_time:.2f} seconds', '✓'],
        ['Total Defects Found', str(defect_count), 
         '✓' if defect_count == 0 else '⚠'],
        ['Quality Assessment', quality_status, 
         '✓' if defect_count == 0 else '✗']
    ]
    
    if defect_count > 0:
        metrics_data.extend([
            ['Average Confidence', f"{confidence_stats.get('average', 0)*100:.1f}%", 
             '✓' if confidence_stats.get('average', 0) > 0.7 else '⚠'],
            ['High Confidence Detections', str(confidence_stats.get('high_confidence_count', 0)), '✓']
        ])
    
    metrics_table = Table(metrics_data, colWidths=[2.2*inch, 2.8*inch, 0.8*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Annotated Result Image
    story.append(PageBreak())
    story.append(Paragraph("Visual Inspection Results", heading_style))
    story.append(Paragraph("Annotated PCB with Detected Defects", section_style))
    
    temp_images = []
    try:
        result_img_path = save_cv2_image_temp(results['result'])
        temp_images.append(result_img_path)
        story.append(Image(result_img_path, width=6*inch, height=5.7*inch))
        story.append(Spacer(1, 0.1*inch))
        
        caption_style = ParagraphStyle('Caption', parent=body_style, 
                                      fontSize=9, textColor=colors.HexColor('#6b7280'),
                                      alignment=TA_CENTER, italic=True)
        story.append(Paragraph("Figure 1: Annotated PCB showing detected defects with bounding boxes and classifications", 
                             caption_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Statistical Analysis (only if defects exist)
        if defect_count > 0:
            story.append(PageBreak())
            story.append(Paragraph("Statistical Analysis", heading_style))
            
            analysis_text = f"""
            The following charts provide a comprehensive statistical breakdown of the detected defects. 
            Analysis includes defect type distribution, confidence levels, size categorization, and 
            spatial distribution patterns. These visualizations help identify trends and potential 
            manufacturing issues.
            """
            story.append(Paragraph(analysis_text, body_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Defect Type Distribution - Bar and Pie Charts side by side
            story.append(Paragraph("Defect Type Analysis", section_style))
            bar_chart = create_defect_type_bar_chart(frequency_analysis)
            story.append(bar_chart)
            story.append(Spacer(1, 0.3*inch))
            
            pie_chart = create_defect_type_pie_chart(frequency_analysis)
            story.append(pie_chart)
            story.append(Spacer(1, 0.3*inch))
            
            # Confidence and Size Distribution
            story.append(PageBreak())
            story.append(Paragraph("Detection Quality Metrics", section_style))
            
            conf_chart = create_confidence_distribution_chart(confidence_stats)
            story.append(conf_chart)
            story.append(Spacer(1, 0.3*inch))
            
            size_chart = create_defect_size_distribution_chart(defects)
            story.append(size_chart)
            story.append(Spacer(1, 0.3*inch))
            
            # Spatial Distribution - Scatter Plot
            story.append(PageBreak())
            story.append(Paragraph("Spatial Distribution Analysis", section_style))
            
            spatial_text = """
            The scatter plot below shows the spatial distribution of defects across the PCB surface. 
            Each point represents a detected defect, with the marker size indicating the defect area 
            and color representing the defect type. This visualization helps identify clustering 
            patterns and potential systematic issues in specific board regions.
            """
            story.append(Paragraph(spatial_text, body_style))
            story.append(Spacer(1, 0.2*inch))
            
            scatter_plot = create_defect_scatter_plot(
                defects, 
                results['test'].shape[1], 
                results['test'].shape[0]
            )
            story.append(scatter_plot)
            story.append(Spacer(1, 0.3*inch))
            
        
        # Detailed Tables Section
        if frequency_analysis and defect_count > 0:
            story.append(PageBreak())
            story.append(Paragraph("Detailed Analysis Tables", heading_style))
            
            # Frequency Analysis Table
            story.append(Paragraph("Defect Type Distribution", section_style))
            freq_data = [['Defect Type', 'Count', 'Percentage', 'Severity']]
            
            severity_map = {
                'open_circuit': 'Critical',
                'short': 'Critical',
                'missing_hole': 'High',
                'mouse_bite': 'Medium',
                'spur': 'Medium',
                'spurious_copper': 'Low'
            }
            
            for defect_type, data in frequency_analysis.items():
                severity = severity_map.get(defect_type, 'Unknown')
                freq_data.append([
                    defect_type.replace('_', ' ').title(),
                    str(data['count']),
                    f"{data['percentage']}%",
                    severity
                ])
            
            freq_table = Table(freq_data, colWidths=[2.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            freq_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(freq_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Confidence Statistics
        if confidence_stats and defect_count > 0:
            story.append(Paragraph("Confidence Statistics", section_style))
            
            conf_text = """
            Detection confidence levels indicate the model's certainty in its predictions. 
            Higher confidence generally correlates with more accurate detections. The table 
            below provides a comprehensive breakdown of confidence metrics.
            """
            story.append(Paragraph(conf_text, body_style))
            story.append(Spacer(1, 0.1*inch))
            
            conf_data = [
                ['Metric', 'Value', 'Interpretation'],
                ['Average Confidence', f"{confidence_stats['average']*100:.1f}%",
                 'Good' if confidence_stats['average'] > 0.7 else 'Fair'],
                ['Minimum Confidence', f"{confidence_stats['min']*100:.1f}%",
                 'Review Required' if confidence_stats['min'] < 0.5 else 'Acceptable'],
                ['Maximum Confidence', f"{confidence_stats['max']*100:.1f}%", 'Excellent'],
                ['High Confidence (≥80%)', str(confidence_stats['high_confidence_count']),
                 f"{(confidence_stats['high_confidence_count']/defect_count)*100:.0f}% of total"],
                ['Medium Confidence (50-80%)', str(confidence_stats['medium_confidence_count']),
                 f"{(confidence_stats['medium_confidence_count']/defect_count)*100:.0f}% of total"],
                ['Low Confidence (<50%)', str(confidence_stats['low_confidence_count']),
                 'Needs Verification' if confidence_stats['low_confidence_count'] > 0 else 'None']
            ]
            
            conf_table = Table(conf_data, colWidths=[2.2*inch, 1.5*inch, 2.1*inch])
            conf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (1, -1), 'LEFT'),
                ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ('ALIGN', (2, 1), (2, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(conf_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Detailed Defects List
        if defects:
            story.append(PageBreak())
            story.append(Paragraph("Detailed Defect Inventory", heading_style))
            
            inventory_text = """
            The following table provides a complete inventory of all detected defects with 
            detailed specifications including location coordinates, dimensions, and confidence scores. 
            This information can be used for precise defect localization and remediation.
            """
            story.append(Paragraph(inventory_text, body_style))
            story.append(Spacer(1, 0.2*inch))
            
            defect_data = [['ID', 'Type', 'Confidence', 'Position\n(x,y)', 'Size\n(w×h)', 'Area\n(px²)']]
            
            for defect in defects:
                # Color code confidence
                conf_value = defect['confidence']*100
                defect_data.append([
                    str(defect['id']),
                    defect['class_name'].replace('_', ' ').title(),
                    f"{conf_value:.1f}%",
                    f"({defect['bbox']['x']}, {defect['bbox']['y']})",
                    f"{defect['bbox']['width']}×{defect['bbox']['height']}",
                    str(defect['area'])
                ])
            
            defect_table = Table(defect_data, colWidths=[0.5*inch, 1.6*inch, 1*inch, 1.1*inch, 0.9*inch, 0.7*inch])
            defect_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(defect_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Add recommendations based on defects
            story.append(Paragraph("Recommendations", section_style))
            
            recommendations = []
            
            if confidence_stats.get('low_confidence_count', 0) > 0:
                recommendations.append("• Manual verification recommended for low-confidence detections")
            
            if confidence_stats.get('average', 0) < 0.7:
                recommendations.append("• Consider re-inspection due to below-average confidence scores")
            
            critical_defects = sum(1 for d in defects if d.get('class_name') in ['open_circuit', 'short'])
            if critical_defects > 0:
                recommendations.append(f"• {critical_defects} critical defect(s) found - immediate attention required")
            
            high_area_defects = sum(1 for d in defects if d.get('area', 0) > 1000)
            if high_area_defects > 0:
                recommendations.append(f"• {high_area_defects} large defect(s) detected - may indicate systemic issues")
            
            if len(set(d.get('class_name') for d in defects)) > 3:
                recommendations.append("• Multiple defect types detected - review manufacturing process")
            
            if not recommendations:
                recommendations.append("• All detections appear standard - proceed with normal quality protocols")
            
            for rec in recommendations:
                story.append(Paragraph(rec, body_style))
            
            story.append(Spacer(1, 0.3*inch))
        
        # Reference Images Section
        story.append(PageBreak())
        story.append(Paragraph("Reference Images", heading_style))
        
        ref_text = """
        The following images show the original template (defect-free reference), the test PCB 
        under inspection, and the generated defect mask used for detection analysis.
        """
        story.append(Paragraph(ref_text, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Template Image
        story.append(Paragraph("Template Image (Defect-free Reference)", section_style))
        template_img_path = save_cv2_image_temp(results['template'])
        temp_images.append(template_img_path)
        story.append(Image(template_img_path, width=4.5*inch, height=3.4*inch))
        story.append(Paragraph("Figure 2: Template PCB serving as defect-free reference standard", 
                             caption_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Test Image
        story.append(Paragraph("Test Image (Inspected PCB)", section_style))
        test_img_path = save_cv2_image_temp(results['test'])
        temp_images.append(test_img_path)
        story.append(Image(test_img_path, width=4.5*inch, height=3.4*inch))
        story.append(Paragraph("Figure 3: Test PCB undergoing automated inspection", 
                             caption_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Defect Mask
        story.append(PageBreak())
        story.append(Paragraph("Defect Detection Mask", section_style))
        mask_text = """
        The defect mask is generated through image subtraction and thresholding techniques. 
        White regions indicate potential defect areas where the test image differs significantly 
        from the template reference.
        """
        story.append(Paragraph(mask_text, body_style))
        story.append(Spacer(1, 0.1*inch))
        
        mask_img_path = save_cv2_image_temp(results['defect_mask'])
        temp_images.append(mask_img_path)
        story.append(Image(mask_img_path, width=4.5*inch, height=3.4*inch))
        story.append(Paragraph("Figure 4: Binary mask highlighting detected defect regions", 
                             caption_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Technical Details Page
        story.append(PageBreak())
        story.append(Paragraph("Technical Information", heading_style))
        
        tech_data = [
            ['Parameter', 'Details'],
            ['Detection Method', 'Reference-based Image Subtraction + CNN Classification'],
            ['Classification Model', 'EfficientNet-B4 with PyTorch'],
            ['Image Processing', 'OpenCV + NumPy'],
            ['Preprocessing Steps', 'Alignment, Grayscale Conversion, Thresholding (Otsu)'],
            ['Defect Localization', 'Contour Detection & Bounding Box Extraction'],
            ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Processing Time', f'{processing_time:.2f} seconds'],
            ['Image Dimensions', f"{results['test'].shape[1]}×{results['test'].shape[0]} pixels"],
        ]
        
        tech_table = Table(tech_data, colWidths=[2.5*inch, 3.3*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(tech_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Defect Classification Legend
        story.append(Paragraph("Defect Classification Reference", section_style))
        
        legend_data = [
            ['Defect Type', 'Description', 'Severity'],
            ['Missing Hole', 'Required hole is absent or incomplete', 'High'],
            ['Mouse Bite', 'Small irregular edge defects', 'Medium'],
            ['Open Circuit', 'Broken or incomplete circuit trace', 'Critical'],
            ['Short', 'Unwanted connection between traces', 'Critical'],
            ['Spur', 'Unwanted protrusion from trace', 'Medium'],
            ['Spurious Copper', 'Excess copper in unintended areas', 'Low']
        ]
        
        legend_table = Table(legend_data, colWidths=[1.6*inch, 3*inch, 1.2*inch])
        legend_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(legend_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Footer
        footer_style = ParagraphStyle('Footer', parent=body_style, 
                                     fontSize=8, textColor=colors.HexColor('#9ca3af'),
                                     alignment=TA_CENTER)
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("_______________________________________________", footer_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("CircuitGuard - Automated PCB Defect Detection System", footer_style))
        story.append(Paragraph("This report is automatically generated and should be verified by qualified personnel", footer_style))
        story.append(Paragraph(f"Report ID: CG-{datetime.now().strftime('%Y%m%d%H%M%S')}", footer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
        
    finally:
        # Clean up temporary image files
        for img_path in temp_images:
            if os.path.exists(img_path):
                os.unlink(img_path)



def save_cv2_image_temp(cv2_image):
    """Save OpenCV image to temporary file with high quality"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_file.name, cv2_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return temp_file.name