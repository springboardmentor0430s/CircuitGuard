"""
CircuitGuard Enhanced PDF Report Generator Module
Generates comprehensive, multi-page PDF inspection reports with professional styling.

Features:
- Modular design with reusable components
- Complete error handling and validation
- Professional multi-page layout
- Advanced metrics and visualizations
- QR code generation for report verification
- Configurable styling and branding
"""

import io
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, 
    Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics import renderPDF


class PDFReportGenerator:
    """Enhanced PDF Report Generator with modular design."""
    
    def __init__(self, branding: Optional[Dict[str, Any]] = None):
        """
        Initialize the PDF generator with optional branding configuration.
        
        Args:
            branding: Dict with optional keys:
                - company_name: str
                - logo_path: str
                - primary_color: str (hex)
                - accent_color: str (hex)
        """
        self.branding = branding or {}
        self.primary_color = self.branding.get('primary_color', '#1e40af')
        self.accent_color = self.branding.get('accent_color', '#3b82f6')
        self.company_name = self.branding.get('company_name', 'CircuitGuard™')
        
        self.styles = self._create_styles()
        self.page_number = 0
        
    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """Create and return all paragraph styles."""
        styles = {}
        
        # Cover page styles
        styles['cover_title'] = ParagraphStyle(
            'CoverTitle', fontSize=28, fontName='Helvetica-Bold', 
            alignment=TA_CENTER, textColor=colors.HexColor(self.primary_color),
            spaceAfter=8, leading=34
        )
        
        styles['cover_subtitle'] = ParagraphStyle(
            'CoverSubtitle', fontSize=13, alignment=TA_CENTER,
            spaceAfter=4, textColor=colors.HexColor('#475569'), fontName='Helvetica'
        )
        
        styles['tagline'] = ParagraphStyle(
            'Tagline', fontSize=10, alignment=TA_CENTER,
            textColor=colors.HexColor('#64748b'), spaceAfter=20
        )
        
        # Section headers
        styles['heading1'] = ParagraphStyle(
            'Heading1', fontSize=15, fontName='Helvetica-Bold',
            spaceBefore=14, spaceAfter=8, textColor=colors.white,
            leftIndent=10, rightIndent=10,
            backColor=colors.HexColor(self.primary_color), borderPadding=8
        )
        
        styles['heading2'] = ParagraphStyle(
            'Heading2', fontSize=12, fontName='Helvetica-Bold',
            spaceBefore=12, spaceAfter=6, textColor=colors.HexColor(self.primary_color),
            borderWidth=1, borderColor=colors.HexColor('#cbd5e1'),
            borderPadding=4, leftIndent=6, backColor=colors.HexColor('#f1f5f9')
        )
        
        styles['heading3'] = ParagraphStyle(
            'Heading3', fontSize=10, fontName='Helvetica-Bold',
            spaceBefore=8, spaceAfter=4, textColor=colors.HexColor('#334155')
        )
        
        # Body text styles
        styles['body'] = ParagraphStyle(
            'CustomBody', fontSize=10, leading=15, alignment=TA_LEFT, spaceAfter=8
        )
        
        styles['body_justified'] = ParagraphStyle(
            'BodyJustified', fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=8
        )
        
        styles['bullet'] = ParagraphStyle(
            'BulletStyle', fontSize=10, leading=14, leftIndent=25,
            spaceAfter=6, bulletIndent=10
        )
        
        styles['caption'] = ParagraphStyle(
            'Caption', fontSize=8, textColor=colors.HexColor('#64748b'),
            alignment=TA_CENTER, leading=10, spaceAfter=4, fontName='Helvetica-Oblique'
        )
        
        styles['label'] = ParagraphStyle(
            'Label', fontSize=9, fontName='Helvetica-Bold',
            textColor=colors.HexColor(self.primary_color), spaceAfter=2
        )
        
        return styles
    
    def _validate_inputs(self, params: Dict, df_log, *image_buffers) -> None:
        """Validate input parameters and data."""
        required_params = ['threshold', 'min_area', 'confidence', 'defect_count']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        if params['confidence'] < 0 or params['confidence'] > 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        # Validate image buffers
        for i, buf in enumerate(image_buffers):
            if buf is not None and not hasattr(buf, 'getvalue'):
                raise ValueError(f"Image buffer {i} must be a BytesIO object")
    
    def _create_metric_box(self, label: str, value: str, 
                          subtext: str = "", color: str = None) -> Table:
        """Create a styled metric display box."""
        if color is None:
            color = self.primary_color
            
        metric_label_style = ParagraphStyle(
            'MetricLabel', fontSize=8, textColor=colors.HexColor('#64748b'),
            alignment=TA_CENTER
        )
        metric_value_style = ParagraphStyle('MetricValue', alignment=TA_CENTER)
        metric_sub_style = ParagraphStyle(
            'MetricSub', fontSize=7, textColor=colors.HexColor('#94a3b8'),
            alignment=TA_CENTER
        )
        
        data = [
            [Paragraph(f'<b>{label}</b>', metric_label_style)],
            [Paragraph(f'<b><font size=16 color="{color}">{value}</font></b>', 
                      metric_value_style)]
        ]
        
        if subtext:
            data.append([Paragraph(f'<i>{subtext}</i>', metric_sub_style)])
        
        metric_table = Table(data, colWidths=[1.8*inch])
        metric_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor(color)),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        return metric_table
    
    def _create_qr_code(self, data: str, size: float = 1.2*inch) -> Drawing:
        """Create a QR code for report verification."""
        qr = QrCodeWidget(data)
        bounds = qr.getBounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        drawing = Drawing(size, size, transform=[size/width, 0, 0, size/height, 0, 0])
        drawing.add(qr)
        return drawing
    
    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        return f"CG-{timestamp}"
    
    def _create_header_footer(self, canvas, doc):
        """Draw header and footer on each page."""
        canvas.saveState()
        
        # Enhanced header
        canvas.setFillColor(colors.HexColor(self.primary_color))
        canvas.rect(0, A4[1] - 0.55*inch, A4[0], 0.55*inch, fill=True, stroke=False)
        
        # Logo placeholder (circle with initials)
        canvas.setFillColor(colors.HexColor(self.accent_color))
        canvas.circle(0.95*inch, A4[1] - 0.28*inch, 0.15*inch, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawCentredString(0.95*inch, A4[1] - 0.3*inch, 'CG')
        
        # Header text
        canvas.setFillColor(colors.white)
        canvas.setFont('Helvetica-Bold', 13)
        canvas.drawString(1.25*inch, A4[1] - 0.32*inch, self.company_name)
        
        canvas.setFont('Helvetica', 9)
        canvas.drawRightString(A4[0] - 0.75*inch, A4[1] - 0.32*inch, 
                              "PCB Defect Analysis Report")
        
        # Footer
        canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
        canvas.setLineWidth(0.5)
        canvas.line(0.75*inch, 0.55*inch, A4[0] - 0.75*inch, 0.55*inch)
        
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.setFont('Helvetica', 7)
        footer_left = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        canvas.drawString(0.75*inch, 0.35*inch, footer_left)
        
        canvas.setFont('Helvetica-Bold', 7)
        canvas.drawRightString(A4[0] - 0.75*inch, 0.35*inch, f"Page {doc.page}")
        
        # Confidentiality notice
        canvas.setFont('Helvetica-Oblique', 6)
        canvas.drawCentredString(A4[0]/2, 0.2*inch, 
                                "CONFIDENTIAL - For Internal Use Only")
        
        canvas.restoreState()
    
    def _add_cover_page(self, story: List, params: Dict, df_log) -> str:
        """Add cover page to report."""
        story.append(Spacer(1, 1.2*inch))
        
        # Title and subtitle
        story.append(Paragraph(self.company_name, self.styles['cover_title']))
        story.append(Paragraph("Advanced PCB Defect Detection & Analysis System", 
                              self.styles['cover_subtitle']))
        story.append(Paragraph("AI-Powered Quality Assurance Platform", 
                              self.styles['tagline']))
        
        story.append(Spacer(1, 0.4*inch))
        
        # Report information with QR code
        report_id = self._generate_report_id()
        report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        
        # Create QR code with report ID and hash
        report_hash = hashlib.sha256(
            f"{report_id}{params['defect_count']}".encode()
        ).hexdigest()[:16]
        qr_data = f"CG:{report_id}:{report_hash}"
        qr_code = self._create_qr_code(qr_data)
        
        # Report info table with QR code
        info_inner_table = Table([
            [Paragraph('<b>Report ID:</b>', self.styles['label']), 
             Paragraph(report_id, self.styles['body'])],
            [Paragraph('<b>Generated:</b>', self.styles['label']), 
             Paragraph(report_date, self.styles['body'])],
            [Paragraph('<b>Analysis Engine:</b>', self.styles['label']), 
             Paragraph('EfficientNet-B4 DL Model v2.1', self.styles['body'])],
            [Paragraph('<b>Detection Method:</b>', self.styles['label']), 
             Paragraph('Differential + AI Classification', self.styles['body'])],
            [Paragraph('<b>Verification Hash:</b>', self.styles['label']), 
             Paragraph(f'<font face="Courier">{report_hash}</font>', self.styles['body'])],
        ], colWidths=[1.4*inch, 2.8*inch], style=TableStyle([
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        
        cover_info_data = [
            [Paragraph('<b>COMPREHENSIVE INSPECTION REPORT</b>', 
                      ParagraphStyle('CoverInfoTitle', fontSize=13, alignment=TA_CENTER,
                                   textColor=colors.white, fontName='Helvetica-Bold'))],
            [Spacer(1, 0.1*inch)],
            [Table([[info_inner_table, qr_code]], colWidths=[4.4*inch, 1.3*inch])]
        ]
        
        cover_info_table = Table(cover_info_data, colWidths=[5.9*inch])
        cover_info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor(self.primary_color)),
            ('BACKGROUND', (0,2), (-1,-1), colors.HexColor('#f8fafc')),
            ('BOX', (0,0), (-1,-1), 2, colors.HexColor(self.primary_color)),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('TOPPADDING', (0,2), (-1,-1), 15),
            ('BOTTOMPADDING', (0,2), (-1,-1), 15),
        ]))
        
        story.append(cover_info_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Key metrics
        quality_rating = self._get_quality_rating(params['defect_count'])
        rating_color = self._get_rating_color(params['defect_count'])
        
        metrics_data = [[
            self._create_metric_box('Total Defects', str(params['defect_count']),
                                   'Identified', 
                                   '#dc2626' if params['defect_count'] > 3 else '#f59e0b'),
            self._create_metric_box('Quality Status', quality_rating.split()[0],
                                   ' '.join(quality_rating.split()[1:]) if len(quality_rating.split()) > 1 else '',
                                   rating_color),
            self._create_metric_box('Confidence Level', 
                                   f"{params['confidence']*100:.0f}%",
                                   'AI Threshold', self.primary_color)
        ]]
        
        metrics_table = Table(metrics_data, colWidths=[1.9*inch]*3)
        metrics_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        
        story.append(Paragraph('<b>INSPECTION SUMMARY</b>',
                              ParagraphStyle('SummaryTitle', fontSize=11, alignment=TA_CENTER,
                                           textColor=colors.HexColor(self.primary_color), 
                                           spaceAfter=12)))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Severity indicator
        story.append(self._create_severity_box(params['defect_count']))
        story.append(PageBreak())
        
        return report_id
    
    def _get_quality_rating(self, defect_count: int) -> str:
        """Determine quality rating based on defect count."""
        if defect_count > 5:
            return 'FAIL'
        elif defect_count > 0:
            return 'ATTENTION REQUIRED'
        return 'PASS'
    
    def _get_rating_color(self, defect_count: int) -> str:
        """Get color for quality rating."""
        if defect_count > 5:
            return '#dc2626'
        elif defect_count > 0:
            return '#f59e0b'
        return '#16a34a'
    
    def _create_severity_box(self, defect_count: int) -> Table:
        """Create severity indicator box."""
        if defect_count > 0:
            severity_text = (f"⚠ <b>ACTION REQUIRED:</b> {defect_count} defect(s) "
                           "detected requiring investigation")
            box_color = colors.HexColor('#fef3c7')
            text_color = '#92400e'
            border_color = '#f59e0b'
        else:
            severity_text = ("✓ <b>INSPECTION PASSED:</b> No defects detected "
                           "within threshold parameters")
            box_color = colors.HexColor('#d1fae5')
            text_color = '#065f46'
            border_color = '#10b981'
        
        severity_box = Table([[Paragraph(severity_text,
                                        ParagraphStyle('Severity', fontSize=10,
                                                     alignment=TA_CENTER,
                                                     textColor=colors.HexColor(text_color)))]],
                            colWidths=[5.5*inch])
        severity_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), box_color),
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor(border_color)),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING', (0,0), (-1,-1), 15),
            ('RIGHTPADDING', (0,0), (-1,-1), 15),
        ]))
        
        return severity_box
    
    def _add_table_of_contents(self, story: List) -> None:
        """Add table of contents."""
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("TABLE OF CONTENTS", self.styles['heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        toc_data = [
            ['Section', 'Description', 'Page'],
            ['1', 'Executive Summary', '3'],
            ['2', 'Analysis Configuration & Parameters', '4'],
            ['3', 'Defect Distribution Analysis', '5'],
            ['4', 'Spatial Analysis & Correlation Study', '6'],
            ['5', 'Detection Pipeline Breakdown', '7'],
            ['6', 'Detailed Defect Log & Findings', '8'],
            ['7', 'Final Inspection Result (Annotated)', '9'],
            ['8', 'Recommendations & Action Items', '10'],
            ['', 'Appendix: Methodology & Disclaimer', '11'],
        ]
        
        toc_table = Table(toc_data, colWidths=[0.7*inch, 4.3*inch, 0.7*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor(self.primary_color)),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('ALIGN', (2,0), (2,-1), 'CENTER'),
            ('FONTSIZE', (0,1), (-1,-1), 9),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            *[('BACKGROUND', (0, i), (-1, i), colors.white) 
              for i in range(1, len(toc_data), 2)]
        ]))
        
        story.append(toc_table)
        story.append(PageBreak())
    
    def _add_executive_summary(self, story: List, params: Dict, df_log) -> None:
        """Add executive summary section."""
        story.append(Paragraph("SECTION 1", self.styles['label']))
        story.append(Paragraph("Executive Summary", self.styles['heading1']))
        story.append(Spacer(1, 0.15*inch))
        
        if params['defect_count'] > 0:
            unique_types = len(df_log['Class'].unique()) if not df_log.empty else 0
            summary_text = f"""
            This comprehensive automated inspection has identified <b>{params['defect_count']} 
            manufacturing defect(s)</b> on the analyzed printed circuit board (PCB) using 
            state-of-the-art computer vision algorithms combined with deep learning classification 
            techniques. The detected anomalies have been categorized across <b>{unique_types} 
            distinct defect classes</b>, each classified based on morphological characteristics, 
            spatial patterns, and similarity to known defect signatures in the training dataset.
            """
            story.append(Paragraph(summary_text, self.styles['body_justified']))
            story.append(Spacer(1, 0.1*inch))
            
            story.append(Paragraph("Critical Findings & Impact Assessment", 
                                 self.styles['heading3']))
            critical_text = """
            The inspection reveals manufacturing anomalies that require immediate attention and 
            corrective action. Each identified defect has been assigned a confidence score 
            indicating the model's certainty in its classification. Detailed spatial coordinates, 
            dimensional measurements, and visual evidence are provided in subsequent sections to 
            facilitate root cause analysis, process optimization, and corrective action planning.
            """
            story.append(Paragraph(critical_text, self.styles['body_justified']))
            
            if params['defect_count'] > 3:
                story.append(Spacer(1, 0.1*inch))
                warning_box = Table([[Paragraph(
                    '<b>⚠ HIGH DEFECT DENSITY ALERT:</b> The number of defects detected '
                    'exceeds normal thresholds. Immediate process review and quality control '
                    'intervention recommended.',
                    ParagraphStyle('Warning', fontSize=9, 
                                 textColor=colors.HexColor('#92400e'), leading=12)
                )]], colWidths=[6.5*inch])
                warning_box.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fef3c7')),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#f59e0b')),
                    ('TOPPADDING', (0,0), (-1,-1), 8),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                    ('LEFTPADDING', (0,0), (-1,-1), 12),
                    ('RIGHTPADDING', (0,0), (-1,-1), 12),
                ]))
                story.append(warning_box)
        else:
            summary_text = """
            <b>✓ Inspection Result: PASSED</b><br/><br/>
            The automated inspection system did not identify any manufacturing defects meeting 
            the specified detection criteria and confidence thresholds. The test PCB appears to 
            conform to the reference template within acceptable manufacturing tolerances. All 
            analyzed regions fall within normal variation parameters.
            """
            story.append(Paragraph(summary_text, self.styles['body']))
            story.append(Spacer(1, 0.1*inch))
            
            info_box = Table([[Paragraph(
                '<b>ℹ Note:</b> While no defects were detected by the automated system, manual '
                'verification by qualified personnel is recommended for mission-critical '
                'applications or safety-sensitive components.',
                ParagraphStyle('Info', fontSize=9, 
                             textColor=colors.HexColor(self.primary_color), leading=12)
            )]], colWidths=[6.5*inch])
            info_box.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#dbeafe')),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor(self.accent_color)),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 12),
                ('RIGHTPADDING', (0,0), (-1,-1), 12),
            ]))
            story.append(info_box)
        
        story.append(PageBreak())
    
    def _add_analysis_parameters(self, story: List, params: Dict) -> None:
        """Add analysis configuration section."""
        story.append(Paragraph("SECTION 2", self.styles['label']))
        story.append(Paragraph("Analysis Configuration & Parameters", 
                             self.styles['heading1']))
        story.append(Spacer(1, 0.15*inch))
        
        intro_text = """
        The following parameters were configured for this inspection cycle. These settings 
        directly influence the sensitivity, accuracy, and detection capabilities of the 
        automated quality control system.
        """
        story.append(Paragraph(intro_text, self.styles['body_justified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Parameters table
        param_data = [
            ['Parameter', 'Value', 'Description'],
            ['Difference Threshold', str(params['threshold']), 
             'Pixel intensity difference threshold for anomaly detection'],
            ['Minimum Defect Area', f"{params['min_area']} px²", 
             'Minimum contour area to classify as valid defect'],
            ['AI Confidence Threshold', f"{params['confidence']*100:.1f}%", 
             'Minimum classification confidence for defect categorization'],
            ['Detection Algorithm', 'Differential Analysis', 
             'Primary method for identifying manufacturing variations'],
            ['Classification Model', 'EfficientNet-B4', 
             'Deep learning architecture for defect classification'],
            ['Image Resolution', 'Full Resolution', 
             'Processing performed at native image resolution'],
        ]
        
        param_table = Table(param_data, colWidths=[2*inch, 1.5*inch, 3*inch])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor(self.primary_color)),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('ALIGN', (0,0), (1,-1), 'LEFT'),
            ('ALIGN', (2,0), (2,-1), 'LEFT'),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            *[('BACKGROUND', (0, i), (-1, i), colors.white) 
              for i in range(1, len(param_data), 2)]
        ]))
        
        story.append(param_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Add parameter explanation
        story.append(Paragraph("Parameter Selection Rationale", self.styles['heading3']))
        rationale_text = """
        The threshold and confidence values were selected based on industry best practices 
        and empirical testing to balance detection sensitivity with false positive rates. 
        The minimum area filter eliminates noise and minor surface variations that do not 
        impact functional performance.
        """
        story.append(Paragraph(rationale_text, self.styles['body_justified']))
        story.append(PageBreak())
    
    def _add_image_with_caption(self, story: List, image_buf, caption: str,
                               width: float = 6*inch) -> None:
        """Add an image with caption to the story."""
        if image_buf is not None:
            try:
                image_buf.seek(0)
                img = RLImage(image_buf, width=width, height=width*0.75)
                story.append(img)
                story.append(Spacer(1, 0.05*inch))
                story.append(Paragraph(f"<i>Figure: {caption}</i>", self.styles['caption']))
            except Exception as e:
                story.append(Paragraph(f"[Image unavailable: {str(e)}]", 
                                     self.styles['caption']))
    
    def _add_defect_charts(self, story: List, bar_chart_buf, pie_chart_buf) -> None:
        """Add defect distribution charts."""
        story.append(Paragraph("SECTION 3", self.styles['label']))
        story.append(Paragraph("Defect Distribution Analysis", self.styles['heading1']))
        story.append(Spacer(1, 0.15*inch))
        
        dist_text = """
        The following visualizations provide statistical analysis of detected defects, 
        categorized by type and quantity. This data aids in identifying systematic issues 
        and patterns in the manufacturing process.
        """
        story.append(Paragraph(dist_text, self.styles['body_justified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Bar chart
        self._add_image_with_caption(story, bar_chart_buf,
                                    "Defect count by classification type", width=5.5*inch)
        story.append(Spacer(1, 0.2*inch))
        
        # Pie chart
        self._add_image_with_caption(story, pie_chart_buf,
                                    "Proportional distribution of defect types", width=4.5*inch)
        
        story.append(PageBreak())
    
    def _add_spatial_analysis(self, story: List, heatmap_buf, scatter_buf) -> None:
        """Add spatial analysis section."""
        story.append(Paragraph("SECTION 4", self.styles['label']))
        story.append(Paragraph("Spatial Analysis & Correlation Study", 
                             self.styles['heading1']))
        story.append(Spacer(1, 0.15*inch))
        
        spatial_text = """
        Spatial distribution analysis reveals patterns in defect occurrence across the PCB 
        surface. Clustering may indicate localized manufacturing issues, tooling problems, 
        or environmental factors affecting specific board regions.
        """
        story.append(Paragraph(spatial_text, self.styles['body_justified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Heatmap
        self._add_image_with_caption(story, heatmap_buf,
                                    "Defect density heatmap showing spatial concentration", 
                                    width=5.5*inch)
        story.append(Spacer(1, 0.2*inch))
        
        # Scatter plot
        self._add_image_with_caption(story, scatter_buf,
                                    "Defect scatter plot with size-encoded markers", 
                                    width=5.5*inch)
        
        story.append(PageBreak())
    
    def _add_detection_pipeline(self, story: List, diff_image_buf, mask_image_buf) -> None:
        """Add detection pipeline section."""
        story.append(Paragraph("SECTION 5", self.styles['label']))
        story.append(Paragraph("Detection Pipeline Breakdown", self.styles['heading1']))
        story.append(Spacer(1, 0.15*inch))
        
        pipeline_text = """
        The detection pipeline consists of multiple stages: image preprocessing, differential 
        analysis, morphological operations, contour detection, and AI-based classification. 
        The following images illustrate intermediate processing stages.
        """
        story.append(Paragraph(pipeline_text, self.styles['body_justified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Difference image
        story.append(Paragraph("Stage 1: Differential Analysis", self.styles['heading3']))
        self._add_image_with_caption(story, diff_image_buf,
                                    "Pixel-wise difference between test and reference images",
                                    width=5*inch)
        story.append(Spacer(1, 0.2*inch))
        
        # Mask image
        story.append(Paragraph("Stage 2: Binary Mask Generation", self.styles['heading3']))
        self._add_image_with_caption(story, mask_image_buf,
                                    "Thresholded binary mask highlighting anomalous regions",
                                    width=5*inch)
        
        story.append(PageBreak())
    
    def _add_defect_log(self, story: List, df_log) -> None:
        """Add detailed defect log."""
        story.append(Paragraph("SECTION 6", self.styles['label']))
        story.append(Paragraph("Detailed Defect Log & Findings", self.styles['heading1']))
        story.append(Spacer(1, 0.15*inch))
        
        if df_log.empty:
            story.append(Paragraph("No defects were detected during this inspection cycle.",
                                 self.styles['body']))
        else:
            log_text = """
            The following table provides comprehensive details for each detected defect, 
            including classification, confidence level, spatial coordinates, and dimensional 
            measurements. This data supports traceability and quality documentation requirements.
            """
            story.append(Paragraph(log_text, self.styles['body_justified']))
            story.append(Spacer(1, 0.15*inch))
            
            # Build defect table
            table_data = [['ID', 'Class', 'Confidence', 'Position (x,y)', 'Size (w×h)']]
            
            for _, row in df_log.iterrows():
                table_data.append([
                    str(row['Defect #']),
                    row['Class'],
                    row['Confidence'],
                    row['Position'],
                    row['Size']
                ])
            
            defect_table = Table(table_data, colWidths=[0.5*inch, 1.5*inch, 1*inch, 
                                                       1.8*inch, 1.5*inch])
            defect_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor(self.primary_color)),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 8),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                *[('BACKGROUND', (0, i), (-1, i), colors.white) 
                  for i in range(1, len(table_data), 2)]
            ]))
            
            story.append(defect_table)
        
        story.append(PageBreak())
    
    def _add_final_result(self, story: List, result_image_buf) -> None:
        """Add final annotated result."""
        story.append(Paragraph("SECTION 7", self.styles['label']))
        story.append(Paragraph("Final Inspection Result (Annotated)", 
                             self.styles['heading1']))
        story.append(Spacer(1, 0.15*inch))
        
        result_text = """
        The annotated image below displays all detected defects with bounding boxes, 
        classification labels, and confidence scores overlaid on the test PCB image.
        """
        story.append(Paragraph(result_text, self.styles['body_justified']))
        story.append(Spacer(1, 0.15*inch))
        
        self._add_image_with_caption(story, result_image_buf,
                                    "Complete inspection result with all detected defects annotated",
                                    width=6.5*inch)
        
        story.append(PageBreak())
    
    def _add_recommendations(self, story: List, params: Dict, df_log) -> None:
        """Add recommendations section."""
        story.append(Paragraph("SECTION 8", self.styles['label']))
        story.append(Paragraph("Recommendations & Action Items", self.styles['heading1']))
        story.append(Spacer(1, 0.15*inch))
        
        if params['defect_count'] > 0:
            # Generate recommendations based on defect analysis
            recommendations = []
            
            if params['defect_count'] > 5:
                recommendations.append(
                    "Immediate process halt recommended for comprehensive root cause analysis"
                )
                recommendations.append(
                    "Conduct full equipment calibration and tooling inspection"
                )
            
            if not df_log.empty:
                top_defect = df_log.groupby('Class').size().idxmax()
                recommendations.append(
                    f"Focus corrective actions on {top_defect} defects (most prevalent type)"
                )
            
            recommendations.extend([
                "Implement enhanced in-process quality checks at identified defect locations",
                "Review environmental conditions (temperature, humidity) during manufacturing",
                "Conduct training refresher for production operators on defect prevention",
                "Schedule follow-up inspection after corrective actions implemented"
            ])
            
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"<b>{i}.</b> {rec}", self.styles['bullet']))
            
            story.append(Spacer(1, 0.15*inch))
            
            # Priority actions box
            priority_box = Table([[Paragraph(
                '<b>⚡ PRIORITY ACTIONS:</b> Address root causes within 24 hours. '
                'Document all corrective actions and re-inspect within next production cycle.',
                ParagraphStyle('Priority', fontSize=9, 
                             textColor=colors.HexColor('#92400e'), leading=12)
            )]], colWidths=[6.5*inch])
            priority_box.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fef3c7')),
                ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#f59e0b')),
                ('TOPPADDING', (0,0), (-1,-1), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
                ('LEFTPADDING', (0,0), (-1,-1), 12),
                ('RIGHTPADDING', (0,0), (-1,-1), 12),
            ]))
            story.append(priority_box)
        else:
            story.append(Paragraph(
                "<b>✓ No immediate actions required.</b> Continue standard quality monitoring "
                "procedures and maintain current process parameters.",
                self.styles['body']
            ))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(
                "Recommended to conduct periodic re-validation inspections to ensure continued "
                "process stability and compliance with quality standards.",
                self.styles['body']
            ))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Add appendix
        story.append(Paragraph("APPENDIX", self.styles['label']))
        story.append(Paragraph("Methodology & Disclaimer", self.styles['heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        methodology_text = """
        <b>Detection Methodology:</b> This report is generated using automated computer vision 
        and deep learning techniques. The system compares test PCB images against reference 
        templates, identifies deviations, and classifies defects using a pre-trained neural 
        network model. Detection accuracy depends on image quality, lighting conditions, and 
        the representativeness of the training data.<br/><br/>
        
        <b>Limitations:</b> Automated inspection systems may not detect all defect types, 
        particularly novel or rare anomalies not present in training data. This system should 
        complement, not replace, human inspection for critical applications.<br/><br/>
        
        <b>Disclaimer:</b> This report is for quality assurance purposes only. Final acceptance 
        decisions should be made by qualified personnel considering all relevant factors, 
        specifications, and regulatory requirements.
        """
        story.append(Paragraph(methodology_text, 
                             ParagraphStyle('Methodology', fontSize=8, leading=11, 
                                          spaceAfter=6, alignment=TA_JUSTIFY)))
    
    def generate_pdf_report(self, params: Dict, df_log, result_image_buf,
                           bar_chart_buf, pie_chart_buf, heatmap_buf,
                           scatter_buf, diff_image_buf, mask_image_buf) -> bytes:
        """
        Generate complete PDF report.
        
        Args:
            params: Analysis parameters dictionary
            df_log: DataFrame with defect log
            result_image_buf: Annotated result image
            bar_chart_buf: Bar chart image
            pie_chart_buf: Pie chart image
            heatmap_buf: Heatmap image
            scatter_buf: Scatter plot image
            diff_image_buf: Difference image
            mask_image_buf: Binary mask image
        
        Returns:
            bytes: PDF file content
        
        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs
        self._validate_inputs(params, df_log, result_image_buf, bar_chart_buf,
                             pie_chart_buf, heatmap_buf, scatter_buf,
                             diff_image_buf, mask_image_buf)
        
        # Create PDF buffer and document
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            topMargin=0.6*inch,
            bottomMargin=0.7*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        
        story = []
        
        try:
            # Build all sections
            self._add_cover_page(story, params, df_log)
            self._add_table_of_contents(story)
            self._add_executive_summary(story, params, df_log)
            self._add_analysis_parameters(story, params)
            self._add_defect_charts(story, bar_chart_buf, pie_chart_buf)
            self._add_spatial_analysis(story, heatmap_buf, scatter_buf)
            self._add_detection_pipeline(story, diff_image_buf, mask_image_buf)
            self._add_defect_log(story, df_log)
            self._add_final_result(story, result_image_buf)
            self._add_recommendations(story, params, df_log)
            
            # Build PDF with header/footer
            doc.build(story, onFirstPage=self._create_header_footer,
                     onLaterPages=self._create_header_footer)
            
            return pdf_buffer.getvalue()
            
        except Exception as e:
            raise RuntimeError(f"PDF generation failed: {str(e)}") from e


# Convenience function for backwards compatibility
def generate_pdf_report(params, df_log, result_image_buf, bar_chart_buf,
                       pie_chart_buf, heatmap_buf, scatter_buf,
                       diff_image_buf, mask_image_buf) -> bytes:
    """
    Generate PDF report using default settings.
    
    This is a convenience wrapper around the PDFReportGenerator class
    for backwards compatibility with existing code.
    """
    generator = PDFReportGenerator()
    return generator.generate_pdf_report(
        params, df_log, result_image_buf, bar_chart_buf,
        pie_chart_buf, heatmap_buf, scatter_buf,
        diff_image_buf, mask_image_buf
    )