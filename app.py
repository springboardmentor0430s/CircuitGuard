"""
CircuitGuard-PCB Web Application (Complete & Final)
Streamlit-based UI for PCB defect detection and classification
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
from datetime import datetime
import os
import torch
import json
import zipfile

from src.web_app.backend import get_backend
from src.web_app.batch_processor import BatchProcessor
from src.web_app.batch_ui import batch_upload_ui
from src.web_app.pdf_report import create_pdf_report
from src.utils.visualization import plot_defect_scatter, plot_defect_area_hist
from src.model.inference import compare_with_ground_truth

# Display / processing max dimension (preserve aspect ratio)
MAX_DISPLAY_DIM = 1024


def prepare_pdf_visualizations(result, formatted, timestamp):
    """
    Prepare all visualizations for the PDF report
    
    Args:
        result: Raw inspection result
        formatted: Formatted result data
        timestamp: Current timestamp for file naming
        
    Returns:
        Dictionary with paths to generated visualizations
    """
    try:
        os.makedirs('temp/charts', exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        raise Exception("Failed to create required directories. Please check permissions.")
    
    viz_paths = {
        'test_image': None,
        'template_image': None,
        'alignment_viz': None,
        'difference_map': None,
        'defect_overlay': None,
        'charts': {}
    }
    
    # Save main images
    test_img_path = os.path.join('temp', f'test_{timestamp}.png')
    template_img_path = os.path.join('temp', f'template_{timestamp}.png')
    aligned_path = os.path.join('temp', f'aligned_{timestamp}.png')
    diff_path = os.path.join('temp', f'diff_{timestamp}.png')
    overlay_path = os.path.join('temp', f'overlay_{timestamp}.png')
    
    # Convert and save images
    test_pil = convert_cv_to_pil(result['images']['test'])
    template_pil = convert_cv_to_pil(result['images']['template'])
    aligned_pil = convert_cv_to_pil(result['images']['aligned'])
    diff_pil = convert_cv_to_pil(result['images']['difference_map'])
    overlay_pil = convert_cv_to_pil(result['images']['annotated'])
    
    test_pil.save(test_img_path)
    template_pil.save(template_img_path)
    aligned_pil.save(aligned_path)
    diff_pil.save(diff_path)
    overlay_pil.save(overlay_path)
    
    viz_paths.update({
        'test_image': test_img_path,
        'template_image': template_img_path,
        'alignment_viz': aligned_path,
        'difference_map': diff_path,
        'defect_overlay': overlay_path
    })
    
    # Generate statistical charts if we have defects
    if formatted['defect_details']:
        # 1. Defect distribution chart
        plt.figure(figsize=(8, 5))
        plt.bar(formatted['class_distribution'].keys(), 
                formatted['class_distribution'].values(),
                color='steelblue')
        plt.title('Defect Type Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        dist_chart_path = os.path.join('temp/charts', f'dist_{timestamp}.png')
        plt.savefig(dist_chart_path)
        plt.close()
        viz_paths['charts']['defect_distribution'] = dist_chart_path
        
        # 2. Confidence histogram
        confidences = [d['confidence (%)'] for d in formatted['defect_details']]
        plt.figure(figsize=(8, 5))
        plt.hist(confidences, bins=20, color='lightcoral')
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence (%)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        conf_chart_path = os.path.join('temp/charts', f'conf_{timestamp}.png')
        plt.savefig(conf_chart_path)
        plt.close()
        viz_paths['charts']['confidence_histogram'] = conf_chart_path
        
        # 3. Size distribution
        sizes = [d['area (px²)'] for d in formatted['defect_details']]
        plt.figure(figsize=(8, 5))
        plt.hist(sizes, bins=20, color='lightgreen')
        plt.title('Defect Size Distribution')
        plt.xlabel('Area (pixels²)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        size_chart_path = os.path.join('temp/charts', f'size_{timestamp}.png')
        plt.savefig(size_chart_path)
        plt.close()
        viz_paths['charts']['size_distribution'] = size_chart_path
    
    return viz_paths

# Page configuration
st.set_page_config(
    page_title="CircuitGuard-PCB",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================

def convert_to_grayscale(pil_image):
    """
    Convert PIL image to grayscale OpenCV format
    Handles both RGB and grayscale inputs
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        Grayscale OpenCV image (numpy array) or None if input is None
    """
    if pil_image is None:
        return None
        
    img_array = np.array(pil_image)
    
    if len(img_array.shape) == 2:
        return img_array
    
    elif len(img_array.shape) == 3:
        channels = img_array.shape[2]
        
        if channels == 1:
            return img_array.squeeze()
        elif channels == 3:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif channels == 4:
            rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    else:
        raise ValueError(f"Unexpected image shape: {img_array.shape}")


def convert_cv_to_pil(cv_image):
    """
    Convert OpenCV image to PIL
    Handles grayscale and color images
    
    Args:
        cv_image: OpenCV image (numpy array)
        
    Returns:
        PIL Image object
    """
    if cv_image is None:
        return None
    
    if len(cv_image.shape) == 2:
        return Image.fromarray(cv_image)
    
    elif len(cv_image.shape) == 3:
        channels = cv_image.shape[2]
        
        if channels == 1:
            return Image.fromarray(cv_image.squeeze())
        elif channels == 3:
            return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        elif channels == 4:
            return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA))
    
    else:
        raise ValueError(f"Unexpected image shape: {cv_image.shape}")


def resize_for_display(pil_image, max_size=600):
    """
    Resize PIL image for optimal display while maintaining aspect ratio
    
    Args:
        pil_image: PIL Image object
        max_size: Maximum dimension (width or height)
        
    Returns:
        Resized PIL Image
    """
    if pil_image is None:
        return None
    
    width, height = pil_image.size
    
    if width <= max_size and height <= max_size:
        return pil_image
    
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)
    
    return pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)


@st.cache_resource
def load_backend():
    """Load backend (cached) - FIXED: Function was missing"""
    return get_backend()


# ==================== NEW ENHANCED VISUALIZATION FUNCTIONS ====================

def create_defect_location_heatmap(defect_details, image_shape):
    """Create 2D heatmap showing defect locations on PCB"""
    if not defect_details:
        return None
    
    # Extract locations
    locations = []
    for d in defect_details:
        loc_str = d['location'].strip('()').split(',')
        x, y = int(loc_str[0]), int(loc_str[1])
        locations.append([x, y])
    
    locations = np.array(locations)
    
    # Create heatmap
    fig = go.Figure(data=go.Histogram2d(
        x=locations[:, 0],
        y=locations[:, 1],
        colorscale='Hot',
        reversescale=True,
        nbinsx=30,
        nbinsy=30
    ))
    
    fig.update_layout(
        title="Defect Location Heatmap",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig


def create_defect_scatter_map(defect_details, image_shape):
    """Create interactive scatter plot of defect locations with details"""
    if not defect_details:
        return None
    
    locations = []
    sizes = []
    confidences = []
    types = []
    colors_map = {
        'mousebite': 'blue',
        'open': 'green', 
        'short': 'red',
        'spur': 'cyan',
        'copper': 'magenta',
        'pinhole': 'yellow'
    }
    colors = []
    
    for d in defect_details:
        loc_str = d['location'].strip('()').split(',')
        x, y = int(loc_str[0]), int(loc_str[1])
        locations.append([x, y])
        sizes.append(d['area (px²)'])
        confidences.append(d['confidence (%)'])
        types.append(d['type'])
        colors.append(colors_map.get(d['type'], 'gray'))
    
    locations = np.array(locations)
    
    fig = go.Figure()
    
    # Add scatter plot for each defect type
    for defect_type in set(types):
        mask = [t == defect_type for t in types]
        indices = [i for i, m in enumerate(mask) if m]
        
        fig.add_trace(go.Scatter(
            x=locations[indices, 0],
            y=locations[indices, 1],
            mode='markers',
            name=defect_type.title(),
            marker=dict(
                size=[sizes[i]/10 for i in indices],  # Scale size for visibility
                color=[confidences[i] for i in indices],
                colorscale='Viridis',
                showscale=True if defect_type == types[0] else False,
                colorbar=dict(title="Confidence %"),
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=[f"Type: {types[i]}<br>Confidence: {confidences[i]:.1f}%<br>Area: {sizes[i]} px²" 
                  for i in indices],
            hovertemplate='<b>%{text}</b><br>Position: (%{x}, %{y})<extra></extra>'
        ))
    
    fig.update_layout(
        title="Spatial Distribution of Defects",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        height=600,
        hovermode='closest',
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange='reversed'),
        legend=dict(x=1.05, y=1)
    )
    
    return fig


def create_confidence_box_plot(defect_details):
    """Create box plot comparing confidence across defect types"""
    if not defect_details:
        return None
    
    # Group by type
    defect_types = {}
    for d in defect_details:
        dtype = d['type']
        if dtype not in defect_types:
            defect_types[dtype] = []
        defect_types[dtype].append(d['confidence (%)'])
    
    fig = go.Figure()
    
    for dtype, confidences in defect_types.items():
        fig.add_trace(go.Box(
            y=confidences,
            name=dtype.title(),
            boxmean='sd',  # Show mean and standard deviation
            hovertemplate='<b>%{fullData.name}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title="Confidence Distribution by Defect Type",
        yaxis_title="Confidence (%)",
        xaxis_title="Defect Type",
        height=450,
        showlegend=False
    )
    
    return fig


def create_size_vs_type_violin(defect_details):
    """Create violin plot showing size distribution per defect type"""
    if not defect_details:
        return None
    
    # Group by type
    defect_types = {}
    for d in defect_details:
        dtype = d['type']
        if dtype not in defect_types:
            defect_types[dtype] = []
        defect_types[dtype].append(d['area (px²)'])
    
    fig = go.Figure()
    
    for dtype, sizes in defect_types.items():
        fig.add_trace(go.Violin(
            y=sizes,
            name=dtype.title(),
            box_visible=True,
            meanline_visible=True,
            hovertemplate='<b>%{fullData.name}</b><br>Area: %{y} px²<extra></extra>'
        ))
    
    fig.update_layout(
        title="Defect Size Distribution by Type",
        yaxis_title="Area (pixels²)",
        xaxis_title="Defect Type",
        height=450
    )
    
    return fig


def create_confidence_gauge(avg_confidence):
    """Create gauge chart for average confidence"""
    # Extract numeric value from string like "95.5% (High)"
    try:
        conf_value = float(avg_confidence.split('%')[0])
    except:
        conf_value = 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=conf_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Average Confidence", 'font': {'size': 24}},
        delta={'reference': 90, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 80], 'color': '#ffffcc'},
                {'range': [80, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))
    
    fig.update_layout(height=350)
    
    return fig


def create_defect_sunburst(defect_details):
    """Create sunburst chart showing defect hierarchy"""
    if not defect_details:
        return None
    
    # Create hierarchical data
    labels = ['All Defects']
    parents = ['']
    values = [len(defect_details)]
    colors = ['lightgray']
    
    # Group by type
    type_counts = {}
    type_conf = {}
    for d in defect_details:
        dtype = d['type']
        if dtype not in type_counts:
            type_counts[dtype] = 0
            type_conf[dtype] = []
        type_counts[dtype] += 1
        type_conf[dtype].append(d['confidence (%)'])
    
    # Add type level
    for dtype, count in type_counts.items():
        labels.append(dtype.title())
        parents.append('All Defects')
        values.append(count)
        avg_conf = np.mean(type_conf[dtype])
        # Color based on confidence
        if avg_conf >= 95:
            colors.append('#2ca02c')  # Green
        elif avg_conf >= 85:
            colors.append('#ff7f0e')  # Orange
        else:
            colors.append('#d62728')  # Red
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Defect Type Hierarchy",
        height=450
    )
    
    return fig


def create_3d_defect_scatter(defect_details):
    """Create 3D scatter plot: location + confidence"""
    if not defect_details:
        return None
    
    locations = []
    confidences = []
    sizes = []
    types = []
    
    for d in defect_details:
        loc_str = d['location'].strip('()').split(',')
        x, y = int(loc_str[0]), int(loc_str[1])
        locations.append([x, y])
        confidences.append(d['confidence (%)'])
        sizes.append(d['area (px²)'])
        types.append(d['type'])
    
    locations = np.array(locations)
    
    fig = go.Figure()
    
    # Add scatter for each type
    for defect_type in set(types):
        mask = [t == defect_type for t in types]
        indices = [i for i, m in enumerate(mask) if m]
        
        fig.add_trace(go.Scatter3d(
            x=locations[indices, 0],
            y=locations[indices, 1],
            z=[confidences[i] for i in indices],
            mode='markers',
            name=defect_type.title(),
            marker=dict(
                size=[sizes[i]/50 for i in indices],
                color=[confidences[i] for i in indices],
                colorscale='Viridis',
                showscale=True if defect_type == types[0] else False,
                colorbar=dict(title="Confidence %", x=1.1),
                line=dict(width=1, color='white')
            ),
            text=[f"{types[i]}: {confidences[i]:.1f}%" for i in indices],
            hovertemplate='<b>%{text}</b><br>Position: (%{x}, %{y})<br>Size: %{marker.size} px²<extra></extra>'
        ))
    
    fig.update_layout(
        title="3D Defect Visualization (X, Y, Confidence)",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Confidence (%)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=600,
        legend=dict(x=0.7, y=0.9)
    )
    
    return fig


def create_defect_timeline_simulation(defect_details):
    """Simulate detection order and create timeline"""
    if not defect_details:
        return None
    
    # Sort by confidence (simulating detection order)
    sorted_defects = sorted(defect_details, key=lambda x: x['confidence (%)'], reverse=True)
    
    cumulative_counts = {}
    detection_order = []
    
    for i, d in enumerate(sorted_defects, 1):
        dtype = d['type']
        if dtype not in cumulative_counts:
            cumulative_counts[dtype] = 0
        cumulative_counts[dtype] += 1
        
        detection_order.append({
            'step': i,
            'type': dtype,
            'cumulative': dict(cumulative_counts)
        })
    
    # Create bar chart
    fig = go.Figure()
    
    all_types = sorted(set([d['type'] for d in defect_details]))
    
    for step_data in detection_order[::max(1, len(detection_order)//20)]:  # Sample for animation
        step = step_data['step']
        counts = [step_data['cumulative'].get(t, 0) for t in all_types]
        
        fig.add_trace(go.Bar(
            x=all_types,
            y=counts,
            name=f"Step {step}",
            text=counts,
            textposition='auto'
        ))
    
    # Use only the last frame
    fig.data = [fig.data[-1]]
    
    fig.update_layout(
        title="Defect Detection Summary (by Confidence Order)",
        xaxis_title="Defect Type",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )
    
    return fig


# ==================== CONTINUE WITH EXISTING FUNCTIONS ====================

def generate_text_log(result: dict, formatted: dict) -> str:
    """Generate text inspection log"""
    
    log = []
    log.append("="*80)
    log.append("PCB INSPECTION LOG")
    log.append("="*80)
    log.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("")
    
    log.append("SUMMARY:")
    log.append("-"*80)
    log.append(f"Total Defects: {formatted['summary']['total_defects']}")
    log.append(f"Average Confidence: {formatted['summary']['average_confidence']}")
    log.append(f"Processing Time: {formatted['processing_time']}")
    log.append(f"Alignment Matches: {formatted['summary']['alignment_matches']}")
    log.append(f"Alignment Inliers: {formatted['summary']['alignment_inliers']}")
    log.append("")
    
    log.append("DEFECT DISTRIBUTION:")
    log.append("-"*80)
    for defect_type, count in formatted['class_distribution'].items():
        log.append(f"{defect_type}: {count}")
    log.append("")
    
    if formatted['defect_details']:
        log.append("DEFECT DETAILS:")
        log.append("-"*80)
        for defect in formatted['defect_details']:
            log.append(f"\nDefect #{defect['id']}:")
            log.append(f"  Type: {defect['type']}")
            log.append(f"  Confidence: {defect['confidence (%)']:.1f}%")
            log.append(f"  Location: {defect['location']}")
            log.append(f"  Area: {defect['area (px²)']} pixels²")
    
    log.append("")
    log.append("="*80)
    log.append("END OF LOG")
    log.append("="*80)
    
    return "\n".join(log)


def create_complete_package(result: dict, formatted: dict, backend, annotated_pil) -> BytesIO:
    """
    Create ZIP package with essential export files
    """
    
    from src.web_app.pdf_report import create_pdf_report
    import matplotlib.pyplot as plt
    
    zip_buffer = BytesIO()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        os.makedirs('temp/charts', exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        # ==================== IMAGES (Full Resolution) ====================
        
        # 1. Annotated image
        img_buffer = BytesIO()
        annotated_pil.save(img_buffer, format='PNG')
        zip_file.writestr(f"images/annotated_{timestamp}.png", img_buffer.getvalue())
        
        # 2. Difference map
        diff_pil = convert_cv_to_pil(result['images']['difference_map'])
        diff_buffer = BytesIO()
        diff_pil.save(diff_buffer, format='PNG')
        zip_file.writestr(f"images/difference_map_{timestamp}.png", diff_buffer.getvalue())
        
        # 3. Defect mask
        mask_pil = convert_cv_to_pil(result['images']['mask'])
        mask_buffer = BytesIO()
        mask_pil.save(mask_buffer, format='PNG')
        zip_file.writestr(f"images/defect_mask_{timestamp}.png", mask_buffer.getvalue())
        
        # 4. Test image
        test_pil = convert_cv_to_pil(result['images']['test'])
        test_buffer = BytesIO()
        test_pil.save(test_buffer, format='PNG')
        zip_file.writestr(f"images/test_original_{timestamp}.png", test_buffer.getvalue())
        
        # 5. Template image
        template_pil = convert_cv_to_pil(result['images']['template'])
        template_buffer = BytesIO()
        template_pil.save(template_buffer, format='PNG')
        zip_file.writestr(f"images/template_{timestamp}.png", template_buffer.getvalue())
        
        # 6. Aligned image
        aligned_pil = convert_cv_to_pil(result['images']['aligned'])
        aligned_buffer = BytesIO()
        aligned_pil.save(aligned_buffer, format='PNG')
        zip_file.writestr(f"images/aligned_{timestamp}.png", aligned_buffer.getvalue())
        
        # ==================== CSV DATA FILES ====================
        
        if formatted['defect_details']:
            # 1. Complete defect details CSV
            defects_df = pd.DataFrame(formatted['defect_details'])
            csv_complete = defects_df.to_csv(index=False)
            zip_file.writestr(f"data/defects_complete_{timestamp}.csv", csv_complete)
            
            # 2. Type distribution CSV
            dist_data = {
                'Defect_Type': list(formatted['class_distribution'].keys()),
                'Count': list(formatted['class_distribution'].values()),
                'Percentage': [count/sum(formatted['class_distribution'].values())*100 
                              for count in formatted['class_distribution'].values()]
            }
            dist_df = pd.DataFrame(dist_data)
            zip_file.writestr(f"data/type_distribution_{timestamp}.csv", dist_df.to_csv(index=False))
            
            # 3. Statistical summary CSV
            stats_data = {
                'Metric': ['Total_Defects', 'Avg_Confidence_%', 'Min_Confidence_%', 
                          'Max_Confidence_%', 'Avg_Area_px2', 'Min_Area_px2', 'Max_Area_px2'],
                'Value': [
                    len(formatted['defect_details']),
                    np.mean([d['confidence (%)'] for d in formatted['defect_details']]),
                    min([d['confidence (%)'] for d in formatted['defect_details']]),
                    max([d['confidence (%)'] for d in formatted['defect_details']]),
                    np.mean([d['area (px²)'] for d in formatted['defect_details']]),
                    min([d['area (px²)'] for d in formatted['defect_details']]),
                    max([d['area (px²)'] for d in formatted['defect_details']])
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            zip_file.writestr(f"data/statistical_summary_{timestamp}.csv", stats_df.to_csv(index=False))
        
        # ==================== JSON DATA FILES ====================
        
        # 1. Complete results JSON
        json_complete = {
            'timestamp': datetime.now().isoformat(),
            'summary': formatted['summary'],
            'defects': formatted['defect_details'],
            'class_distribution': formatted['class_distribution'],
            'alignment_info': result['alignment_info'],
            'processing_time': formatted['processing_time']
        }
        zip_file.writestr(f"data/complete_results_{timestamp}.json", 
                         json.dumps(json_complete, indent=2))
        
        # 2. Summary only JSON
        summary_json = {
            'timestamp': datetime.now().isoformat(),
            'total_defects': formatted['summary']['total_defects'],
            'average_confidence': formatted['summary']['average_confidence'],
            'processing_time': formatted['processing_time'],
            'status': 'PASSED' if formatted['summary']['total_defects'] == 0 else 'FAILED'
        }
        zip_file.writestr(f"data/summary_{timestamp}.json", 
                         json.dumps(summary_json, indent=2))
        
        # ==================== VISUALIZATION CHARTS ====================
        
        if formatted['defect_details']:
            
            confidences = [d['confidence (%)'] for d in formatted['defect_details']]
            areas = [d['area (px²)'] for d in formatted['defect_details']]
            types = [d['type'] for d in formatted['defect_details']]
            
            # 1. Defect distribution bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(formatted['class_distribution'].keys(), 
                   formatted['class_distribution'].values(),
                   color='steelblue', edgecolor='black', linewidth=1.5)
            plt.title('Defect Type Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Defect Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            chart_path = f'temp/charts/distribution_{timestamp}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            zip_file.write(chart_path, f"charts/distribution_bar_{timestamp}.png")
            os.remove(chart_path)
            
            # 2. Confidence histogram
            plt.figure(figsize=(10, 6))
            plt.hist(confidences, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
            plt.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(confidences):.1f}%')
            plt.title('Confidence Score Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Confidence (%)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            chart_path = f'temp/charts/confidence_{timestamp}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            zip_file.write(chart_path, f"charts/confidence_histogram_{timestamp}.png")
            os.remove(chart_path)
            
            # 3. Size histogram
            plt.figure(figsize=(10, 6))
            plt.hist(areas, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
            plt.axvline(np.mean(areas), color='darkgreen', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(areas):.0f}px²')
            plt.title('Defect Size Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Area (pixels²)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            chart_path = f'temp/charts/size_{timestamp}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            zip_file.write(chart_path, f"charts/size_histogram_{timestamp}.png")
            os.remove(chart_path)
            
            # 4. Scatter plot - Area vs Confidence
            plt.figure(figsize=(10, 8))
            unique_types = list(set(types))
            colors_scatter = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
            
            for i, dtype in enumerate(unique_types):
                dtype_indices = [j for j, t in enumerate(types) if t == dtype]
                dtype_areas = [areas[j] for j in dtype_indices]
                dtype_confs = [confidences[j] for j in dtype_indices]
                plt.scatter(dtype_areas, dtype_confs, label=dtype.title(), 
                          alpha=0.6, s=100, c=[colors_scatter[i]], edgecolors='black', linewidth=1)
            
            plt.title('Defect Area vs Confidence', fontsize=14, fontweight='bold')
            plt.xlabel('Area (pixels²)')
            plt.ylabel('Confidence (%)')
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            chart_path = f'temp/charts/scatter_{timestamp}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            zip_file.write(chart_path, f"charts/scatter_area_confidence_{timestamp}.png")
            os.remove(chart_path)
            
            # 5. Box plot - Confidence by Type
            plt.figure(figsize=(10, 6))
            type_conf_dict = {}
            for d in formatted['defect_details']:
                if d['type'] not in type_conf_dict:
                    type_conf_dict[d['type']] = []
                type_conf_dict[d['type']].append(d['confidence (%)'])
            
            plt.boxplot(type_conf_dict.values(), labels=[t.title() for t in type_conf_dict.keys()])
            plt.title('Confidence Distribution by Type', fontsize=14, fontweight='bold')
            plt.ylabel('Confidence (%)')
            plt.xlabel('Defect Type')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            chart_path = f'temp/charts/boxplot_{timestamp}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            zip_file.write(chart_path, f"charts/boxplot_confidence_{timestamp}.png")
            os.remove(chart_path)
            
            # 6. Spatial distribution
            locations_x = []
            locations_y = []
            for d in formatted['defect_details']:
                loc_str = d['location'].strip('()').split(',')
                locations_x.append(int(loc_str[0]))
                locations_y.append(int(loc_str[1]))
            
            plt.figure(figsize=(10, 8))
            plt.hist2d(locations_x, locations_y, bins=20, cmap='hot', alpha=0.8)
            plt.colorbar(label='Defect Density')
            
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
            plt.gca().invert_yaxis()
            plt.tight_layout()
            chart_path = f'temp/charts/spatial_{timestamp}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            zip_file.write(chart_path, f"charts/spatial_distribution_{timestamp}.png")
            os.remove(chart_path)
        
        # ==================== TEXT REPORTS ====================
        
        # 1. Complete inspection log
        log_text = generate_text_log(result, formatted)
        zip_file.writestr(f"reports/inspection_log_{timestamp}.txt", log_text)
        
        # ==================== PDF REPORT ====================
        
        try:
            pdf_path = f"temp/report_{timestamp}.pdf"
            chart_images = {}
            
            # Generate charts for PDF
            if formatted['defect_details']:
                # Distribution chart
                plt.figure(figsize=(8, 5))
                plt.bar(formatted['class_distribution'].keys(), 
                        formatted['class_distribution'].values(),
                        color='steelblue')
                plt.title('Defect Type Distribution')
                plt.xticks(rotation=45)
                plt.tight_layout()
                dist_chart_path = f'temp/dist_{timestamp}.png'
                plt.savefig(dist_chart_path)
                plt.close()
                chart_images['distribution'] = dist_chart_path

                # Confidence histogram
                plt.figure(figsize=(8, 5))
                plt.hist([d['confidence (%)'] for d in formatted['defect_details']], 
                        bins=20, color='lightcoral')
                plt.title('Confidence Score Distribution')
                plt.xlabel('Confidence (%)')
                plt.ylabel('Frequency')
                plt.tight_layout()
                conf_chart_path = f'temp/conf_{timestamp}.png'
                plt.savefig(conf_chart_path)
                plt.close()
                chart_images['confidence'] = conf_chart_path
            
            # Prepare images for PDF
            result_images = {
                'annotated': result['images']['annotated'],
                'difference_map': result['images']['difference_map'],
                'mask': result['images']['mask'],
                'test': result['images']['test'],
                'template': result['images']['template'],
                'aligned': result['images']['aligned']
            }
            
            # Generate PDF
            create_pdf_report(
                formatted_data=formatted,
                output_path=pdf_path,
                class_names=backend.class_names,
                result_images=result_images,
                chart_paths=chart_images if chart_images else None
            )
            
            # Add PDF to ZIP
            if os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as pdf_file:
                    zip_file.writestr(f"reports/inspection_report_{timestamp}.pdf", 
                                    pdf_file.read())
            
            # Cleanup
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                for chart_path in chart_images.values():
                    if os.path.exists(chart_path):
                        os.remove(chart_path)
            except:
                pass
                
        except Exception as e:
            print(f"PDF generation error: {str(e)}")
        
        # ==================== README FILE ====================
        
        readme_content = f"""PCB INSPECTION PACKAGE
========================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: CircuitGuard-PCB v1.0
Model: EfficientNet-B4

CONTENTS:
---------
/images/          - All inspection images (6 files, full resolution)
/data/            - CSV and JSON data files (5 files)
/charts/          - Statistical visualization charts (6 files)
/reports/         - Text log and PDF report (2 files)

INSPECTION SUMMARY:
-------------------
Total Defects: {formatted['summary']['total_defects']}
Average Confidence: {formatted['summary']['average_confidence']}
Processing Time: {formatted['processing_time']}
Status: {'PASSED' if formatted['summary']['total_defects'] == 0 else 'FAILED'}

FILES:
------
Images:
  - annotated_{timestamp}.png       : Final result with labeled defects
  - difference_map_{timestamp}.png  : Difference visualization
  - defect_mask_{timestamp}.png     : Binary defect mask
  - test_original_{timestamp}.png   : Original test image
  - template_{timestamp}.png        : Reference template
  - aligned_{timestamp}.png         : Aligned test image

Data (CSV):
  - defects_complete_{timestamp}.csv      : All defect details
  - type_distribution_{timestamp}.csv     : Defect type breakdown
  - statistical_summary_{timestamp}.csv   : Key statistics

Data (JSON):
  - complete_results_{timestamp}.json : Complete inspection data
  - summary_{timestamp}.json          : Quick summary

Charts:
  - distribution_bar_{timestamp}.png           : Type distribution
  - confidence_histogram_{timestamp}.png       : Confidence scores
  - size_histogram_{timestamp}.png             : Defect sizes
  - scatter_area_confidence_{timestamp}.png    : Area vs confidence
  - boxplot_confidence_{timestamp}.png         : Confidence by type
  - spatial_distribution_{timestamp}.png       : Location heatmap

Reports:
  - inspection_log_{timestamp}.txt        : Human-readable log
  - inspection_report_{timestamp}.pdf     : Comprehensive PDF report

Generated by CircuitGuard-PCB System
"""
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer


def initialize_session_state():
    """Initialize session state variables"""
    if 'inspection_history' not in st.session_state:
        st.session_state.inspection_history = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'has_gpu' not in st.session_state:
        try:
            st.session_state.has_gpu = torch.cuda.is_available()
        except Exception as e:
            print(f"Error checking GPU availability: {str(e)}")
            st.session_state.has_gpu = False
    if 'pdf_data' not in st.session_state:
        st.session_state.pdf_data = None
    if 'zip_data' not in st.session_state:
        st.session_state.zip_data = None


def save_to_history(result, template_name, test_name):
    """Save inspection result to history"""
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'template': template_name,
        'test': test_name,
        'num_defects': result['num_defects'] if result['success'] else 0,
        'processing_time': result['processing_time'],
        'success': result['success']
    }
    st.session_state.inspection_history.append(history_entry)


# ==================== MAIN APPLICATION ====================

def main():
    """Main application"""
    
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🔍 CircuitGuard-PCB</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated PCB Defect Detection & Classification System</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Display app title if logo image is not available
        st.markdown("### 🔍 CircuitGuard-PCB")
        
        st.header("ℹ️ About")
        st.info(
            "**CircuitGuard-PCB** uses state-of-the-art computer vision and deep learning "
            "to automatically detect and classify manufacturing defects in PCBs.\n\n"
            "**Supported Defect Types:**\n"
            "🔸 Mousebite - Small perforations\n"
            "🔸 Open Circuit - Broken traces\n"
            "🔸 Short Circuit - Unintended connections\n"
            "🔸 Spur - Unwanted copper extensions\n"
            "🔸 Copper Issues - Missing/excess copper\n"
            "🔸 Pin-hole - Microscopic holes"
        )
        
        st.header("📋 Quick Start")
        st.markdown("""
        **Single Inspection:**
        1. 📤 Upload template (reference) image
        2. 📤 Upload test (inspection) image  
        3. 🔍 Click "Detect Defects"
        4. 📊 Review results
        5. 💾 Download reports
        
        **Batch Inspection:**
        1. Switch to "Batch Mode"
        2. 📦 Upload multiple pairs or ZIP
        3. 🚀 Process all at once
        4. 📥 Export results
        """)
        
        st.header("⚙️ System Status")
        backend = load_backend()  # FIXED: Line 1174
        
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("Device", "GPU" if st.session_state.has_gpu else "CPU")
        with status_col2:
            st.metric("Model", "EfficientNet-B4")
        
        st.text(f"Classes: {len(backend.class_names)}")
        if st.session_state.has_gpu:
            st.text(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            st.text("Memory: CPU Mode")
        
        # Mode selection
        st.markdown("---")
        st.header("🎯 Operation Mode")
        mode = st.radio(
            "Select mode:",
            ["Single Inspection", "Batch Processing", "Inspection History"],
            key="mode"
        )
    
    # Main content based on mode
    if mode == "Single Inspection":
        single_inspection_mode()
    elif mode == "Batch Processing":
        batch_upload_ui()
    else:
        history_mode()


# ==================== SINGLE INSPECTION MODE ====================

def single_inspection_mode():
    """Single inspection mode UI"""
    
    st.header("📤 Upload Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Template Image (Reference)")
        template_file = st.file_uploader(
            "Upload defect-free PCB template",
            type=['jpg', 'jpeg', 'png'],
            key="template",
            help="This is your defect-free reference image"
        )
        
        if template_file:
            template_image = Image.open(template_file)
            # Resize uploaded template for display (400px max)
            template_display = resize_for_display(template_image, max_size=400)
            st.image(template_display, caption=f"Template: {template_file.name}", width=400)
            st.caption(f"Original size: {template_image.size[0]} × {template_image.size[1]} pixels")
    
    with col2:
        st.subheader("Test Image (Inspection)")
        test_file = st.file_uploader(
            "Upload PCB to inspect for defects",
            type=['jpg', 'jpeg', 'png'],
            key="test",
            help="This is the PCB you want to inspect"
        )
        
        if test_file:
            test_image = Image.open(test_file)
            # Resize uploaded test image for display (400px max)
            test_display = resize_for_display(test_image, max_size=400)
            st.image(test_display, caption=f"Test: {test_file.name}", width=400)
            st.caption(f"Original size: {test_image.size[0]} × {test_image.size[1]} pixels")
    
    # Options
    st.markdown("---")
    st.subheader("🎛️ Processing Options")
    
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        show_intermediate = st.checkbox("Show processing steps", value=True)
    with col_opt2:
        show_details = st.checkbox("Show defect details", value=True)
    with col_opt3:
        auto_report = st.checkbox("Auto-generate PDF", value=False)
    
    # Process button
    if st.button("🔍 Detect Defects", type="primary", use_container_width=True):
        
        # Reset download session state
        st.session_state.pdf_data = None
        st.session_state.zip_data = None
        
        if template_file is None or test_file is None:
            st.error("⚠️ Please upload both template and test images!")
        else:
            # Resize images for processing (max 1024px dimension)
            template_resized = template_image.copy()
            test_resized = test_image.copy()
            template_resized.thumbnail((MAX_DISPLAY_DIM, MAX_DISPLAY_DIM), Image.Resampling.LANCZOS)
            test_resized.thumbnail((MAX_DISPLAY_DIM, MAX_DISPLAY_DIM), Image.Resampling.LANCZOS)
            
            # Convert PIL to OpenCV with smart conversion
            # Use resized images for processing to reduce memory and speed up inference
            template_cv = convert_to_grayscale(template_resized)
            test_cv = convert_to_grayscale(test_resized)
            
            # Progress bar
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            progress_text.text("🔄 Loading model...")
            progress_bar.progress(10)
            
            backend = load_backend()  # FIXED: Line 1287
            
            progress_text.text("🔄 Aligning images...")
            progress_bar.progress(30)
            
            progress_text.text("🔄 Detecting defects...")
            progress_bar.progress(60)
            
            result = backend.process_image_pair(template_cv, test_cv)
            
            progress_text.text("🔄 Classifying defects...")
            progress_bar.progress(90)
            
            progress_text.text("✓ Complete!")
            progress_bar.progress(100)
            
            # Save to session state and history
            st.session_state.current_result = result
            save_to_history(result, template_file.name, test_file.name)
            
            # Clear progress indicators
            progress_text.empty()
            progress_bar.empty()
            
            # Display results WITH backend parameter
            display_results(result, show_intermediate, show_details, auto_report, backend)


# ==================== DISPLAY RESULTS ====================

def display_results(result, show_intermediate=True, show_details=True, auto_report=False, backend=None):
    """Display inspection results"""
    
    if not result['success']:
        st.markdown(f'<div class="error-box">❌ <b>Processing Failed</b><br>{result["error"]}</div>', 
                   unsafe_allow_html=True)
        return
    
    # Load backend if not provided
    if backend is None:
        backend = get_backend()
    
    formatted = backend.format_results_for_display(result)
    
    st.markdown('<div class="success-box">✅ <b>Inspection Complete!</b> Defect detection and classification finished successfully.</div>', 
               unsafe_allow_html=True)
    
    # Add info about image optimization
    st.info("ℹ️ **Note:** Images are optimized for display (max 700px). Full-resolution images available in downloads.")
    
    # Summary metrics
    st.header("📊 Inspection Summary")
    
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        defect_color = "normal" if formatted['summary']['total_defects'] == 0 else "inverse"
        st.metric("Defects Found", 
                 formatted['summary']['total_defects'],
                 delta="Clean" if formatted['summary']['total_defects'] == 0 else "Issues detected",
                 delta_color=defect_color)
    
    with metric_col2:
        st.metric("Avg Confidence", formatted['summary']['average_confidence'])
    
    with metric_col3:
        st.metric("Processing Time", formatted['processing_time'])
    
    with metric_col4:
        st.metric("Feature Matches", formatted['summary']['alignment_matches'])
    
    with metric_col5:
        quality = formatted['summary']['alignment_inliers'] / formatted['summary']['alignment_matches'] * 100 if formatted['summary']['alignment_matches'] > 0 else 0
        st.metric("Alignment Quality", f"{quality:.0f}%")
    
    # ==================== ENHANCED DEFECT ANALYSIS SECTION ====================
    st.header("📈 Defect Analysis & Interactive Visualizations")
    
    class_stats = formatted['class_distribution']
    
    if sum(class_stats.values()) > 0:
        # Create multiple tabs for different visualization types
        viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
            "📊 Distribution", 
            "🗺️ Spatial Analysis", 
            "📐 Size & Confidence",
            "🎯 Comparison",
            "🔬 Advanced"
        ])
        
        with viz_tab1:
            st.subheader("Defect Type Distribution")
            
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                # Bar chart
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=list(class_stats.keys()),
                        y=list(class_stats.values()),
                        marker=dict(
                            color=list(class_stats.values()),
                            colorscale='Blues',
                            showscale=True,
                            colorbar=dict(title="Count")
                        ),
                        text=list(class_stats.values()),
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                    )
                ])
                fig_bar.update_layout(
                    title="Bar Chart - Defect Counts",
                    xaxis_title="Defect Class",
                    yaxis_title="Count",
                    height=400,
                    hovermode='x'
                )
                st.plotly_chart(fig_bar, use_container_width=True, key="bar_dist")
            
            with col_dist2:
                # Pie chart with hole
                fig_pie = go.Figure(data=[
                    go.Pie(
                        labels=list(class_stats.keys()),
                        values=list(class_stats.values()),
                        hole=0.4,
                        marker=dict(colors=px.colors.sequential.Blues_r),
                        textposition='inside',
                        textinfo='label+percent',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                        pull=[0.1 if v == max(class_stats.values()) else 0 for v in class_stats.values()]
                    )
                ])
                fig_pie.update_layout(
                    title="Pie Chart - Distribution %",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="pie_dist")
            
            # Sunburst chart
            st.subheader("Hierarchical View")
            sunburst_fig = create_defect_sunburst(formatted['defect_details'])
            if sunburst_fig:
                st.plotly_chart(sunburst_fig, use_container_width=True, key="sunburst")
            
            # Confidence Gauge
            st.subheader("Overall Confidence Score")
            gauge_fig = create_confidence_gauge(formatted['summary']['average_confidence'])
            st.plotly_chart(gauge_fig, use_container_width=True, key="gauge")
        
        with viz_tab2:
            st.subheader("Spatial Distribution of Defects")
            
            # Get image dimensions
            h, w = result['images']['test'].shape[:2]
            
            # Location heatmap
            st.markdown("**Defect Location Heatmap**")
            st.caption("Shows concentration of defects across the PCB surface")
            heatmap_fig = create_defect_location_heatmap(formatted['defect_details'], (h, w))
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True, key="heatmap")
            
            # Scatter map
            st.markdown("**Interactive Defect Map**")
            st.caption("Click on points to see defect details. Size indicates defect area, color indicates confidence.")
            scatter_map_fig = create_defect_scatter_map(formatted['defect_details'], (h, w))
            if scatter_map_fig:
                st.plotly_chart(scatter_map_fig, use_container_width=True, key="scatter_map")
        
        with viz_tab3:
            st.subheader("Size and Confidence Analysis")
            
            col_size1, col_size2 = st.columns(2)
            
            with col_size1:
                # Scatter: Area vs Confidence
                st.markdown("**Area vs Confidence Correlation**")
                scatter_fig = plot_defect_scatter(formatted['defect_details'])
                if scatter_fig:
                    st.plotly_chart(scatter_fig, use_container_width=True, key="scatter_area_conf")
            
            with col_size2:
                # Area histogram
                st.markdown("**Size Distribution**")
                hist_fig = plot_defect_area_hist(formatted['defect_details'])
                if hist_fig:
                    st.plotly_chart(hist_fig, use_container_width=True, key="hist_area")
            
            # Violin plot for size
            st.markdown("**Size Distribution by Defect Type**")
            st.caption("Violin plots show the full distribution shape with quartiles")
            violin_fig = create_size_vs_type_violin(formatted['defect_details'])
            if violin_fig:
                st.plotly_chart(violin_fig, use_container_width=True, key="violin_size")
        
        with viz_tab4:
            st.subheader("Comparative Analysis")
            
            # Box plot for confidence
            st.markdown("**Confidence Comparison Across Types**")
            st.caption("Box plots show median, quartiles, and outliers for each defect type")
            box_fig = create_confidence_box_plot(formatted['defect_details'])
            if box_fig:
                st.plotly_chart(box_fig, use_container_width=True, key="box_conf")
            
            # Detection order simulation
            st.markdown("**Detection Summary (Ordered by Confidence)**")
            st.caption("Shows how defects were detected in order of confidence")
            timeline_fig = create_defect_timeline_simulation(formatted['defect_details'])
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True, key="timeline")
            
            # Statistical summary table
            st.markdown("**Statistical Summary by Type**")
            stats_data = []
            for dtype in set([d['type'] for d in formatted['defect_details']]):
                type_defects = [d for d in formatted['defect_details'] if d['type'] == dtype]
                stats_data.append({
                    'Defect Type': dtype.title(),
                    'Count': len(type_defects),
                    'Avg Confidence': f"{np.mean([d['confidence (%)'] for d in type_defects]):.1f}%",
                    'Avg Size': f"{np.mean([d['area (px²)'] for d in type_defects]):.0f} px²",
                    'Min Size': f"{min([d['area (px²)'] for d in type_defects])} px²",
                    'Max Size': f"{max([d['area (px²)'] for d in type_defects])} px²"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        with viz_tab5:
            st.subheader("Advanced 3D Visualization")
            st.caption("3D view showing X position, Y position, and Confidence level")
            
            # 3D scatter
            scatter_3d_fig = create_3d_defect_scatter(formatted['defect_details'])
            if scatter_3d_fig:
                st.plotly_chart(scatter_3d_fig, use_container_width=True, key="scatter_3d")
            
            # Correlation matrix
            st.markdown("**Feature Correlation Matrix**")
            
            # Prepare data for correlation
            corr_data = {
                'X Position': [],
                'Y Position': [],
                'Confidence': [],
                'Area': []
            }
            
            for d in formatted['defect_details']:
                loc_str = d['location'].strip('()').split(',')
                x, y = int(loc_str[0]), int(loc_str[1])
                corr_data['X Position'].append(x)
                corr_data['Y Position'].append(y)
                corr_data['Confidence'].append(d['confidence (%)'])
                corr_data['Area'].append(d['area (px²)'])
            
            corr_df = pd.DataFrame(corr_data)
            correlation_matrix = corr_df.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 12},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
            ))
            
            fig_corr.update_layout(
                title="Feature Correlation Heatmap",
                height=450,
                xaxis={'side': 'bottom'}
            )
            
            st.plotly_chart(fig_corr, use_container_width=True, key="corr_matrix")
            
            # Key insights
            st.markdown("**🔍 Key Insights:**")
            
            insights = []
            
            # Most common defect
            most_common = max(class_stats, key=class_stats.get)
            insights.append(f"• Most common defect: **{most_common.title()}** ({class_stats[most_common]} occurrences)")
            
            # Average confidence
            avg_conf = np.mean([d['confidence (%)'] for d in formatted['defect_details']])
            insights.append(f"• Average detection confidence: **{avg_conf:.1f}%**")
            
            # Size range
            sizes = [d['area (px²)'] for d in formatted['defect_details']]
            insights.append(f"• Defect size range: **{min(sizes)}-{max(sizes)} px²**")
            
            # Confidence reliability
            high_conf = len([d for d in formatted['defect_details'] if d['confidence (%)'] >= 90])
            insights.append(f"• High confidence detections (≥90%): **{high_conf}/{len(formatted['defect_details'])}** ({high_conf/len(formatted['defect_details'])*100:.1f}%)")
            
            for insight in insights:
                st.markdown(insight)
    
    else:
        st.success("🎉 Excellent! No defects detected - This PCB appears to be defect-free!")
    
    # ==================== END OF ENHANCED SECTION ====================
    
    # Defect details table
    if show_details and formatted['defect_details']:
        st.header("🔍 Detailed Defect Information")
        details_df = pd.DataFrame(formatted['defect_details'])
        st.dataframe(details_df, use_container_width=True, height=400)
    
    # Processing steps visualization
    if show_intermediate:
        st.header("🖼️ Processing Pipeline Visualization")
        
        st.info("ℹ️ **Note:** Images displayed at 320×320 px for optimal viewing. Full-resolution images available in downloads.")
        
        tab_aligned, tab_diff, tab_mask = st.tabs(["Aligned Image", "Difference Map", "Defect Mask"])
        
        with tab_aligned:
            col_a1, col_a2 = st.columns([2, 1])
            with col_a1:
                # Resize aligned image for display (320px max)
                aligned_pil = convert_cv_to_pil(result['images']['aligned'])
                aligned_display = resize_for_display(aligned_pil, max_size=320)
                st.image(aligned_display, caption="Aligned Test Image", width=320)
                h, w = result['images']['aligned'].shape[:2]
                st.caption(f"Display: 320×320 px | Original: {w} × {h} px")
            with col_a2:
                st.metric("Feature Matches", result['alignment_info']['num_matches'])
                st.metric("Inliers", result['alignment_info']['num_inliers'])
                
                # Alignment quality indicator
                quality = result['alignment_info']['num_inliers'] / result['alignment_info']['num_matches'] * 100 if result['alignment_info']['num_matches'] > 0 else 0
                if quality >= 80:
                    st.success(f"Quality: {quality:.1f}% ✓")
                elif quality >= 60:
                    st.warning(f"Quality: {quality:.1f}% ⚠️")
                else:
                    st.error(f"Quality: {quality:.1f}% ✗")
        
        with tab_diff:
            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                # Resize difference map for display (320px max)
                diff_pil = convert_cv_to_pil(result['images']['difference_map'])
                diff_display = resize_for_display(diff_pil, max_size=320)
                st.image(diff_display, caption="Difference Map (Highlights Changes)", width=320)
                h, w = result['images']['difference_map'].shape[:2]
                st.caption(f"Display: 320×320 px | Original: {w} × {h} px")
            with col_d2:
                st.info("🔥 **Brighter areas** indicate larger differences from the template")
                st.info("🔵 **Dark areas** show matching regions")
        
        with tab_mask:
            col_m1, col_m2 = st.columns([2, 1])
            with col_m1:
                # Resize mask for display (320px max)
                mask_pil = convert_cv_to_pil(result['images']['mask'])
                mask_display = resize_for_display(mask_pil, max_size=320)
                st.image(mask_display, caption="Binary Defect Mask", width=320)
                h, w = result['images']['mask'].shape[:2]
                st.caption(f"Display: 320×320 px | Original: {w} × {h} px")
            with col_m2:
                st.info("⬜ **White regions** indicate detected defects")
                st.info("⬛ **Black areas** show defect-free regions")
                
                # Calculate defect coverage
                if len(result['images']['mask'].shape) == 2:
                    total_pixels = result['images']['mask'].shape[0] * result['images']['mask'].shape[1]
                    defect_pixels = np.sum(result['images']['mask'] > 0)
                    coverage = (defect_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                    st.metric("Defect Coverage", f"{coverage:.2f}%")
    
    # ================= INTERACTIVE ANALYSIS =================
    st.header("⚗️ Interactive Analysis")

    # Overlay viewer
    with st.expander("🔀 Overlay Viewer (blend images)"):
        images_map = {
            'Template': result['images']['template'],
            'Aligned': result['images']['aligned'],
            'Test': result['images']['test'],
            'Annotated': result['images']['annotated'],
            'Difference Map': result['images']['difference_map'],
            'Mask': result['images']['mask']
        }

        left_sel = st.selectbox("Left image", list(images_map.keys()), index=0, key='overlay_left')
        right_sel = st.selectbox("Right image", list(images_map.keys()), index=3, key='overlay_right')
        alpha = st.slider("Blend (Left → Right)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        left_img = convert_cv_to_pil(images_map[left_sel])
        right_img = convert_cv_to_pil(images_map[right_sel])

        try:
            # Ensure both images are same size/mode for blending
            left_rgba = left_img.convert('RGBA')
            right_rgba = right_img.convert('RGBA').resize(left_rgba.size)
            blended = Image.blend(left_rgba, right_rgba, alpha)
            # Resize blended image for display
            blended_display = resize_for_display(blended, max_size=700)
            st.image(blended_display, caption=f"{left_sel} blended with {right_sel} (alpha={alpha:.2f})", width=700)
        except Exception as e:
            st.error(f"Overlay failed: {e}")

    # Interactive defect filtering and highlighting
    if formatted['defect_details']:
        with st.expander("🧭 Defect Filter & Map"):
            defects = formatted['defect_details']

            areas = [d['area (px²)'] for d in defects]
            confs = [d['confidence (%)'] for d in defects]
            classes = sorted(list(set([d['type'] for d in defects])))

            min_area = int(min(areas)) if areas else 0
            max_area = int(max(areas)) if areas else 1
            area_range = st.slider("Area range (px²)", min_value=min_area, max_value=max_area, value=(min_area, max_area))

            conf_min = int(min(confs)) if confs else 0
            conf_max = int(max(confs)) if confs else 100
            conf_range = st.slider("Confidence range (%)", min_value=0, max_value=100, value=(conf_min, conf_max))

            class_sel = st.multiselect("Filter by class", classes, default=classes)

            # Apply filters
            filtered = [d for d in defects if (area_range[0] <= d['area (px²)'] <= area_range[1]) and (conf_range[0] <= d['confidence (%)'] <= conf_range[1]) and (d['type'] in class_sel)]

            st.write(f"Showing {len(filtered)} / {len(defects)} defects")

            # Highlight filtered defects on annotated image
            try:
                annotated_cv = result['images']['annotated'].copy()
                # draw highlight boxes
                for d in filtered:
                    bbox = d.get('bbox', None)
                    if bbox:
                        x, y, w, h = bbox
                        cv2.rectangle(annotated_cv, (x, y), (x + w, y + h), (0, 255, 255), 3)

                highlighted_pil = convert_cv_to_pil(annotated_cv)
                # Resize for display
                highlighted_display = resize_for_display(highlighted_pil, max_size=700)
                st.image(highlighted_display, caption='Annotated image (filtered highlights)', width=700)
            except Exception as e:
                st.error(f"Failed to draw highlights: {e}")

    # Interactive charts
    with st.expander("📈 Interactive Charts"):
        # Scatter (area vs confidence)
        scatter_fig = plot_defect_scatter(formatted['defect_details'])
        st.plotly_chart(scatter_fig, use_container_width=True, key='plotly_scatter_defects')

        # Area histogram
        hist_fig = plot_defect_area_hist(formatted['defect_details'])
        st.plotly_chart(hist_fig, use_container_width=True, key='plotly_hist_area')

        # Ground truth comparison (optional)
        st.markdown("**Optional:** Upload a ground-truth label file (text format: x1 y1 x2 y2 class_id per line) to compute detection metrics.")
        gt_file = st.file_uploader("Upload ground truth labels (optional)", type=['txt'], key='gt_upload')
        if gt_file is not None:
            try:
                # Save uploaded file to temp and run comparison
                tmp_path = os.path.join('temp', f"gt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(tmp_path, 'wb') as fh:
                    fh.write(gt_file.getvalue())

                comp = compare_with_ground_truth(result['classifications'], tmp_path, backend.config)  # FIXED: Line 1764 - use backend parameter
                st.write("### Ground truth comparison")
                st.json(comp)
            except Exception as e:
                st.error(f"Ground truth comparison failed: {e}")
    
    # Final annotated result
    st.header("✨ Final Inspection Result")
    
    st.info("ℹ️ **Display Size:** Image shown at 450px for optimal viewing. Full-resolution version available in downloads.")
    
    # Create layout with image and legend side by side
    result_col1, result_col2 = st.columns([3, 1])
    
    with result_col1:
        # Resize annotated image for display (450px max) - keep original in memory
        annotated_pil = convert_cv_to_pil(result['images']['annotated'])
        annotated_display = resize_for_display(annotated_pil, max_size=450)
        st.image(annotated_display, caption="Annotated PCB with Detected & Classified Defects", width=450)
        
        # Show dimensions
        h, w = result['images']['annotated'].shape[:2]
        st.caption(f"📐 Display: 450×450 px | Original: {w} × {h} px")
    
    with result_col2:
        st.markdown("### 🎨 Color Legend")
        
        # Color-coded legend with better formatting
        st.markdown("""
        **Defect Classification:**
        
        🔵 **Mousebite**  
        Small perforations
        
        🟢 **Open Circuit**  
        Broken traces
        
        🔴 **Short Circuit**  
        Unintended connections
        
        🟡 **Spur**  
        Copper extensions
        
        🟣 **Copper Issue**  
        Missing/excess copper
        
        🟠 **Pin-hole**  
        Microscopic holes
        """)
        
        st.markdown("---")
        
        # Add quick stats
        if formatted['defect_details']:
            st.markdown("### 📊 Quick Stats")
            st.metric("Total Defects", formatted['summary']['total_defects'])
            st.metric("Avg Confidence", formatted['summary']['average_confidence'])
    
    # Download section
    st.header("💾 Export & Download Options")
    
    st.info("ℹ️ **All downloads are in full original resolution**")
    
    # Create tabs for different export options
    export_tab1, export_tab2, export_tab3, export_tab4 = st.tabs([
        "📥 Images", "📊 Data Files", "📄 Reports", "📦 Complete Package"
    ])
    
    with export_tab1:
        st.subheader("Download Images")
        
        img_col1, img_col2, img_col3 = st.columns(3)
        
        with img_col1:
            # Use ORIGINAL (not resized) image for download
            buffered = BytesIO()
            annotated_pil.save(buffered, format="PNG")
            file_size = len(buffered.getvalue()) / 1024
            st.download_button(
                label="📥 Annotated Image",
                data=buffered.getvalue(),
                file_name=f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
                key="download_annotated"
            )
            st.caption(f"Size: {file_size:.1f} KB")
        
        with img_col2:
            diff_buffered = BytesIO()
            diff_pil = convert_cv_to_pil(result['images']['difference_map'])
            diff_pil.save(diff_buffered, format="PNG")
            file_size = len(diff_buffered.getvalue()) / 1024
            st.download_button(
                label="📥 Difference Map",
                data=diff_buffered.getvalue(),
                file_name=f"difference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
                key="download_diff"
            )
            st.caption(f"Size: {file_size:.1f} KB")
        
        with img_col3:
            mask_buffered = BytesIO()
            mask_pil = convert_cv_to_pil(result['images']['mask'])
            mask_pil.save(mask_buffered, format="PNG")
            file_size = len(mask_buffered.getvalue()) / 1024
            st.download_button(
                label="📥 Defect Mask",
                data=mask_buffered.getvalue(),
                file_name=f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
                key="download_mask"
            )
            st.caption(f"Size: {file_size:.1f} KB")
    
    with export_tab2:
        st.subheader("Download Data Files")
        
        data_col1, data_col2 = st.columns(2)
        
        with data_col1:
            if formatted['defect_details']:
                csv_data = pd.DataFrame(formatted['defect_details']).to_csv(index=False)
                st.download_button(
                    label="📥 Defects Data (CSV)",
                    data=csv_data,
                    file_name=f"defects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_csv"
                )
            else:
                st.info("No defects to export")
        
        with data_col2:
            json_data = {
                'timestamp': datetime.now().isoformat(),
                'summary': formatted['summary'],
                'defects': formatted['defect_details'],
                'class_distribution': formatted['class_distribution']
            }
            
            st.download_button(
                label="📥 Full Results (JSON)",
                data=json.dumps(json_data, indent=2),
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                key="download_json"
            )
    
    with export_tab3:
        st.subheader("Download Reports")
        
        report_col1, report_col2 = st.columns(2)
        
        with report_col1:
            st.write("**PDF Inspection Report**")
            
            # Handle auto-generate option
            should_generate = auto_report or st.button("📄 Generate PDF", use_container_width=True, key="gen_pdf")

            if should_generate:
                with st.spinner("Creating comprehensive PDF with images..."):
                    try:
                        # Make sure temp directory exists
                        os.makedirs('temp', exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        pdf_filename = os.path.join('temp', f"report_{timestamp}.pdf")

                        # Pass images directly to PDF generator
                        result_images = {
                            'annotated': result['images']['annotated'],
                            'difference_map': result['images']['difference_map'],
                            'mask': result['images']['mask'],
                            'test': result['images']['test'],
                            'template': result['images']['template'],
                            'aligned': result['images']['aligned']
                        }

                        # Generate charts for PDF
                        chart_images = {}
                        if formatted['defect_details']:
                            # 1. Defect distribution chart
                            plt.figure(figsize=(8, 5))
                            plt.bar(formatted['class_distribution'].keys(), 
                                    formatted['class_distribution'].values(),
                                    color='steelblue')
                            plt.title('Defect Type Distribution')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            dist_chart_path = os.path.join('temp', f'dist_{timestamp}.png')
                            plt.savefig(dist_chart_path)
                            plt.close()
                            chart_images['distribution'] = dist_chart_path

                            # 2. Confidence histogram
                            confidences = [d['confidence (%)'] for d in formatted['defect_details']]
                            plt.figure(figsize=(8, 5))
                            plt.hist(confidences, bins=20, color='lightcoral')
                            plt.title('Confidence Score Distribution')
                            plt.xlabel('Confidence (%)')
                            plt.ylabel('Frequency')
                            plt.tight_layout()
                            conf_chart_path = os.path.join('temp', f'conf_{timestamp}.png')
                            plt.savefig(conf_chart_path)
                            plt.close()
                            chart_images['confidence'] = conf_chart_path

                        try:
                            # Generate PDF with all content
                            create_pdf_report(
                                formatted_data=formatted,
                                output_path=pdf_filename,
                                class_names=backend.class_names,
                                result_images=result_images,
                                chart_paths=chart_images if chart_images else None
                            )

                            # Read generated PDF into session state
                            with open(pdf_filename, 'rb') as pdf_file:
                                pdf_data = pdf_file.read()
                                if not pdf_data:
                                    raise ValueError("Generated PDF is empty")
                                st.session_state.pdf_data = pdf_data
                                st.success("✅ PDF with images generated successfully!")

                        except Exception as pdf_err:
                            st.error(f"PDF generation failed: {str(pdf_err)}")
                            print(f"Detailed PDF error: {str(pdf_err)}")
                            st.session_state.pdf_data = None
                        
                        finally:
                            # Clean up temp files
                            try:
                                if os.path.exists(pdf_filename):
                                    os.remove(pdf_filename)
                                for chart_path in chart_images.values():
                                    if os.path.exists(chart_path):
                                        os.remove(chart_path)
                            except Exception as cleanup_err:
                                print(f"Cleanup warning: {str(cleanup_err)}")

                    except Exception as e:
                        st.error(f"Error preparing PDF content: {str(e)}")
                        print(f"Content preparation error: {str(e)}")
                        st.session_state.pdf_data = None

            # Show download button if we have PDF data
            if st.session_state.pdf_data is not None:
                try:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=st.session_state.pdf_data,
                        file_name=f"PCB_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf"
                    )
                except Exception as e:
                    st.error(f"Download setup failed: {str(e)}")
                    st.session_state.pdf_data = None
        
        with report_col2:
            st.write("**Text Inspection Log**")
            log_text = generate_text_log(result, formatted)
            
            st.download_button(
                label="📥 Download Text Log",
                data=log_text,
                file_name=f"inspection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_log"
            )
    
    with export_tab4:
        st.subheader("Download Complete Package")
        st.info("📦 All images, data, reports, and charts in a single ZIP file")

        # Auto-generate complete package if not already in session
        if st.session_state.zip_data is None:
            try:
                with st.spinner("Creating comprehensive package..."):
                    zip_buffer = create_complete_package(result, formatted, backend, annotated_pil)
                    st.session_state.zip_data = zip_buffer.getvalue()
                    st.success("✅ Package ready for download!")
            except Exception as e:
                st.error(f"Error preparing package: {str(e)}")
                print(f"Package creation error: {str(e)}")
                import traceback
                traceback.print_exc()

        # Regenerate package button
        if st.button("🔄 Regenerate Package", type="secondary", use_container_width=True, key="regen_zip"):
            try:
                with st.spinner("Regenerating package..."):
                    zip_buffer = create_complete_package(result, formatted, backend, annotated_pil)
                    st.session_state.zip_data = zip_buffer.getvalue()
                    st.success("✅ Package updated!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                traceback.print_exc()

        # Download button
        if st.session_state.zip_data is not None:
            st.download_button(
                label="📥 Download ZIP Package",
                data=st.session_state.zip_data,
                file_name=f"PCB_Package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_zip"
            )
            
            with st.expander("📋 Package Contents"):
                st.markdown("""
                **This package includes:**
                - 📁 `/images/` - 6 inspection images (full resolution)
                - 📁 `/data/` - 5 CSV and JSON files
                - 📁 `/charts/` - 6 visualization charts
                - 📁 `/reports/` - Text log and PDF report
                - 📄 `README.txt` - Package documentation
                
                **Total: 20 files** organized in 4 folders
                """)


# ==================== HISTORY MODE ====================

def history_mode():
    """Inspection history mode UI"""
    
    st.header("📜 Inspection History & Analytics")
    
    if not st.session_state.inspection_history:
        st.markdown("""
        <div class="info-box">
        ℹ️ <b>No History Yet</b><br>
        Your inspection history will appear here after you perform some inspections.
        Start by switching to "Single Inspection" or "Batch Processing" mode.
        </div>
        """, unsafe_allow_html=True)
        return
    
    history_df = pd.DataFrame(st.session_state.inspection_history)
    
    st.subheader("📊 Overall Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Inspections", len(history_df))
    
    with col2:
        successful = history_df['success'].sum()
        st.metric("Successful", successful)
    
    with col3:
        total_defects = history_df['num_defects'].sum()
        st.metric("Total Defects", total_defects)
    
    with col4:
        avg_defects = history_df['num_defects'].mean()
        st.metric("Avg Defects/Inspection", f"{avg_defects:.1f}")
    
    with col5:
        avg_time = history_df['processing_time'].mean()
        st.metric("Avg Time", f"{avg_time:.2f}s")
    
    st.subheader("📈 Trends & Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Timeline", "Statistics", "Raw Data"])
    
    with tab1:
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=list(range(len(history_df))),
            y=history_df['num_defects'],
            mode='lines+markers',
            name='Defects',
            line=dict(color='steelblue', width=3),
            marker=dict(size=10, symbol='circle'),
            fill='tozeroy',
            hovertemplate='<b>Inspection #%{x}</b><br>Defects: %{y}<extra></extra>'
        ))
        
        fig_timeline.update_layout(
            title="Defect Count Over Time",
            xaxis_title="Inspection Number",
            yaxis_title="Number of Defects",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab2:
        fig_time = go.Figure(data=[
            go.Histogram(
                x=history_df['processing_time'],
                nbinsx=20,
                marker_color='lightcoral',
                hovertemplate='Time: %{x:.2f}s<br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig_time.update_layout(
            title="Processing Time Distribution",
            xaxis_title="Processing Time (seconds)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    with tab3:
        st.dataframe(
            history_df.style.background_gradient(subset=['num_defects'], cmap='YlOrRd'),
            use_container_width=True
        )
    
    st.subheader("🛠️ Actions")
    
    action_col1, action_col2 = st.columns(2)
    
    with action_col1:
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                st.session_state.inspection_history = []
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm")
    
    with action_col2:
        csv_history = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Export History (CSV)",
            data=csv_history,
            file_name=f"inspection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    main()