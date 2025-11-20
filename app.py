import streamlit as st
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import timm
import cv2
import numpy as np
import io
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import the PDF generator module
from pdf_generator import generate_pdf_report

# --- PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="CircuitGuard PCB Detection",
    page_icon="🔍"
)

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load and cache the EfficientNet model."""
    model = timm.create_model('efficientnet_b4', pretrained=False)
    num_features = model.classifier.in_features
    class_names = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
    model.classifier = nn.Linear(num_features, len(class_names))
    model_path = 'models/best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, class_names

model, class_names = load_model()

# Pre-define transforms
inference_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. HELPER FUNCTIONS ---
def clear_results():
    """Clears previous analysis results from the session state."""
    keys_to_delete = [k for k in st.session_state.keys() if k not in ['template', 'test']]
    for key in keys_to_delete:
        del st.session_state[key]

# --- Chart Generation Functions ---
def create_bar_chart(df):
    """Create bar chart with improved styling."""
    fig, ax = plt.subplots(figsize=(6, 5.7), dpi=100)
    defect_counts = df['Class'].value_counts()
    defect_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title('Defect Type Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Defect Class', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_pie_chart(df):
    """Create pie chart with improved styling."""
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=100)
    defect_counts = df['Class'].value_counts()
    colors_pie = plt.cm.Set3(range(len(defect_counts)))
    ax.pie(defect_counts, labels=defect_counts.index, autopct='%1.1f%%', 
           startangle=90, colors=colors_pie, textprops={'fontsize': 10})
    ax.set_title('Defect Proportions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_heatmap(image_shape, df):
    """Create heatmap with optimized performance."""
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    if not df.empty:
        for _, row in df.iterrows():
            pos_str = row['Position'].strip('()')
            x, y = map(int, pos_str.split(','))
            w, h = map(int, row['Size'].split('x'))
            y_end = min(y + h, image_shape[0])
            x_end = min(x + w, image_shape[1])
            heatmap[y:y_end, x:x_end] += 1.0
    
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=100)
    sns.heatmap(heatmap, ax=ax, cbar=True, cmap="hot", cbar_kws={'label': 'Defect Density'})
    ax.set_title('Defect Location Heatmap', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig

def create_scatter_plot(df):
    """Create scatter plot with optimized data processing."""
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=100)
    if not df.empty:
        confidence_values = df['Confidence'].str.replace('%', '').astype(float)
        defect_areas = df['Size'].apply(lambda s: np.prod([int(x) for x in s.split('x')]))
        
        scatter_data = pd.DataFrame({
            'Defect_Area': defect_areas,
            'Confidence_Value': confidence_values,
            'Class': df['Class']
        })
        
        sns.scatterplot(data=scatter_data, x='Defect_Area', y='Confidence_Value', 
                       hue='Class', ax=ax, s=100, alpha=0.7, edgecolor='black')
        ax.set_title('Confidence vs. Defect Area', fontsize=14, fontweight='bold')
        ax.set_xlabel('Defect Area (pixels²)', fontsize=11)
        ax.set_ylabel('Confidence (%)', fontsize=11)
        ax.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    else:
        ax.set_title('Confidence vs. Defect Area', fontsize=14, fontweight='bold')
        ax.set_xlabel('Defect Area (pixels²)', fontsize=11)
        ax.set_ylabel('Confidence (%)', fontsize=11)
    plt.tight_layout()
    return fig

# --- Image/Figure to Bytes Helper ---
def get_bytes_from_fig(fig):
    """Converts a Matplotlib figure to PNG bytes efficiently."""
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def get_bytes_from_image(image_array, format='PNG'):
    """Converts a NumPy image array to bytes efficiently."""
    if image_array.ndim == 2:
        pil_img = Image.fromarray(image_array)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format=format, optimize=True)
    buf.seek(0)
    return buf

# --- 3. USER INTERFACE ---
st.title("🔍 CircuitGuard - Advanced PCB Defect Detection")
st.write("Upload a template and test image, adjust the parameters, and find manufacturing defects.")

# Status indicator
col_status1, col_status2 = st.columns([3, 1])
with col_status1:
    st.success("✓ AI Model loaded successfully!")
with col_status2:
    st.metric("Model", "EfficientNet-B4", delta="v2.1")

st.markdown("---")

# Image upload section
col1, col2 = st.columns(2)
with col1:
    st.header("📁 Template Image")
    st.file_uploader(
        "Upload reference PCB image", 
        type=["jpg", "png", "jpeg"], 
        key="template", 
        on_change=clear_results,
        help="Upload the defect-free reference template"
    )
    if st.session_state.get('template'):
        st.image(st.session_state['template'], caption="Template Preview", width=300)

with col2:
    st.header("📁 Test Image")
    st.file_uploader(
        "Upload test PCB image", 
        type=["jpg", "png", "jpeg"], 
        key="test", 
        on_change=clear_results,
        help="Upload the PCB to be inspected"
    )
    if st.session_state.get('test'):
        st.image(st.session_state['test'], caption="Test Preview", width=300)

st.markdown("---")

# Detection parameters in sidebar
st.sidebar.header("⚙️ Detection Parameters")
st.sidebar.markdown("Adjust these settings to fine-tune detection sensitivity")

diff_threshold = st.sidebar.slider(
    "Difference Threshold", 
    min_value=0, 
    max_value=100, 
    value=30,
    help="Pixel intensity difference sensitivity"
)

min_area = st.sidebar.slider(
    "Minimum Defect Area (pixels)", 
    min_value=10, 
    max_value=500, 
    value=50,
    help="Smallest detectable defect size"
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.50, 
    step=0.05,
    help="Minimum AI confidence required"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Lower thresholds increase sensitivity but may produce more false positives.")

# --- 4. ANALYSIS PIPELINE ---
analyze_button = st.button("🔬 Analyze for Defects", type="primary", use_container_width=True)

if analyze_button:
    if st.session_state.get('template') and st.session_state.get('test'):
        with st.spinner('🔄 Analyzing... This may take a few moments.'):
            template_file = st.session_state.template
            test_file = st.session_state.test
            
            # Load and process images
            template_pil = Image.open(template_file).convert('RGB')
            test_pil = Image.open(test_file).convert('RGB')
            template_cv = np.array(template_pil)
            test_cv = np.array(test_pil)
            height, width, _ = template_cv.shape
            test_resized = cv2.resize(test_cv, (width, height))

            # Differential analysis
            template_gray = cv2.cvtColor(template_cv, cv2.COLOR_RGB2GRAY)
            test_gray = cv2.cvtColor(test_resized, cv2.COLOR_RGB2GRAY)
            difference = cv2.absdiff(template_gray, test_gray)
            _, threshold_mask = cv2.threshold(difference, diff_threshold, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3,3), np.uint8)
            cleaned_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Defect detection and classification
            output_image = test_resized.copy()
            defect_count = 0
            defect_log_list = []

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, contour in enumerate(contours):
                progress_bar.progress((idx + 1) / len(contours))
                status_text.text(f"Processing contour {idx + 1} of {len(contours)}...")
                
                if cv2.contourArea(contour) < min_area: 
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                roi_pil = Image.fromarray(test_resized[y:y+h, x:x+w])
                roi_tensor = inference_transforms(roi_pil).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(roi_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    if confidence.item() < confidence_threshold: 
                        continue
                        
                    prediction = class_names[predicted_idx.item()]
                
                defect_count += 1
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{prediction} ({confidence.item()*100:.1f}%)"
                cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                defect_log_list.append({
                    "Defect #": defect_count, 
                    "Class": prediction, 
                    "Confidence": f"{confidence.item()*100:.2f}%", 
                    "Position": f"({x}, {y})", 
                    "Size": f"{w}x{h}"
                })
            
            progress_bar.empty()
            status_text.empty()
            
            df_log = pd.DataFrame(defect_log_list)
            
            # Generate assets for PDF
            result_image_buf = get_bytes_from_image(output_image)
            diff_image_buf = get_bytes_from_image(difference)
            mask_image_buf = get_bytes_from_image(cleaned_mask)
            
            bar_chart_buf = io.BytesIO()
            pie_chart_buf = io.BytesIO()
            heatmap_buf = io.BytesIO()
            scatter_buf = io.BytesIO()

            if not df_log.empty:
                bar_chart_buf = get_bytes_from_fig(create_bar_chart(df_log))
                pie_chart_buf = get_bytes_from_fig(create_pie_chart(df_log))
                heatmap_buf = get_bytes_from_fig(create_heatmap(test_resized.shape, df_log))
                scatter_buf = get_bytes_from_fig(create_scatter_plot(df_log))
            
            report_params = {
                'threshold': diff_threshold, 
                'min_area': min_area, 
                'confidence': confidence_threshold, 
                'defect_count': defect_count
            }
            
            # Generate PDF using the imported function
            with st.spinner('📄 Generating comprehensive PDF report...'):
                pdf_bytes = generate_pdf_report(
                    report_params, df_log, result_image_buf, 
                    bar_chart_buf, pie_chart_buf,
                    heatmap_buf, scatter_buf,
                    diff_image_buf, mask_image_buf
                )

            # Store results in session state
            st.session_state['analysis_complete'] = True
            st.session_state['result_image'] = output_image
            st.session_state['defect_count'] = defect_count
            st.session_state['defect_log_df'] = df_log
            st.session_state['diff_image'] = difference
            st.session_state['mask_image'] = cleaned_mask
            st.session_state['pdf_report'] = pdf_bytes
            
            st.success("✅ Analysis complete!")
            #st.balloons()
            
    else:
        st.warning("⚠️ Please upload both a template and a test image before analyzing.")

# --- 5. DISPLAY RESULTS ---
if st.session_state.get('analysis_complete'):
    st.markdown("---")
    
    # Results header with metric
    result_col1, result_col2, result_col3 = st.columns([2, 1, 1])
    with result_col1:
        st.header(f"📊 Analysis Complete")
    with result_col2:
        st.metric("Defects Found", st.session_state['defect_count'], 
                 delta="Critical" if st.session_state['defect_count'] > 5 else None,
                 delta_color="inverse")
    with result_col3:
        quality = "PASS" if st.session_state['defect_count'] == 0 else "FAIL" if st.session_state['defect_count'] > 5 else "REVIEW"
        st.metric("Quality Status", quality)
    
    st.markdown("---")
    
    # Main results display
    res_col1, res_col2 = st.columns([3, 2])
    with res_col1:
        st.subheader("🖼️ Final Annotated Result")
        st.image(st.session_state['result_image'], caption="Annotated Test Image with Defect Markers", use_container_width=True)
    
    with res_col2:
        st.subheader("📈 Defect Summary")
        if not st.session_state['defect_log_df'].empty:
            defect_counts = st.session_state['defect_log_df']['Class'].value_counts()
            for defect_type, count in defect_counts.items():
                st.metric(defect_type, count)
        else:
            st.info("✓ No defects found with the current parameters.")
    
    st.markdown("---")

    # Analysis breakdown
    st.subheader("🔍 Analysis Breakdown")
    col1_breakdown, col2_breakdown = st.columns(2)
    with col1_breakdown:
        st.image(st.session_state['diff_image'], caption="Raw Difference Image", use_container_width=True)
    with col2_breakdown:
        st.image(st.session_state['mask_image'], caption="Cleaned Defect Mask", use_container_width=True)
    
    st.markdown("---")

    # Charts (only if defects found)
    if not st.session_state['defect_log_df'].empty:
        st.subheader("📊 Defect Distribution Charts")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.pyplot(create_bar_chart(st.session_state['defect_log_df']))
        with chart_col2:
            st.pyplot(create_pie_chart(st.session_state['defect_log_df']))
        
        st.markdown("---")
        
        st.subheader("🗺️ Advanced Analysis")
        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            st.pyplot(create_heatmap(st.session_state['result_image'].shape, st.session_state['defect_log_df']))
        with chart_col4:
            st.pyplot(create_scatter_plot(st.session_state['defect_log_df']))
        
        st.markdown("---")
        
        st.subheader("📋 Defect Details")
        st.dataframe(st.session_state['defect_log_df'], use_container_width=True)

    st.markdown("---")

    # Download section
    st.subheader("💾 Download Results")
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        output_image_rgb = cv2.cvtColor(st.session_state['result_image'], cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(output_image_rgb)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        st.download_button(
            label="📸 Download Labeled Image (.png)", 
            data=buf.getvalue(), 
            file_name="analysis_result.png", 
            mime="image/png",
            use_container_width=True
        )
    
    with dl_col2:
        if not st.session_state['defect_log_df'].empty:
            csv = st.session_state['defect_log_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download Log (.csv)", 
                data=csv, 
                file_name="defect_log.csv", 
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("📊 Download Log (.csv)", disabled=True, use_container_width=True)

    with dl_col3:
        st.download_button(
            label="📄 Download Full Report (.pdf)",
            data=st.session_state['pdf_report'],
            file_name=f"CircuitGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p><b>CircuitGuard™</b> - AI-Powered PCB Defect Detection System v2.1</p>
        <p style='font-size: 12px;'>Powered by EfficientNet-B4 Deep Learning Architecture</p>
    </div>
    """,
    unsafe_allow_html=True
)