"""
Advanced settings and configuration
"""

import streamlit as st


def advanced_settings_ui(config: dict):
    """UI for advanced settings"""
    
    st.header("⚙️ Advanced Settings")
    
    with st.expander("🔧 Detection Parameters", expanded=False):
        st.subheader("Image Alignment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alignment_method = st.selectbox(
                "Feature Detection Method",
                ["orb", "sift"],
                index=0 if config['alignment']['method'] == 'orb' else 1
            )
        
        with col2:
            max_features = st.slider(
                "Max Features",
                1000, 10000, 
                config['alignment']['max_features'],
                step=500
            )
        
        st.subheader("Defect Detection")
        
        col3, col4 = st.columns(2)
        
        with col3:
            min_area = st.slider(
                "Min Defect Area (pixels²)",
                10, 500,
                config['contours']['min_area'],
                step=10
            )
        
        with col4:
            max_area = st.slider(
                "Max Defect Area (pixels²)",
                1000, 100000,
                config['contours']['max_area'],
                step=1000
            )
        
        threshold_method = st.selectbox(
            "Thresholding Method",
            ["otsu", "adaptive"],
            index=0 if config['thresholding']['method'] == 'otsu' else 1
        )
    
    with st.expander("🎨 Visualization Options", expanded=False):
        
        col5, col6 = st.columns(2)
        
        with col5:
            show_alignment = st.checkbox("Show alignment info", value=True)
            show_difference = st.checkbox("Show difference map", value=True)
        
        with col6:
            show_mask = st.checkbox("Show defect mask", value=True)
            show_confidence = st.checkbox("Show confidence scores", value=True)
        
        annotation_style = st.radio(
            "Annotation Style",
            ["Bounding boxes", "Filled regions", "Contours"],
            horizontal=True
        )
    
    with st.expander("📊 Performance Options", expanded=False):
        
        enable_gpu = st.checkbox(
            "Enable GPU Acceleration" if st.session_state.get('has_gpu', False) else "GPU Not Available",
            value=st.session_state.get('has_gpu', False),
            disabled=not st.session_state.get('has_gpu', False)
        )
        
        batch_size = st.slider(
            "Batch Size (for multiple images)",
            1, 32, 8
        )
        
        cache_results = st.checkbox("Cache processing results", value=True)
    
    # Save settings button
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        # Update config
        config['alignment']['method'] = alignment_method
        config['alignment']['max_features'] = max_features
        config['contours']['min_area'] = min_area
        config['contours']['max_area'] = max_area
        config['thresholding']['method'] = threshold_method
        
        st.success("✓ Settings saved successfully!")
    
    return config