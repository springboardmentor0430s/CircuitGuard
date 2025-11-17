"""
Batch Upload UI Module for PCB Inspection
"""

import streamlit as st
import os
from PIL import Image
import zipfile
from datetime import datetime
import pandas as pd
import json
from io import BytesIO

from .batch_processor import BatchProcessor
from .backend import get_backend

def batch_upload_ui():
    """Render the batch upload interface"""
    st.header("📦 Batch PCB Inspection")
    
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
        
    st.markdown("""
    <div class="info-box">
    ℹ️ <b>Batch Processing Options:</b><br>
    1. Upload multiple template-test image pairs<br>
    2. Upload a ZIP file containing pairs in folders
    </div>
    """, unsafe_allow_html=True)
    
    upload_method = st.radio(
        "Select Upload Method:",
        ["Individual Pairs", "ZIP Upload"],
        help="Choose how you want to upload PCB images for batch processing"
    )
    
    if upload_method == "Individual Pairs":
        handle_individual_pairs()
    else:
        handle_zip_upload()


def handle_individual_pairs():
    """Handle individual image pair uploads"""
    st.subheader("Upload Image Pairs")
    
    # Container for image pairs
    if 'image_pairs' not in st.session_state:
        st.session_state.image_pairs = []
    
    # Add new pair
    st.markdown("### Add New Pair")
    col1, col2 = st.columns(2)
    
    with col1:
        template = st.file_uploader(
            "Upload Template (Reference)",
            type=['jpg', 'jpeg', 'png'],
            key=f"template_{len(st.session_state.image_pairs)}"
        )
    
    with col2:
        test = st.file_uploader(
            "Upload Test Image",
            type=['jpg', 'jpeg', 'png'],
            key=f"test_{len(st.session_state.image_pairs)}"
        )
    
    if template and test:
        if st.button("Add Pair to Batch", type="primary"):
            st.session_state.image_pairs.append({
                'template': template,
                'test': test,
                'template_name': template.name,
                'test_name': test.name
            })
            st.success(f"✅ Added pair: {template.name} - {test.name}")
            st.rerun()
    
    # Show current pairs
    if st.session_state.image_pairs:
        st.markdown("### Current Batch")
        for idx, pair in enumerate(st.session_state.image_pairs):
            cols = st.columns([3, 3, 1])
            with cols[0]:
                st.text(f"Template: {pair['template_name']}")
            with cols[1]:
                st.text(f"Test: {pair['test_name']}")
            with cols[2]:
                if st.button("❌", key=f"remove_{idx}"):
                    st.session_state.image_pairs.pop(idx)
                    st.rerun()
        
        # Process button
        process_individual_pairs()


def handle_zip_upload():
    """Handle ZIP file upload containing image pairs"""
    st.subheader("Upload ZIP File")
    
    st.info("""
    ZIP file should contain folders with template-test image pairs.
    Example structure:
    ```
    batch.zip
    ├── pair_1/
    │   ├── template.png
    │   └── test.png
    ├── pair_2/
    │   ├── template.jpg
    │   └── test.jpg
    ```
    """)
    
    zip_file = st.file_uploader(
        "Upload ZIP with image pairs",
        type=['zip'],
        help="Upload a ZIP file containing folders with template-test image pairs"
    )
    
    if zip_file:
        # Process ZIP file
        try:
            with zipfile.ZipFile(zip_file) as z:
                # Get all folders
                folders = set()
                for name in z.namelist():
                    if name.endswith('/'):
                        folders.add(name)
                    else:
                        folders.add(os.path.dirname(name) + '/')
                
                folders = sorted(list(folders))
                if not folders:
                    st.error("❌ No folders found in ZIP file!")
                    return
                
                # Show structure
                st.markdown("### ZIP Contents")
                for folder in folders:
                    files = [f for f in z.namelist() if f.startswith(folder) and not f.endswith('/')]
                    if len(files) != 2:
                        st.warning(f"⚠️ Skipping {folder}: Expected 2 images, found {len(files)}")
                        continue
                        
                    st.text(f"📁 {folder}")
                    for f in files:
                        st.text(f"  └─ {os.path.basename(f)}")
                
                # Process button
                if st.button("🚀 Process ZIP", type="primary"):
                    process_zip_file(z, folders)
        
        except Exception as e:
            st.error(f"❌ Error reading ZIP file: {str(e)}")


def process_individual_pairs():
    """Process uploaded individual pairs"""
    if not st.session_state.image_pairs:
        return
    
    if st.button("🚀 Process Batch", type="primary", use_container_width=True):
        process_batch(st.session_state.image_pairs)


def process_zip_file(zip_file, folders):
    """Process pairs from ZIP file"""
    pairs = []
    
    with st.spinner("Preparing image pairs..."):
        for folder in folders:
            files = [f for f in zip_file.namelist() 
                    if f.startswith(folder) and not f.endswith('/')]
            
            if len(files) != 2:
                continue
                
            # Find template and test images
            template_file = next((f for f in files if 'template' in f.lower()), files[0])
            test_file = next((f for f in files if 'test' in f.lower()), files[1])
            
            # Read images
            template_data = zip_file.read(template_file)
            test_data = zip_file.read(test_file)
            
            pairs.append({
                'template': BytesIO(template_data),
                'test': BytesIO(test_data),
                'template_name': os.path.basename(template_file),
                'test_name': os.path.basename(test_file)
            })
    
    if pairs:
        process_batch(pairs)
    else:
        st.error("❌ No valid image pairs found in ZIP!")


def process_batch(pairs):
    """Process a batch of image pairs"""
    
    backend = get_backend()
    processor = BatchProcessor()
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    results_placeholder = st.empty()
    
    try:
        batch_results = []
        
        for idx, pair in enumerate(pairs):
            progress = (idx + 1) / len(pairs)
            pair_name = f"{pair['template_name']} - {pair['test_name']}"
            
            progress_text.text(f"Processing pair {idx + 1}/{len(pairs)}: {pair_name}")
            progress_bar.progress(progress)
            
            # Load images
            template_img = Image.open(pair['template'])
            test_img = Image.open(pair['test'])
            
            # Process pair
            result = processor.process_pair(template_img, test_img)
            
            # Format results
            formatted = backend.format_results_for_display(result)
            formatted['pair_name'] = pair_name
            batch_results.append(formatted)
            
            # Update display
            display_batch_results(batch_results, results_placeholder)
        
        progress_text.text("✅ Batch processing complete!")
        progress_bar.progress(1.0)
        
        # Save results
        st.session_state.batch_results = batch_results
        
        # Show export options
        show_batch_export_options(batch_results)
        
    except Exception as e:
        st.error(f"❌ Error during batch processing: {str(e)}")
        progress_bar.empty()
        progress_text.empty()


def display_batch_results(results, placeholder):
    """Display batch processing results in real-time"""
    
    with placeholder.container():
        # Summary metrics
        st.markdown("### Batch Summary")
        
        total_defects = sum(r['summary']['total_defects'] for r in results)
        avg_confidence = sum(float(r['summary']['average_confidence'].split('%')[0]) 
                           for r in results) / len(results)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Pairs Processed", len(results))
        with col2:
            st.metric("Total Defects Found", total_defects)
        with col3:
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")
        
        # Results table
        st.markdown("### Results by Pair")
        
        table_data = []
        for r in results:
            table_data.append({
                'Pair Name': r['pair_name'],
                'Defects Found': r['summary']['total_defects'],
                'Confidence': r['summary']['average_confidence'],
                'Processing Time': r['processing_time'],
                'Status': '✅ Pass' if r['summary']['total_defects'] == 0 else '⚠️ Review'
            })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(
                df.style.background_gradient(subset=['Defects Found'], cmap='YlOrRd'),
                use_container_width=True
            )


def show_batch_export_options(results):
    """Show export options for batch results"""
    
    st.markdown("### 📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export CSV
        csv_data = []
        for r in results:
            base_info = {
                'pair_name': r['pair_name'],
                'total_defects': r['summary']['total_defects'],
                'avg_confidence': r['summary']['average_confidence'],
                'processing_time': r['processing_time']
            }
            
            if r['defect_details']:
                for defect in r['defect_details']:
                    row = base_info.copy()
                    row.update(defect)
                    csv_data.append(row)
            else:
                csv_data.append(base_info)
        
        csv_buffer = BytesIO()
        pd.DataFrame(csv_data).to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_buffer.getvalue(),
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export JSON
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'total_pairs': len(results),
            'results': results
        }
        
        st.download_button(
            label="📥 Download JSON Data",
            data=json.dumps(json_data, indent=2),
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )