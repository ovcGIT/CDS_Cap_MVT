import streamlit as st
import pandas as pd
import os
from PIL import Image
import io
import numpy as np
import time
import glob
import tempfile
import zipfile
import shutil
from datetime import datetime
from model import predict_defect, AVAILABLE_OBJECTS
from utils import create_image_preview, load_statistics, save_statistics, process_batch, process_folder

# Set page configuration
st.set_page_config(
    page_title="Defect Detection System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables if they don't exist
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "results" not in st.session_state:
    st.session_state.results = []
if "statistics" not in st.session_state:
    st.session_state.statistics = load_statistics()
if "selected_object_type" not in st.session_state:
    st.session_state.selected_object_type = AVAILABLE_OBJECTS[0] if AVAILABLE_OBJECTS else "capsule"
if "defect_threshold" not in st.session_state:
    st.session_state.defect_threshold = 0.15  # Default threshold value

# Function to display sample results
def display_sample_results(results):
    if results:
        st.subheader("Sample Results")
        max_samples = min(6, len(results))
        cols = st.columns(3)
        for i in range(max_samples):
            col = cols[i % 3]
            result = results[i]
            col.image(result["image"], caption=result["filename"])
            if result["is_defective"]:
                col.error(f"⚠️ Defect: {result['confidence']:.1f}%")
            else:
                col.success(f"✅ No defect: {result['confidence']:.1f}%")
            
            if "object_type" in result:
                col.write(f"Object Type: {result['object_type']}")

# Main title
st.title("Defect Detection System")

# Sidebar for navigation
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Upload & Classify", "View Results", "Results by Object Type", "Statistics"]
)

# Home page
if page == "Home":
    st.header("Welcome to the Defect Detection System")
    
    st.write("""
    This application helps you identify defective objects in images using machine learning.
    
    ### Features:
    - Upload and classify individual images
    - Batch process multiple images
    - View classification results with confidence scores
    - Track statistics on defect detection
    
    ### Getting Started:
    Navigate to the 'Upload & Classify' section to begin analyzing your images.
    """)
    
    # Display some statistics if available
    if len(st.session_state.statistics) > 0:
        st.subheader("Quick Statistics")
        col1, col2, col3 = st.columns(3)
        
        # Calculate totals across all dates
        total_images = sum(st.session_state.statistics[date]["total"] for date in st.session_state.statistics)
        total_defects = sum(st.session_state.statistics[date]["defective"] for date in st.session_state.statistics)
        defect_rate = (total_defects / total_images * 100) if total_images > 0 else 0
        
        col1.metric("Total Images Processed", total_images)
        col2.metric("Detected Defects", total_defects)
        col3.metric("Defect Rate", f"{defect_rate:.1f}%")

# Upload and classification page
elif page == "Upload & Classify":
    st.header("Upload and Classify Images")
    
    # Create tabs for single image and batch upload
    tab1, tab2 = st.tabs(["Single Image", "Batch Processing"])
    
    # Single Image Upload
    with tab1:
        # Object type selector
        st.session_state.selected_object_type = st.selectbox(
            "Select object type for defect detection", 
            AVAILABLE_OBJECTS,
            index=AVAILABLE_OBJECTS.index(st.session_state.selected_object_type) if st.session_state.selected_object_type in AVAILABLE_OBJECTS else 0
        )
        
        # Defect threshold slider
        st.session_state.defect_threshold = st.slider(
            "Defect detection threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.defect_threshold,
            step=0.01,
            help="Lower values will detect more defects (more sensitive). Higher values require stronger evidence of defects (less sensitive)."
        )
        
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process and display the image
            image = Image.open(uploaded_file)
            preview = create_image_preview(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(preview, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.info("Image Details:")
                st.write(f"Filename: {uploaded_file.name}")
                st.write(f"Size: {image.width} x {image.height} pixels")
                st.write(f"Format: {image.format}")
                st.write(f"Object Type: {st.session_state.selected_object_type}")
                
                # Button to trigger classification
                if st.button("Classify Image"):
                    with st.spinner("Processing image..."):
                        # Call the prediction model with the selected object type and threshold
                        is_defective, confidence = predict_defect(
                            image, 
                            st.session_state.selected_object_type,
                            threshold=st.session_state.defect_threshold
                        )
                        
                        # Save results
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        result = {
                            "filename": uploaded_file.name,
                            "timestamp": timestamp,
                            "is_defective": is_defective,
                            "confidence": confidence,
                            "image": preview,
                            "object_type": st.session_state.selected_object_type
                        }
                        
                        st.session_state.uploaded_images.append(image)
                        st.session_state.results.append(result)
                        
                        # Update statistics
                        today = datetime.now().strftime("%Y-%m-%d")
                        if today not in st.session_state.statistics:
                            st.session_state.statistics[today] = {"total": 0, "defective": 0}
                        
                        st.session_state.statistics[today]["total"] += 1
                        if is_defective:
                            st.session_state.statistics[today]["defective"] += 1
                        
                        save_statistics(st.session_state.statistics)
                    
                    # Display the result
                    if is_defective:
                        st.error(f"⚠️ Defect detected with {confidence:.1f}% confidence")
                    else:
                        st.success(f"✅ No defect detected ({confidence:.1f}% confidence)")
    
    # Batch Processing
    with tab2:
        # Create sub-tabs for different batch processing methods
        method_tab1, method_tab2 = st.tabs(["Multiple Files", "Zip Archive"])
        
        # Object type selector for batch processing (shared by both methods)
        batch_object_type = st.selectbox(
            "Select object type for batch defect detection", 
            AVAILABLE_OBJECTS,
            index=AVAILABLE_OBJECTS.index(st.session_state.selected_object_type) if st.session_state.selected_object_type in AVAILABLE_OBJECTS else 0,
            key="batch_object_selector"
        )
        
        # Defect threshold slider for batch processing
        batch_threshold = st.slider(
            "Defect detection threshold for batch processing", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.defect_threshold,
            step=0.01,
            key="batch_threshold_slider",
            help="Lower values will detect more defects (more sensitive). Higher values require stronger evidence of defects (less sensitive)."
        )
        
        # Method 1: Upload multiple individual files
        with method_tab1:
            uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            
            if uploaded_files:
                st.write(f"Uploaded {len(uploaded_files)} images")
                
                if st.button("Process Multiple Files", key="process_multiple"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each image in the batch with the selected object type and threshold
                    results = process_batch(uploaded_files, progress_bar, status_text, batch_object_type, threshold=batch_threshold)
                    
                    # Add results to session state
                    for result in results:
                        st.session_state.results.append(result)
                        st.session_state.uploaded_images.append(result["image_obj"])
                        
                        # Update statistics
                        today = datetime.now().strftime("%Y-%m-%d")
                        if today not in st.session_state.statistics:
                            st.session_state.statistics[today] = {"total": 0, "defective": 0}
                        
                        st.session_state.statistics[today]["total"] += 1
                        if result["is_defective"]:
                            st.session_state.statistics[today]["defective"] += 1
                    
                    save_statistics(st.session_state.statistics)
                    
                    # Display summary
                    defective_count = sum(1 for r in results if r["is_defective"])
                    st.write(f"Processing complete: {len(results)} images analyzed, {defective_count} defects found")
                    
                    # Show a sample of results
                    display_sample_results(results)
        
        # Method 2: Upload a ZIP file containing images
        with method_tab2:
            st.write("Upload a ZIP file containing images of the same object type:")
            uploaded_zip = st.file_uploader("Upload ZIP file", type=["zip"], key="zip_uploader")
            
            if uploaded_zip is not None:
                if st.button("Process ZIP Archive", key="process_zip"):
                    # Create a temporary directory to extract the zip file
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Write the zip file to the temp directory
                        zip_path = os.path.join(temp_dir, "uploaded.zip")
                        with open(zip_path, "wb") as f:
                            f.write(uploaded_zip.getvalue())
                        
                        # Extract the zip file
                        extract_dir = os.path.join(temp_dir, "extracted")
                        os.makedirs(extract_dir, exist_ok=True)
                        
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                            
                            # Process the extracted images
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            results = process_folder(extract_dir, progress_bar, status_text, batch_object_type, threshold=batch_threshold)
                            
                            # Add results to session state
                            for result in results:
                                st.session_state.results.append(result)
                                st.session_state.uploaded_images.append(result["image_obj"])
                                
                                # Update statistics
                                today = datetime.now().strftime("%Y-%m-%d")
                                if today not in st.session_state.statistics:
                                    st.session_state.statistics[today] = {"total": 0, "defective": 0}
                                
                                st.session_state.statistics[today]["total"] += 1
                                if result["is_defective"]:
                                    st.session_state.statistics[today]["defective"] += 1
                            
                            save_statistics(st.session_state.statistics)
                            
                            # Display summary
                            defective_count = sum(1 for r in results if r["is_defective"])
                            st.write(f"Processing complete: {len(results)} images analyzed, {defective_count} defects found")
                            
                            # Show a sample of results
                            display_sample_results(results)
                            
                        except Exception as e:
                            st.error(f"Error processing ZIP file: {e}")

if page == "View Results":
    st.header("Classification Results")
    
    if not st.session_state.results:
        st.info("No images have been classified yet. Go to 'Upload & Classify' to process some images.")
    else:
        # Filter options
        st.subheader("Filter Results")
        col1, col2 = st.columns(2)
        
        with col1:
            filter_option = st.selectbox(
                "Show only:",
                ["All", "Defective", "Non-Defective"]
            )
        
        with col2:
            sort_option = st.selectbox(
                "Sort by:",
                ["Newest First", "Oldest First", "Confidence (High to Low)", "Confidence (Low to High)"]
            )
        
        # Apply filters
        filtered_results = st.session_state.results.copy()
        
        if filter_option == "Defective":
            filtered_results = [r for r in filtered_results if r["is_defective"]]
        elif filter_option == "Non-Defective":
            filtered_results = [r for r in filtered_results if not r["is_defective"]]
        
        # Apply sorting
        if sort_option == "Newest First":
            filtered_results = sorted(filtered_results, key=lambda x: x["timestamp"], reverse=True)
        elif sort_option == "Oldest First":
            filtered_results = sorted(filtered_results, key=lambda x: x["timestamp"])
        elif sort_option == "Confidence (High to Low)":
            filtered_results = sorted(filtered_results, key=lambda x: x["confidence"], reverse=True)
        elif sort_option == "Confidence (Low to High)":
            filtered_results = sorted(filtered_results, key=lambda x: x["confidence"])
        
        # Display results
        st.write(f"Showing {len(filtered_results)} of {len(st.session_state.results)} results")
        
        # Display in a grid
        cols_per_row = 3
        for i in range(0, len(filtered_results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(filtered_results):
                    result = filtered_results[idx]
                    col.image(result["image"], caption=result["filename"], use_container_width=True)
                    
                    # Display result with color coding
                    if result["is_defective"]:
                        col.error(f"⚠️ Defective ({result['confidence']:.1f}%)")
                    else:
                        col.success(f"✅ No defect ({result['confidence']:.1f}%)")
                    
                    # Display object type if available
                    if "object_type" in result:
                        col.write(f"Object Type: {result['object_type']}")
                    
                    col.write(f"Processed: {result['timestamp']}")

# Results by Object Type page
if page == "Results by Object Type":
    st.header("Results by Object Type")
    
    if not st.session_state.results:
        st.info("No images have been classified yet. Go to 'Upload & Classify' to process some images.")
    else:
        # Get unique object types from the results
        object_types = set()
        for result in st.session_state.results:
            if "object_type" in result:
                object_types.add(result["object_type"])
        
        if not object_types:
            st.warning("No object type information found in the results.")
        else:
            # Allow user to select which object type to view
            selected_type = st.selectbox(
                "Select object type to view",
                sorted(list(object_types))
            )
            
            # Filter results by the selected object type
            object_results = [r for r in st.session_state.results if r.get("object_type") == selected_type]
            
            if not object_results:
                st.info(f"No results found for object type: {selected_type}")
            else:
                # Additional filter options
                col1, col2 = st.columns(2)
                
                with col1:
                    filter_option = st.selectbox(
                        "Show only:",
                        ["All", "Defective", "Non-Defective"],
                        key="object_filter"
                    )
                
                with col2:
                    sort_option = st.selectbox(
                        "Sort by:",
                        ["Newest First", "Oldest First", "Confidence (High to Low)", "Confidence (Low to High)"],
                        key="object_sort"
                    )
                
                # Apply filters
                filtered_results = object_results.copy()
                
                if filter_option == "Defective":
                    filtered_results = [r for r in filtered_results if r["is_defective"]]
                elif filter_option == "Non-Defective":
                    filtered_results = [r for r in filtered_results if not r["is_defective"]]
                
                # Apply sorting
                if sort_option == "Newest First":
                    filtered_results = sorted(filtered_results, key=lambda x: x["timestamp"], reverse=True)
                elif sort_option == "Oldest First":
                    filtered_results = sorted(filtered_results, key=lambda x: x["timestamp"])
                elif sort_option == "Confidence (High to Low)":
                    filtered_results = sorted(filtered_results, key=lambda x: x["confidence"], reverse=True)
                elif sort_option == "Confidence (Low to High)":
                    filtered_results = sorted(filtered_results, key=lambda x: x["confidence"])
                
                # Statistics for the selected object type
                defect_count = sum(1 for r in filtered_results if r["is_defective"])
                total_count = len(filtered_results)
                defect_rate = (defect_count / total_count * 100) if total_count > 0 else 0
                
                # Display object type specific statistics
                st.subheader(f"Statistics for {selected_type}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Images", total_count)
                col2.metric("Defective Images", defect_count)
                col3.metric("Defect Rate", f"{defect_rate:.1f}%")
                
                # Display results
                st.subheader(f"Showing {len(filtered_results)} results for {selected_type}")
                
                # Display in a grid
                cols_per_row = 3
                for i in range(0, len(filtered_results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(filtered_results):
                            result = filtered_results[idx]
                            col.image(result["image"], caption=result["filename"], use_container_width=True)
                            
                            # Display result with color coding
                            if result["is_defective"]:
                                col.error(f"⚠️ Defective ({result['confidence']:.1f}%)")
                            else:
                                col.success(f"✅ No defect ({result['confidence']:.1f}%)")
                            
                            col.write(f"Processed: {result['timestamp']}")

# Statistics page
if page == "Statistics":
    st.header("Defect Detection Statistics")
    
    if not st.session_state.statistics:
        st.info("No statistics available yet. Process some images to generate statistics.")
    else:
        # Prepare data for visualization
        dates = list(st.session_state.statistics.keys())
        total_counts = [st.session_state.statistics[date]["total"] for date in dates]
        defective_counts = [st.session_state.statistics[date]["defective"] for date in dates]
        
        # Calculate overall statistics
        total_images = sum(total_counts)
        total_defects = sum(defective_counts)
        defect_rate = (total_defects / total_images * 100) if total_images > 0 else 0
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Images Processed", total_images)
        col2.metric("Total Defects Detected", total_defects)
        col3.metric("Overall Defect Rate", f"{defect_rate:.1f}%")
        
        # Create dataframe for charting
        df = pd.DataFrame({
            "Date": dates,
            "Total Images": total_counts,
            "Defective Images": defective_counts
        })
        
        # Calculate defect rates
        df["Defect Rate (%)"] = (df["Defective Images"] / df["Total Images"] * 100).round(1)
        
        # Display daily statistics
        st.subheader("Daily Statistics")
        st.dataframe(df)
        
        # Bar chart for images processed
        st.subheader("Images Processed by Day")
        st.bar_chart(df.set_index("Date")[["Total Images", "Defective Images"]])
        
        # Line chart for defect rate
        st.subheader("Defect Rate Trend")
        st.line_chart(df.set_index("Date")["Defect Rate (%)"])
