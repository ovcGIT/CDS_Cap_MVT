"""
Utility functions for the defect detection application.
"""

import os
import json
import glob
from datetime import datetime
import io
from PIL import Image
import time
import numpy as np
from model import predict_defect, preprocess_image

def create_image_preview(image, max_size=(300, 300)):
    """
    Create a preview of an image with a maximum size.
    
    Args:
        image (PIL.Image): The input image
        max_size (tuple): Maximum dimensions (width, height)
        
    Returns:
        PIL.Image: Resized image
    """
    image.thumbnail(max_size)
    return image

def load_statistics():
    """
    Load statistics from the statistics file.
    
    Returns:
        dict: Statistics data
    """
    try:
        if os.path.exists('statistics.json'):
            with open('statistics.json', 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading statistics: {e}")
        return {}

def save_statistics(statistics):
    """
    Save statistics to the statistics file.
    
    Args:
        statistics (dict): Statistics data to save
    """
    try:
        with open('statistics.json', 'w') as f:
            json.dump(statistics, f)
    except Exception as e:
        print(f"Error saving statistics: {e}")

def process_batch(uploaded_files, progress_bar, status_text, object_type="capsule", threshold=0.15):
    """
    Process a batch of uploaded image files.
    
    Args:
        uploaded_files (list): List of uploaded file objects
        progress_bar: Streamlit progress bar object
        status_text: Streamlit empty text element for status updates
        object_type (str): Type of object to detect defects in
        threshold (float): Threshold value for defect detection (0.0-1.0)
        
    Returns:
        list: Results for each processed image
    """
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing image {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        try:
            # Process the image
            image = Image.open(uploaded_file)
            
            # Classify the image with the specified object type and threshold
            is_defective, confidence = predict_defect(image, object_type, threshold=threshold)
            
            # Create preview
            preview = create_image_preview(image.copy())
            
            # Store results
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = {
                "filename": uploaded_file.name,
                "timestamp": timestamp,
                "is_defective": is_defective,
                "confidence": confidence,
                "image": preview,
                "image_obj": image,
                "object_type": object_type
            }
            
            results.append(result)
            
            # Small delay for UI responsiveness
            time.sleep(0.1)
            
        except Exception as e:
            # Handle errors
            print(f"Error processing {uploaded_file.name}: {e}")
            status_text.text(f"Error processing {uploaded_file.name}")
            
    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    return results

def process_folder(folder_path, progress_bar, status_text, object_type="capsule", threshold=0.15):
    """
    Process all images in a folder.
    
    Args:
        folder_path (str): Path to the folder containing images
        progress_bar: Streamlit progress bar object
        status_text: Streamlit empty text element for status updates
        object_type (str): Type of object to detect defects in
        threshold (float): Threshold value for defect detection (0.0-1.0)
        
    Returns:
        list: Results for each processed image
    """
    results = []
    
    # Find all image files in the folder and subfolders
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
    
    # Remove any duplicates
    image_files = list(set(image_files))
    
    # Sort the files for consistent ordering
    image_files.sort()
    
    if not image_files:
        status_text.text("No image files found in the folder")
        return results
    
    # Process each image
    for i, file_path in enumerate(image_files):
        # Update progress
        progress = (i + 1) / len(image_files)
        progress_bar.progress(progress)
        
        # Get just the filename for display
        filename = os.path.basename(file_path)
        status_text.text(f"Processing image {i+1}/{len(image_files)}: {filename}")
        
        try:
            # Open and process the image
            image = Image.open(file_path)
            
            # Classify the image with the specified object type and threshold
            is_defective, confidence = predict_defect(image, object_type, threshold=threshold)
            
            # Create preview
            preview = create_image_preview(image.copy())
            
            # Store results
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = {
                "filename": filename,
                "timestamp": timestamp,
                "is_defective": is_defective,
                "confidence": confidence,
                "image": preview,
                "image_obj": image,
                "object_type": object_type
            }
            
            results.append(result)
            
            # Small delay for UI responsiveness
            time.sleep(0.1)
            
        except Exception as e:
            # Handle errors
            print(f"Error processing {filename}: {e}")
            status_text.text(f"Error processing {filename}")
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    return results
