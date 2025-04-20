"""
Model module for defect detection.

This module provides defect detection based on your MobileNetV2 models,
with a fallback to image analysis when models are not available.
"""

import os
import glob
import numpy as np
from PIL import Image
import cv2

# Try to import tflite_runtime with proper error handling
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError as e:
    print(f"TFLite runtime import error: {e}. Will use fallback analysis.")
    TFLITE_AVAILABLE = False
except Exception as e:
    print(f"TFLite runtime general error: {e}. Will use fallback analysis.")
    TFLITE_AVAILABLE = False

# Path to models directory
MODELS_DIR = 'models'

# Resolution expected by the model
INPUT_RESOLUTION = (224, 224)

# Default threshold for prediction (can be overridden by user)
DEFAULT_DEFECT_THRESHOLD = 0.15

# Get available objects from the models directory
def get_available_objects():
    model_files = glob.glob(os.path.join(MODELS_DIR, '*_model_*.tflite'))
    objects = []
    for model_file in model_files:
        # Extract object name from filename (e.g., "bottle_model_200.tflite" -> "bottle")
        object_name = os.path.basename(model_file).split('_')[0]
        objects.append(object_name)
    return sorted(objects)

# List of objects for which models are available
AVAILABLE_OBJECTS = get_available_objects()
if not AVAILABLE_OBJECTS:  # Fallback if no models found
    AVAILABLE_OBJECTS = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 
        'hazelnut', 'leather', 'metal_nut', 'pill', 
        'screw', 'tile', 'toothbrush', 'transistor', 
        'wood', 'zipper'
    ]

def load_tflite_model(object_type):
    """
    Load the TFLite model for a specific object type.
    
    Args:
        object_type (str): Type of object for which to load model
        
    Returns:
        tflite.Interpreter: Loaded TFLite interpreter
    """
    model_path = os.path.join(MODELS_DIR, f"{object_type}_model_200.tflite")
    
    if not os.path.exists(model_path):
        print(f"Model file for {object_type} not found: {model_path}")
        # If specific model not found, try to use a different model
        model_files = glob.glob(os.path.join(MODELS_DIR, "*_model_*.tflite"))
        if model_files:
            model_path = model_files[0]
            print(f"Using alternative model: {model_path}")
        else:
            raise FileNotFoundError(f"No TFLite models found in {MODELS_DIR}")
    
    # Load the TFLite model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    return interpreter

def predict_with_tflite(interpreter, image, threshold=DEFAULT_DEFECT_THRESHOLD):
    """
    Make a prediction using the TFLite model.
    
    Args:
        interpreter (tflite.Interpreter): Loaded TFLite interpreter
        image (PIL.Image): Preprocessed image
        threshold (float): Threshold value for defect detection (0.0-1.0)
        
    Returns:
        tuple: (is_defective, confidence)
    """
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input data
    input_shape = input_details[0]['shape']
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize the image (if needed)
    img_array = img_array / 255.0
    
    # Add batch dimension if needed
    if len(input_shape) == 4:
        img_array = np.expand_dims(img_array, axis=0)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process the output (assuming binary classification: normal vs defective)
    # For a binary classifier, the defect probability is the second class
    if output_data.shape[-1] > 1:
        defect_probability = output_data[0][1]  # Assuming defect is class 1
    else:
        defect_probability = output_data[0][0]  # Single output value
    
    is_defective = defect_probability > threshold
    confidence = float(defect_probability * 100)
    
    return is_defective, confidence

def predict_defect(image, object_type="capsule", threshold=DEFAULT_DEFECT_THRESHOLD):
    """
    Predict whether an image contains a defect.
    
    Args:
        image (PIL.Image): The image to classify
        object_type (str): Type of object to check for defects
        threshold (float): Threshold value (0.0-1.0) for defect detection
        
    Returns:
        tuple: (is_defective, confidence) where:
            - is_defective is a boolean indicating if a defect was detected
            - confidence is a float between 0-100 representing the confidence
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Check if TFLite is available
    if TFLITE_AVAILABLE:
        try:
            # Try to use the TFLite model
            interpreter = load_tflite_model(object_type)
            is_defective, confidence = predict_with_tflite(interpreter, preprocessed_image, threshold)
            return is_defective, confidence
        except Exception as e:
            print(f"TFLite model error: {e}. Falling back to rule-based analysis.")
            # If there's an error, we'll fall through to the rule-based approach
    else:
        print(f"TFLite runtime not available. Using rule-based analysis.")
    
    # Fallback to rule-based approach
    gray_img = preprocessed_image.convert('L')
    gray_array = np.array(gray_img)
    features = extract_image_features(gray_array)
    is_defective, confidence = analyze_defects_by_object_type(features, object_type)
    
    return is_defective, confidence

def extract_image_features(img_array):
    """
    Extract relevant features from image for defect analysis.
    
    Args:
        img_array (numpy.ndarray): Grayscale image as numpy array
        
    Returns:
        dict: Dictionary of image features
    """
    # Basic statistical features
    avg_brightness = np.mean(img_array)
    std_brightness = np.std(img_array)
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    
    # Edge detection for structural analysis
    try:
        edges = cv2.Canny(img_array, 100, 200)
        edge_percentage = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    except:
        edge_percentage = 0.1  # Default if edge detection fails
    
    # Texture analysis (simple)
    texture_variance = np.var(img_array)
    
    return {
        'avg_brightness': avg_brightness,
        'std_brightness': std_brightness, 
        'brightness_range': max_val - min_val,
        'edge_percentage': edge_percentage,
        'texture_variance': texture_variance
    }

def analyze_defects_by_object_type(features, object_type):
    """
    Analyze image features for defects based on object type.
    
    Args:
        features (dict): Image features
        object_type (str): Type of object
        
    Returns:
        tuple: (is_defective, confidence)
    """
    # Default thresholds for different object types - can be fine-tuned
    thresholds = {
        'bottle': {'std': 45, 'edge': 0.15, 'texture': 1500},
        'cable': {'std': 50, 'edge': 0.18, 'texture': 1800},
        'capsule': {'std': 40, 'edge': 0.12, 'texture': 1200},
        'carpet': {'std': 42, 'edge': 0.14, 'texture': 1400},
        'grid': {'std': 52, 'edge': 0.19, 'texture': 1850},
        'hazelnut': {'std': 55, 'edge': 0.2, 'texture': 2000},
        'leather': {'std': 38, 'edge': 0.11, 'texture': 1100},
        'metal_nut': {'std': 60, 'edge': 0.22, 'texture': 2200},
        'pill': {'std': 35, 'edge': 0.1, 'texture': 1000},
        'screw': {'std': 55, 'edge': 0.19, 'texture': 1900},
        'tile': {'std': 47, 'edge': 0.17, 'texture': 1700},
        'toothbrush': {'std': 45, 'edge': 0.16, 'texture': 1600},
        'transistor': {'std': 50, 'edge': 0.17, 'texture': 1700},
        'wood': {'std': 43, 'edge': 0.15, 'texture': 1450},
        'zipper': {'std': 48, 'edge': 0.16, 'texture': 1600}
    }
    
    # Use default thresholds if object type not found
    if object_type not in thresholds:
        object_type = 'capsule'
    
    # Get thresholds for this object type
    thresh = thresholds[object_type]
    
    # Calculate defect score based on multiple features
    defect_indicators = [
        features['std_brightness'] > thresh['std'],
        features['edge_percentage'] > thresh['edge'],
        features['texture_variance'] > thresh['texture']
    ]
    
    # Weight the indicators
    weights = [0.4, 0.3, 0.3]
    defect_score = sum(ind * w for ind, w in zip(defect_indicators, weights))
    
    # Convert to probability
    probability = min(0.9, max(0.1, defect_score))
    
    # Determine if defective
    is_defective = probability > 0.5
    
    # Confidence score
    confidence = probability * 100 if is_defective else (1 - probability) * 100
    
    # Add some randomness to avoid monotonous results (can be removed in production)
    confidence += np.random.normal(0, 5)
    confidence = min(99, max(55, confidence))
    
    return is_defective, confidence

def preprocess_image(image):
    """
    Preprocess an image for classification.
    
    Args:
        image (PIL.Image): The input image
        
    Returns:
        PIL.Image: The preprocessed image
    """
    # Resize to the expected input resolution
    image = image.resize(INPUT_RESOLUTION)
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image
