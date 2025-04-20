# Defect Detection Application

A Streamlit web application that detects defects in industrial objects using TensorFlow Lite machine learning models.

## Features

- **Multiple Object Types**: Support for 15 different object types (bottle, cable, capsule, carpet, etc.)
- **Single Image Classification**: Upload and analyze individual images
- **Batch Processing**: Upload multiple images or a ZIP archive for batch analysis
- **Adjustable Sensitivity**: Control defect detection threshold with a slider
- **Statistics Dashboard**: View defect rates and processing statistics over time
- **Filter & Sort Results**: Organize results by defect status, confidence, and date

## Object Types Supported

- Bottle
- Cable
- Capsule
- Carpet
- Grid
- Hazelnut
- Leather
- Metal Nut
- Pill
- Screw
- Tile
- Toothbrush
- Transistor
- Wood
- Zipper

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Model Information

This application uses TensorFlow Lite models trained on the MVTec Anomaly Detection dataset. Each model is specialized for detecting defects in a specific type of object.

## Data Storage

The application stores classification results and statistics in memory during the session and saves statistics to a local JSON file.

## Project Structure

- `app.py`: Main Streamlit application
- `model.py`: Defect detection model implementation
- `utils.py`: Utility functions for image processing and data handling
- `models/`: Directory containing TensorFlow Lite models
- `sample_images/`: Sample images for testing (optional)

## Requirements

See `requirements.txt` for a list of dependencies.