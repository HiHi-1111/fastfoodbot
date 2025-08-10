import easyocr
import numpy as np
import cv2
from PIL import Image
import torch

# Check if CUDA is available
gpu_available = torch.cuda.is_available()

# Initialize reader with GPU if available
reader = easyocr.Reader(['en'], gpu=gpu_available)

def find_text(image_section, text_to_find):
    """
    Detect text in image using EasyOCR and check if text_to_find exists.
    
    Args:
        image_section: numpy/opencv array of color pixels
        text_to_find: string to search for (case-sensitive)
    
    Returns:
        bool: True if text found, False otherwise
    """
    # Convert CV2 array to RGB if needed
    if isinstance(image_section, np.ndarray):
        if len(image_section.shape) == 3 and image_section.shape[2] == 3:
            image_section = cv2.cvtColor(image_section, cv2.COLOR_BGR2RGB)

    # Get optimal size for OCR (not too big, not too small)
    height, width = image_section.shape[:2]
    target_width = 800  # Good balance between speed and accuracy
    scale = target_width / width
    if width > target_width:
        new_width = target_width
        new_height = int(height * scale)
        image_section = cv2.resize(image_section, (new_width, new_height))

    # Detect text
    results = reader.readtext(image_section)
    
    # Check if target text exists in any detected text
    for (_, text, confidence) in results:
        if confidence > 0.5 and text_to_find.lower() in text.lower():
            return True
            
    return False

# # Add this to text_find.py to debug GPU detection
# def check_gpu():
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"Current device: {torch.cuda.get_device_name(0)}")

# # Call this when initializing your application
# check_gpu()
