import pytesseract  # Import the Tesseract OCR library
from pytesseract import Output  # Import Output to use specific output types
import cv2  # OpenCV library for image processing
import numpy as np  # NumPy for numerical operations

def read(img):
    # Use Tesseract to extract data from the image
    d = pytesseract.image_to_data(img, output_type=Output.DICT)

    n_boxes = len(d['level'])  # Get the number of detected boxes
    new_d = {}  # Dictionary to store text and its center coordinates

    for i in range(n_boxes):
        # Get the bounding box coordinates and dimensions
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        
        # Draw a rectangle around the detected text in the image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Check if the detected text is not empty
        if d['text'][i].strip():  # Ignore empty text entries
            new_d[d['text'][i]] = [int(x + w / 2), int(y + h / 2)]  # Store text with center coordinates

    return new_d  # Return the dictionary containing detected text and their positions

