import cv2  # Import OpenCV library for image and video processing
import numpy as np  # Import NumPy for numerical operations

def deskew(image):  # Define a function 'deskew' that takes an image file path as input
    # Load an image from the specified file path
    img = cv2.imread(image)  # Use img_path as the argument
     
    # Convert the image from BGR to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image (black becomes white and vice versa)
    gray = cv2.bitwise_not(gray)
    
    # Threshold the inverted grayscale image using Otsu's method to create a binary image
    # This will automatically determine the optimal threshold value
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Get the coordinates of all points where the thresholded image is greater than 0 (i.e., white pixels)
    coords = np.column_stack(np.where(thresh > 0))
    
    # Get the minimum area rectangle that can enclose the points
    # This function returns the center, size, and angle of the rectangle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust the angle based on its value
    if angle < -45:  # If the angle is less than -45 degrees
        angle = -(90 + angle)  # Convert it to a positive angle for rotation
    else:
        angle = 90 - angle  # Otherwise, adjust it to the rotation angle for the rectangle
    
    # Get the height and width of the original image
    (h, w) = img.shape[:2]
    
    # Calculate the center of the image for rotation
    center = (w // 2, h // 2)
    
    # Get the rotation matrix for rotating the image around its center
    # The third parameter (1.0) indicates the scaling factor (no scaling here)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the original image using the rotation matrix
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
   
    return rotated  # Return the deskewed (rotated) image
