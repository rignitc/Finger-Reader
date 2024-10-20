import cv2  # OpenCV library for computer vision
import numpy as np  # NumPy for numerical operations
import math  # Math library for mathematical functions
from ocr import read  # Importing read function for OCR
from deskew import deskew  # Importing deskew function to correct image orientation
from sound import sound  # Importing sound function to convert text to speech

# Initialize video capture from the default camera (0)
cap = cv2.VideoCapture(0)

# Function to calculate the Euclidean distance between two points
def sqrt_d(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Variable to hold the last recognized voice
voice = None

# Main loop to continuously capture frames from the webcam
while True:
    # Read a frame from the camera
    _, img = cap.read()

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the minimum and maximum HSV values for color detection (This is the range of skin color)
    hsv_min = np.array([0, 30, 60])  # Lower bound for HSV values
    hsv_max = np.array([20, 150, 255])  # Upper bound for HSV values

    # Create a mask to isolate the colors within the specified range (Isolating the image of finger)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # Correct any skew in the mask image (If the code is slow, comment this line. It works without deskewing, but camera has to be held as to get vertical image)
    mask = deskew(mask)  # Correct any skew in the mask image

    # Apply thresholding to create a binary image
    ret, thresh = cv2.threshold(mask, 2, 255, 0)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if contours:
        # Get the largest contour based on area
        c = max(contours, key=cv2.contourArea)

        m = 1000  # Initialize a large value for minimum y-coordinate
        ans = None  # Variable to hold the coordinates of the lowest point in the largest contour

        # Iterate through the points in the largest contour to find the lowest point
        for i in c:
            for j in i:
                if j[1] < m:  # Check if the y-coordinate is less than the current minimum
                    m = j[1]  # Update minimum y-coordinate
                    ans = j[0]  # Store the x-coordinate
                    ans1 = j[1]  # Store the y-coordinate

        # cx and cy are the coordinates of the lowest point in the largest contour (The points are the coordinates of the tip of the finger)
        cx, cy = ans, ans1
        min_dist = 1000  # Initialize a large distance for comparison
        final_text = None  # Variable to hold the recognized text

        # Perform OCR on the image and get detected text and their positions
        for text, center in read(img).items():
            # Calculate the distance between the detected text and the lowest contour point tip of the finger)(
            dist = sqrt_d(cx, cy, center[0], center[1])
            if dist < min_dist:  # If this distance is the smallest found
                min_dist = dist  # Update minimum distance
                final_text = text  # Store the recognized text

        # Try to speak the recognized text if it is different from the last spoken text
        try:
            if voice != final_text:  # If the new text is different from the previous
                sound(final_text)  # Convert the text to speech
                print(final_text)  # Print the recognized text
                voice = final_text  # Update the last spoken text
        except Exception as e:
            print(f"Error in sound function: {e}")  # Print any errors

    # Show the processed image in a window named 'mask'
    cv2.imshow('mask', img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
   
