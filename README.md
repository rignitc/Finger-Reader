# Finger-Reader

## Project Description

Designed and implemented a portable, finger-worn device to assist children and individuals with dyslexia in reading. Assists the user by reading out the words to which the user points. Also has an added mode to to detect objects to which the used points. The following work is presented is this project:
1. Designing and implementing the hardware of the device.
2. Developing the Python codes for reading assistance, utilizing the `OpenCV` library for image processing and the `Tesseract OCR engine` to extract text from images.
3. Developing the Python codes for object recognition using the `YOLOv4 algorithm`.

## Hardware

![alt text](Images/Screenshot%20from%202024-02-06%2021-31-54-modified.png)

* Raspberry Pi (3 B+ model)
* Raspberry Pi Camera Module v2
* Camera mount (3D printed)
* Wrist casing for Raspberry Pi (3D printed)
* Speaker

## Setting up the Reading Assistance feature

__Code__: [reader.py](reader.py)

The Python source code is run on the Raspberry Pi. OpenCV library is used to enhance the text in the‬
image and the Pytesseract library (OCR engine wrapper) extracts text from images. The text is converted to‬
speech by the gTTS library (Google Text-to-Speech). Represented pictorially as:

![alt text](Images/Screenshot%20from%202024-02-06%2021-32-05-modified.png)

### Defining function to Deskew the image

__Code__: [deskew.py](deskew.py)

To install OpenCV and numpy, run:
```python
pip install opencv-python
```
```python
pip install numpy
```
Certain image processing techniques need to be done, to obtain good results from OCR. Deskewing is the process of correcting the twists (skew) of the text in an image,  rotating the image such that text is vertically aligned. It includes the following process.
1. Image Loading & Grayscale Conversion: The image is loaded and converted to grayscale, simplifying processing by reducing it to a single channel.
2. Inversion & Thresholding: The grayscale image is inverted to enhance contrast, and Otsu's thresholding is applied to create a binary image, clearly separating the foreground from the background.
3. Contour Extraction & Rectangle Calculation: Coordinates of the foreground pixels are extracted, and the minimum area rectangle that encloses these points is determined, providing information about the image's tilt.
4. Angle Adjustment & Rotation: The angle of the rectangle is adjusted for manageable rotation, and a rotation matrix is created to deskew the image back to its straight position.

### Defining Text to Speech function 

Code: [sound.py](sound.py)

The text output by the OCR engine is converted to‬ speech by the gTTS library (Google Text-to-Speech). Install packages,
```python
pip install gtts
pip install playsound
```
### Reading the text 

Code: [reader.py](reader.py) 

1. __Color Detection__: Converts each frame to the HSV color space and creates a mask to isolate a specific color range using cv2.inRange(). This mask helps in filtering the image to highlight the finger or other objects of interest.
2. __Contour Detection__: After thresholding the mask to create a binary image, the code identifies contours using cv2.findContours(). This helps in detecting the finger in the image.
3. __Finding the Tip of the Finger__: The code calculates the lowest point of the largest contour, which represents the tip of the finger. It iterates through the contour points to determine the point with the minimum y-coordinate.
4. __Tesseract OCR engine__:
   __Code__: [ocr.py](ocr.py)
   
   It is used to extract the text from th image. Pytesseract or Python-tesseract is an OCR tool for python that also serves as a wrapper for the Tesseract-OCR Engine. The distance between each of the word and finger tip is measured, and the one closest is read.

To install Tesseract OCR Engine:

__For Windows__: Download the Tesseract installer from the [Github Link](https://github.com/tesseract-ocr/tesseract)

__For macOS__: 
```
brew install tesseract
```
__For Ubuntu/Debian__:
```
sudo apt-get update
sudo apt-get install tesseract-ocr
```
To install the python packages to use OCR engine, run:
```python
pip install pytesseract
```
## Object Detection (object_detection.py & object_tiny_detection.py)
 The device has an added mode that helps the user in object detection. The YOLOv4 algorithm open source by : [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) 
 was used for object detection

 *(imp algorithm) The Object to which the user points is to be identified. This would be the object that is closer to the centre of the image. Therefore the algorithm works by outtputing the object that is closer to the centre of the image. 







