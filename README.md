# Finger-Reader
## Hardware

![alt text](Images/Screenshot%20from%202024-02-06%2021-31-54-modified.png)

* Camera 
* 3D printed camera mount
* Raspberry Pi (3 B+ model)
* Wrist casing for Raspberry Pi
* Speaker

## Working

![alt text](Images/Screenshot%20from%202024-02-06%2021-32-05-modified.png)

The Python source code is run on the Raspberry Pi. OpenCV library is used to enhance the text in the‬
image and the Pytesseract library (OCR engine wrapper) extracts text from images. The text is converted to‬
speech by the gTTS library (Google Text-to-Speech)

### Image Processing (deskew.py)

![alt text](Images/Screenshot%20from%202024-02-06%2021-33-59-modified.png)

Certain image processing techniques need to be done, to obtain good results from OCR. These are used to enhance the text in the image. The following operation are done:
* Gray Scale Conversion : Gray Scale Conversion converts the RGB colour scale into gray level intensity image. The converted image will have 256 pixel levels.
* Binarization: Image binarization is a method of transforming a gray scale image to black (0) and white (255) pixel values.
* De-skewing: It is the process of correcting the twists (skew) of the text in an image. This technique rotates the image such that text is vertically aligned. In this technique, firstly the text block in an image is detected then the algorithm computes the angel of the rotated text and finally the image is rotated to correct the skew.

### Reading the text (reader.py)

* The Tesseract OCR engine is used to extract the text from th image. Pytesseract or Python-tesseract is an OCR tool for python that also serves as a wrapper for the Tesseract-OCR Engine.
* (imp algorithm) The word to which the finger points is to be read. To detect the end point of the finger contour in python is used and the central point of the contour is defined. The word closest to the center point is read.

### Text to Speech(sound.py)
* The text output by the OCR engine is converted to‬ speech by the gTTS library (Google Text-to-Speech).

## Object Detection (object_detection.py & object_tiny_detection.py)
 The device has an added mode that helps the user in object detection. The YOLOv4 algorithm open source by : [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) 
 was used for object detection

 *(imp algorithm) The Object to which the user points is to be identified. This would be the object that is closer to the centre of the image. Therefore the algorithm works by outtputing the object that is closer to the centre of the image. 







