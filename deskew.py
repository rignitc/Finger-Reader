import sys
import cv2
import numpy as np
img=cv2.imread(r"/home/shameem/RIG/finger_reader/test_images/m20.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=cv2.bitwise_not(gray)         #convert to an inverted gray scale image


thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1] #will set a threshold vlaue by OTSU method 

coords= np.column_stack(np.where(thresh>0))
angle=cv2.minAreaRect(coords)[-1]
if angle< -45:
    angle=-(90+angle)
else:
    angle= 90-angle
(h,w)=img.shape[:2]
center=(w // 2,h // 2)
M=cv2.getRotationMatrix2D(center,angle,1.0)
rotated = cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)


print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow('input',img)
cv2.imshow('rotated',rotated)