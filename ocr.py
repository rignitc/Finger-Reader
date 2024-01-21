import pytesseract
from pytesseract import Output
import cv2
import numpy as np


def read(img):
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # thresh = 127
    # im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    # thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    # coords= np.column_stack(np.where(thresh>0))
    # angle=cv2.minAreaRect(coords)[-1]
    # if angle< -45:
    #     angle=-(90+angle)
    # else:
    #     angle= 90-angle
    # (h,w)=img.shape[:2]
    # center=(w // 2,h // 2)
    # M=cv2.getRotationMatrix2D(center,angle,1.0)
    # rotated = cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)

    
    
    
    n_boxes = len(d['level'])
    new_d = {}
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        new_d[d['text'][i]] = [int(x+w/2), int(y+h/2)]
    return new_d

