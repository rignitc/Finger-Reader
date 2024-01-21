import cv2
import numpy as np
from ocr import read
import math
from sound import sound

cap=cv2.VideoCapture(0)

def sqrt_d(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)
voice=None

while True:
    

        _,img=cap.read()
        
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv_min = np.array([0, 30, 60])
        hsv_max = np.array([20, 150, 255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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


        ret, thresh = cv2.threshold(mask, 2, 255, 0)

        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c=max(contours,key=cv2.contourArea)

        
         

        
        m = 1000
        ans = None
        for i in c:
            for j in i:
            
                if j[1] < m:
                    m = j[1]
                    ans = j[0]
                    ans1=j[1]

        cx, cy = ans,ans1
        min_dist = 1000
        final_text = None
    
        for text, center in read(img).items():
            dist = sqrt_d(cx, cy, center[0], center[1])
            if dist < min_dist:
                min_dist = dist
                final_text = text
        
        try:
        
            if voice != final_text:
                sound(final_text)
                print(final_text)
                voice=final_text
        except:
            pass



        cv2.imshow('mask',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
   