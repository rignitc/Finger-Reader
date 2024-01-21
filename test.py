from sre_constants import ANY
import cv2
import numpy as np
from sound import sound

classes=[]
f=open('coco.names','r')
font = cv2.FONT_HERSHEY_PLAIN

classes=f.read().splitlines()

net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
prev_label = ""
cap=cv2.VideoCapture(0)
while True:
    _,img=cap.read()
    height,width,_=img.shape
    
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
    x,y=160,160
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    centre=[]
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >0.2:
                centre_x=(detection[0]*width)
                centre_y=(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                

                centre.append(np.array((centre_x,centre_y)))
                class_ids.append(class_id)
                boxes.append([w,h])

    b=np.array((x,y))
    min_ = 1000
    min_index = 0
    for i in range(len(centre)):
        dist = np.linalg.norm(centre[i]-b)
        if dist < min_:
            min_ = dist
            min_index = i
    
    # dist_square=np.sum(dist,axis=1,keepdims=True)
    # min_index=np.argmin(dist_square)
    try:
        obj_x=int(centre[min_index][0]-w/2)
        obj_y=int(centre[min_index][1]-h/2)
        obj_w=boxes[min_index][0]
        obj_h=boxes[min_index][1]
        label=str(classes[class_ids[min_index]])
        color=(0,0,255)
        cv2.rectangle(img,(obj_x,obj_y),(obj_x+obj_w,obj_y+obj_h),color, 2 )
        cv2.putText(img,label, (obj_x,obj_y+20),font,2,(255,0,0),2)
        #new_label : str
        #if new_label != label:
            #new_label=label
            #print(new_label)
        if label != prev_label:
            sound(label)
            prev_label = label
    except:
        pass
    
    


    cv2.imshow("Image",cv2.resize(img, (1000,800)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    
    
    



