import cv2
import numpy as np
import time 
rect=[]
#net = cv2.dnn.readNet('yolov3D_training_last.weights', 'yolov3D_testing.cfg')
#net = cv2.dnn.readNet('yolov3E_training_last.weights', 'yolov3E_testing.cfg')
#net = cv2.dnn.readNet('yolov3SSS_training_last.weights', 'yolov3S_testing.cfg')
#net = cv2.dnn.readNet('custom-yolov4S-tiny-detector_best.weights', 'custom-yolov4S-tiny-detector.cfg')
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4tiny.cfg')
#net = cv2.dnn.readNet("yolo20frames/yolov3-spp.weights", "yolo20frames/yolov3-spp.cfg")
classes = []
with open("coco.names", "r") as f:      
    classes = f.read().splitlines()   # "coco.names" # "classes.txt"
#####
'''
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
#t1=time.time()

img = cv2.imread("111.jpg")
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

if len(indexes)>0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        rect.append((x,y,w,h))
        label = str(classes[class_ids[i]])
        print(type(label),x,y,w,h)
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,0,255), 2)
# t2=time.time()
# print(t2-t1)
#cv2.imshow('Image', img)
# thresh=1
# faces = face_cascade.detectMultiScale(img, 1.3, 5)
# print(faces)
print(rect)
_,net_rect=cv2.groupRectangles(rect, 2)
print(net_rect)
cv2.imshow("Image",cv2.resize(img, (600,500)))
#cv2.waitKey(1)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

'''

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    #t1=time.time()
    _, img = cap.read()
    height, width, _ = img.shape

    #blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (0,0,255), 2)
    # t2=time.time()
    # print(t2-t1)
    #cv2.imshow('Image', img)
    cv2.imshow("Image",cv2.resize(img, (1000,800)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()