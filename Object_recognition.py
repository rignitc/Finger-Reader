import cv2
import numpy as np
from sound import sound  # Using your custom sound function

classes = []
f = open('coco.names', 'r')
font = cv2.FONT_HERSHEY_PLAIN

# Read the class names from coco.names
classes = f.read().splitlines()

# Load YOLOv4 model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
prev_label = ""  # To track the previous label for sound triggering

# Open webcam feed
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()  # Read frame from the camera
    height, width, _ = img.shape  # Get image dimensions

    # Preprocess the image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the names of output layers and perform forward pass
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    centre = []  # To store center coordinates of detected objects
    boxes = []  # To store bounding box sizes
    confidences = []  # To store confidence levels
    class_ids = []  # To store class IDs of detected objects

    # Loop through each detection
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]  # Get confidence scores for all classes
            class_id = np.argmax(scores)  # Get the class with the highest confidence
            confidence = scores[class_id]  # Get the confidence level

            if confidence > 0.2:  # Only consider confident detections
                centre_x = int(detection[0] * width)  # Center x of the object
                centre_y = int(detection[1] * height)  # Center y of the object
                w = int(detection[2] * width)  # Width of the bounding box
                h = int(detection[3] * height)  # Height of the bounding box

                centre.append(np.array((centre_x, centre_y)))  # Append the center coordinates
                class_ids.append(class_id)  # Append the class ID
                boxes.append([w, h])  # Append the bounding box dimensions

    # Fixed center (where you're looking for the closest object)
    b = np.array((width // 2, height // 2))  # Center of the frame

    # Find the closest object to the center of the frame
    min_ = float('inf')  # Set initial minimum distance to a large value
    min_index = -1  # Track the index of the closest object

    for i in range(len(centre)):
        dist = np.linalg.norm(centre[i] - b)  # Calculate Euclidean distance
        if dist < min_:  # Update the closest object
            min_ = dist
            min_index = i

    try:
        # Get the coordinates of the closest object
        obj_x = int(centre[min_index][0] - boxes[min_index][0] / 2)
        obj_y = int(centre[min_index][1] - boxes[min_index][1] / 2)
        obj_w = boxes[min_index][0]
        obj_h = boxes[min_index][1]
        label = str(classes[class_ids[min_index]])  # Get the label of the closest object
        color = (0, 0, 255)  # Red color for the bounding box

        # Draw a rectangle around the closest object
        cv2.rectangle(img, (obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h), color, 2)
        # Add the label to the object
        cv2.putText(img, label, (obj_x, obj_y + 20), font, 2, (255, 0, 0), 2)

        # If the label has changed, play the sound using your function
        if label != prev_label:
            sound(label)  # Trigger the sound function from your file
            prev_label = label  # Update the previous label to the new one

    except:
        pass  # Handle the case when there are no detected objects

    # Display the image with bounding boxes
    cv2.imshow("Image", cv2.resize(img, (1000, 800)))

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
