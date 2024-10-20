# Import necessary libraries
import cv2  # OpenCV for computer vision tasks
import numpy as np  # For numerical operations


# Load the YOLOv4-tiny model's weights and configuration file
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4tiny.cfg')

# Load the class names (labels) from the COCO dataset
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()  # "coco.names" contains the object class names

# Open a connection to the webcam (0 is typically the default camera)
cap = cv2.VideoCapture(0)

# Set the font for the labels to be displayed on the output image
font = cv2.FONT_HERSHEY_PLAIN

# Generate random colors for each detection (max 100 colors)
colors = np.random.uniform(0, 255, size=(100, 3))

# Infinite loop to process the video feed frame by frame
while True:
    # Capture the frame from the webcam
    ret, img = cap.read()

    # Get the height, width, and channels of the frame
    height, width, _ = img.shape

    # Preprocess the frame for YOLO by creating a blob from the image
    # Resize the image to 320x320, normalize pixel values to the range [0,1], swap red and blue channels (swapRB=True)
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    
    # Set the input blob for the YOLO network
    net.setInput(blob)

    # Get the names of the output layers for the YOLO model
    output_layers_names = net.getUnconnectedOutLayersNames()

    # Perform a forward pass through the YOLO network to get the output from the layers
    layerOutputs = net.forward(output_layers_names)

    # Initialize lists to hold bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop through each output from the network layers
    for output in layerOutputs:
        # Loop through each detection in the output
        for detection in output:
            scores = detection[5:]  # Get the scores for all object classes
            class_id = np.argmax(scores)  # Get the class with the highest score
            confidence = scores[class_id]  # Get the confidence of the detected class

            # Only consider detections with confidence greater than 0.5 (50%)
            if confidence > 0.5:
                # Scale the bounding box back to the size of the original image
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append the bounding box coordinates, confidence, and class ID to their respective lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to eliminate overlapping boxes with lower confidences
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if any boxes survived NMS and draw them on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]  # Get the bounding box coordinates
            label = str(classes[class_ids[i]])  # Get the class label for the detected object
            confidence = str(round(confidences[i], 2))  # Round the confidence score to 2 decimal places
            color = colors[i]  # Get a random color for the box
            
            # Draw a rectangle around the detected object
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Put the class label and confidence near the bounding box
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (0, 0, 255), 2)

    # Display the processed image with detections, resized for better visibility
    cv2.imshow("Image", cv2.resize(img, (1000, 800)))

    # Wait for a key press and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
