import cv2
import numpy as np
import os

# Paths to model configuration, weights, and object names
yolo_blueprint = r"C:\Users\LENOVO\.dev\python\Object_Detection\yolov3.cfg"
weights_path = r"C:\Users\LENOVO\.dev\python\Object_Detection\yolov3.weights"
coco_names_path = r"C:\Users\LENOVO\.dev\python\Object_Detection\coco.names"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(yolo_blueprint, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load COCO class names
try:
    with open(coco_names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Error: COCO names file not found.")
    exit()

# Get output layer names
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    print("Error: Unable to retrieve output layers.")
    exit()

# Initialize webcam (use video file path instead of 0 for video input)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Unable to access camera.")
    exit()

def process_frame(photo):
    """Process a single frame for object detection."""
    height, width, channels = photo.shape

    # Create a blob from the input image
    blob = cv2.dnn.blobFromImage(photo, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Lists to hold detected items
    items = []
    confidences = []
    boxes = []

    for detection in detections:
        for item in detection:
            scores = item[5:]
            item_id = np.argmax(scores)
            confidence = scores[item_id]
            if confidence > 0.5:  # Detection threshold
                # Calculate rectangle coordinates
                center_x = int(item[0] * width)
                center_y = int(item[1] * height)
                w = int(item[2] * width)
                h = int(item[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                items.append(item_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[items[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(photo, (x, y), (x + w, y + h), color, 2)
        cv2.putText(photo, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return photo

# Main loop
while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    # Process the frame for object detection
    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Live Object Detection', processed_frame)

    # Break loop on pressing 'n' or close the terminal on pressing 'a'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):  # Stop detection
        break
    elif key == ord('a'):  # Close terminal
        os._exit(0)

# Release resources
camera.release()
cv2.destroyAllWindows()
