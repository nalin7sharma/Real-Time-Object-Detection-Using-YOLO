import cv2
import numpy as np
import os
import argparse
from time import time
from collections import defaultdict

def main():
    # Configuration management
    parser = argparse.ArgumentParser(description='Real-time Object Detection with YOLO')
    parser.add_argument('--model-cfg', type=str, default='yolov3.cfg',
                       help='Path to YOLO model config file')
    parser.add_argument('--model-weights', type=str, default='yolov3.weights',
                       help='Path to YOLO pretrained weights')
    parser.add_argument('--classes-file', type=str, default='coco.names',
                       help='Path to object class names file')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Minimum probability to filter weak detections')
    parser.add_argument('--nms-threshold', type=float, default=0.4,
                       help='Threshold for non-maximum suppression')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Enable GPU acceleration if available')
    args = parser.parse_args()

    # Validate file paths
    def validate_file(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path

    try:
        model_cfg = validate_file(args.model_cfg)
        model_weights = validate_file(args.model_weights)
        classes_file = validate_file(args.classes_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load YOLO model with error handling
    try:
        net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    except cv2.error as e:
        print(f"Failed to load YOLO model: {e}")
        return

    # Configure backend preferences
    if args.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU acceleration")
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU processing")

    # Load class names with improved error handling
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    class_colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Get output layer names using OpenCV 4.x compatible method
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Initialize video capture with auto-retry
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device")
        return

    # Performance metrics
    frame_count = 0
    total_processing_time = 0
    detection_history = defaultdict(int)

    try:
        while True:
            start_time = time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Create blob from frame with optimized parameters
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (416, 416), 
                swapRB=True, crop=False
            )
            net.setInput(blob)
            layer_outputs = net.forward(output_layers)

            # Process detections
            boxes = []
            confidences = []
            class_ids = []
            h, w = frame.shape[:2]

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > args.confidence_threshold:
                        box = detection[0:4] * np.array([w, h, w, h])
                        (center_x, center_y, width, height) = box.astype("int")
                        x = int(center_x - (width / 2))
                        y = int(center_y - (height / 2))
                        
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, 
                args.confidence_threshold, 
                args.nms_threshold
            )

            # Update detection history and draw results
            current_detections = set()
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    current_detections.add(classes[class_id])
                    
                    # Draw bounding box and label
                    color = class_colors[class_id].tolist()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    label = f"{classes[class_id]}: {confidences[i]:.2f}"
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        frame, (x, y - label_height - baseline),
                        (x + label_width, y),
                        color, -1
                    )
                    cv2.putText(
                        frame, label, (x, y - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                    )

            # Update detection history
            for cls in current_detections:
                detection_history[cls] += 1

            # Calculate and display performance metrics
            processing_time = time() - start_time
            total_processing_time += processing_time
            frame_count += 1
            
            fps = frame_count / total_processing_time
            cv2.putText(
                frame, f"FPS: {fps:.2f} | Objects: {len(current_detections)}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            # Show output
            cv2.imshow('Object Detection', frame)

            # Handle keyboard input
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("\nDetection summary:")
        for cls, count in detection_history.items():
            print(f"{cls}: {count} detections")

if __name__ == "__main__":
    main()
