import cv2
import numpy as np
import os
import tempfile
import streamlit as st
from collections import defaultdict

def main():
    st.title("Real-Time Object Detection with YOLO and Streamlit")
    
    # Sidebar configuration
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.4, 0.01)
    use_gpu = st.sidebar.checkbox("Use GPU Acceleration", value=False)
    
    # File uploaders
    st.sidebar.header("Model Configuration")
    model_cfg = st.sidebar.file_uploader("YOLO Config File (.cfg)", type=["cfg"])
    model_weights = st.sidebar.file_uploader("YOLO Weights File (.weights)", type=["weights"])
    classes_file = st.sidebar.file_uploader("Class Names File (.names)", type=["names"])
    
    # Initialize session state
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = defaultdict(int)
    if 'net' not in st.session_state:
        st.session_state.net = None
    if 'classes' not in st.session_state:
        st.session_state.classes = []
    if 'class_colors' not in st.session_state:
        st.session_state.class_colors = []

    # Handle model loading
    if model_cfg and model_weights and classes_file:
        try:
            # Save uploaded files to temporary files
            with tempfile.NamedTemporaryFile(delete=False) as cfg_file, \
                 tempfile.NamedTemporaryFile(delete=False) as weights_file, \
                 tempfile.NamedTemporaryFile(delete=False) as names_file:
                
                cfg_file.write(model_cfg.read())
                weights_file.write(model_weights.read())
                names_file.write(classes_file.read())
                
                cfg_path = cfg_file.name
                weights_path = weights_file.name
                names_path = names_file.name

            # Load network
            net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            
            # Configure GPU if requested
            if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Load class names and colors
            with open(names_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            class_colors = np.random.uniform(0, 255, size=(len(classes), 3))
            
            # Update session state
            st.session_state.net = net
            st.session_state.classes = classes
            st.session_state.class_colors = class_colors
            
            # Cleanup temporary files
            os.remove(cfg_path)
            os.remove(weights_path)
            os.remove(names_path)
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
    else:
        st.warning("Please upload all model files to continue.")
        return

    # Detection controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Detection") and not st.session_state.detection_active:
            st.session_state.detection_active = True
    with col2:
        if st.button("Stop Detection") and st.session_state.detection_active:
            st.session_state.detection_active = False

    # Main detection loop
    if st.session_state.detection_active:
        video_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        
        try:
            while st.session_state.detection_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video")
                    break

                # YOLO detection implementation
                blob = cv2.dnn.blobFromImage(
                    frame, 1/255.0, (416, 416), 
                    swapRB=True, crop=False
                )
                st.session_state.net.setInput(blob)
                
                # Get output layers
                layer_names = st.session_state.net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in st.session_state.net.getUnconnectedOutLayers()]
                layer_outputs = st.session_state.net.forward(output_layers)

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

                        if confidence > confidence_threshold:
                            box = detection[0:4] * np.array([w, h, w, h])
                            (center_x, center_y, width, height) = box.astype("int")
                            x = int(center_x - (width / 2))
                            y = int(center_y - (height / 2))
                            
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # Apply NMS
                indices = cv2.dnn.NMSBoxes(
                    boxes, confidences, 
                    confidence_threshold, 
                    nms_threshold
                )

                # Draw detections and update history
                current_detections = set()
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        class_id = class_ids[i]
                        current_detections.add(st.session_state.classes[class_id])
                        
                        # Draw bounding box and label
                        color = st.session_state.class_colors[class_id].tolist()
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        
                        label = f"{st.session_state.classes[class_id]}: {confidences[i]:.2f}"
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
                    st.session_state.detection_history[cls] += 1

                # Convert to RGB for Streamlit display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame, channels="RGB")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    # Display detection history
    st.sidebar.header("Detection Summary")
    for cls, count in st.session_state.detection_history.items():
        st.sidebar.text(f"{cls}: {count} detections")

if __name__ == "__main__":
    main()
