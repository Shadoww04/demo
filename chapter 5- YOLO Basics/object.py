from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('../Yolo-Weights/yolov8l.pt')

# Perform inference on the image
results = model("images/1.jpg", show=True)

# Set to keep track of already printed object names
detected_objects = set()

# Check if any detections were made
no_detections = True  # Flag to track if any detections occur

for result in results:
    # The result contains the detected boxes and the class indices
    boxes = result.boxes

    if len(boxes) > 0:  # Check if there are any detections
        no_detections = False
        for box in boxes:
            class_id = int(box.cls[0])  # Get the class id of the detected object
            class_name = model.names[class_id]  # Get the class name from the model

            # Only print the object name if it hasn't been printed already
            if class_name not in detected_objects:
                print(f"Detected: {class_name}")
                detected_objects.add(class_name)  # Add the object name to the set

# If no detections were made, display a message
if no_detections:
    print("No objects detected in the image.")
