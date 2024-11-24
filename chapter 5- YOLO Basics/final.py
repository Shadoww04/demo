# import cv2
# import pytesseract
# from ultralytics import YOLO
#
# # Path to the image file
# image_path = 'images/7.jpeg'  # Make sure this path is correct
#
# # Load the image using OpenCV
# img = cv2.imread(image_path)
#
# # Check if the image is loaded successfully
# if img is None:
#     print(f"Error: Unable to load image at {image_path}. Please check the path.")
# else:
#     print("Image loaded successfully for OCR and YOLO.")
#
#     # Step 1: Convert the image to grayscale for OCR
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Step 2: Apply thresholding to improve OCR accuracy
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
#
#     # Step 3: Apply Gaussian Blur to reduce noise for better OCR results
#     blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
#
#     # Step 4: Use pytesseract to extract data with Page Segmentation Mode (PSM)
#     custom_config = r'--psm 3'  # Use PSM 3 (fully automatic page segmentation)
#
#     # Extract data with pytesseract's image_to_data() to get word-level details
#     data = pytesseract.image_to_data(blurred, config=custom_config, output_type=pytesseract.Output.DICT)
#
#     # Step 5: Draw bounding boxes around recognized words (OCR)
#     num_items = len(data['text'])
#     for i in range(num_items):
#         if int(data['conf'][i]) > 80:  # Only draw boxes around words with confidence > 0
#             x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#             # Draw rectangle around the word (green color)
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
#
#     # Step 6: Output the extracted text from OCR
#     text = pytesseract.image_to_string(blurred)
#     print("Extracted Text from OCR:")
#     print(text)
#
#     # Step 7: Perform Object Detection with YOLO
#     # Load the YOLOv8 model
#     model = YOLO('../Yolo-Weights/yolov8l.pt')  # Adjust the path to your model file
#
#     # Set a confidence threshold for object detection
#     confidence_threshold = 0.5  # Only consider detections with confidence above 50%
#
#     # Perform inference on the image (with confidence threshold)
#     results = model(image_path, conf=confidence_threshold, show=True)  # show=True to display the image
#
#     # Step 8: Set to keep track of already printed object names (for YOLO)
#     detected_objects = set()
#
#     # Step 9: Get the class names for the detected objects from YOLO
#     for result in results:
#         # The result contains the detected boxes and the class indices
#         boxes = result.boxes
#         for box in boxes:
#             confidence = box.conf[0]  # Get the confidence score of the detection
#             if confidence >= confidence_threshold:  # Filter based on the threshold
#                 class_id = int(box.cls[0])  # Get the class id of the detected object
#                 class_name = model.names[class_id]  # Get the class name from the model
#
#                 # Only print the object name if it hasn't been printed already
#                 if class_name not in detected_objects:
#                     print(f"Detected Object (YOLO): {class_name} with confidence {confidence:.2f}")
#                     detected_objects.add(class_name)  # Add the object name to the set
#
#     # Step 10: Display the final processed image (with both OCR and YOLO bounding boxes)
#     cv2.imshow('Combined Image with OCR and YOLO', img)  # Display the image with both OCR and YOLO boxes
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


import os
import cv2
import pytesseract
from ultralytics import YOLO

# --- CONFIGURATION ---
IMAGE_PATH = 'images/7.jpeg'  # Path to the input image
YOLO_MODEL_PATH = '../Yolo-Weights/yolov8l.pt'  # Path to YOLO model weights
OCR_CONFIDENCE_THRESHOLD = 80  # Minimum confidence for OCR text recognition
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for YOLO object detection
OUTPUT_IMAGE_PATH = 'output/processed_image.jpg'  # Path to save the processed image

# --- FUNCTION DEFINITIONS ---
def preprocess_image_for_ocr(image):
    """Convert the image to grayscale, apply thresholding, and Gaussian blur for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    return blurred

def extract_text_with_ocr(image, confidence_threshold):
    """Extract text using pytesseract and draw bounding boxes around high-confidence words."""
    config = r'--psm 3'
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    extracted_text = ""
    for i in range(len(data['text'])):
        if int(data['conf'][i]) >= confidence_threshold:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
            extracted_text += data['text'][i] + " "
    return extracted_text.strip()

def detect_objects_with_yolo(model, image_path, confidence_threshold):
    """Detect objects in an image using YOLO and return a set of detected object names."""
    results = model(image_path, conf=confidence_threshold, show=False)  # Run inference
    detected_objects = set()
    for result in results:
        for box in result.boxes:
            if box.conf[0] >= confidence_threshold:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_objects.add((class_name, float(box.conf[0])))
    return detected_objects

# --- MAIN PROGRAM ---
if __name__ == "__main__":
    # Step 1: Load the image
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}. Please check the path.")
        exit(1)
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Failed to load image at {IMAGE_PATH}.")
        exit(1)
    print("Image loaded successfully.")

    # Step 2: Preprocess the image for OCR
    preprocessed_img = preprocess_image_for_ocr(img)

    # Step 3: Perform OCR and display results
    extracted_text = extract_text_with_ocr(preprocessed_img, OCR_CONFIDENCE_THRESHOLD)
    print("Extracted Text from OCR:")
    print(extracted_text)

    # Step 4: Load the YOLO model
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: YOLO model weights not found at {YOLO_MODEL_PATH}.")
        exit(1)
    model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded successfully.")

    # Step 5: Perform object detection with YOLO
    detected_objects = detect_objects_with_yolo(model, IMAGE_PATH, YOLO_CONFIDENCE_THRESHOLD)
    print("Detected Objects with YOLO:")
    for obj, confidence in detected_objects:
        print(f"{obj} (confidence: {confidence:.2f})")

    # Step 6: Save and display the final image
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE_PATH, img)
    print(f"Processed image saved at {OUTPUT_IMAGE_PATH}.")
    cv2.imshow("Processed Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
