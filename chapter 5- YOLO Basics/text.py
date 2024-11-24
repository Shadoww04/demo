import cv2
import pytesseract

# Path to the image file
image_path = 'images/text5.png'

# Load the image using OpenCV
img = cv2.imread(image_path)

# Check if the image is loaded successfully
if img is None:
    print(f"Error: Unable to load image at {image_path}. Please check the path.")
else:
    print("Image loaded successfully.")

    # Step 1: Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply thresholding to improve OCR accuracy
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Step 3: Optional: Apply noise reduction (blurring) for better OCR results
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Step 4: Use pytesseract to extract data with Page Segmentation Mode (PSM)
    custom_config = r'--psm 3'  # Use PSM 3 (fully automatic page segmentation)

    # Extract data with pytesseract's image_to_data() (provides word-level details)
    data = pytesseract.image_to_data(blurred, config=custom_config, output_type=pytesseract.Output.DICT)

    # Step 5: Loop through the data and draw rectangles around words
    num_items = len(data['text'])
    for i in range(num_items):
        if int(data['conf'][i]) > 0:  # Only draw boxes around words with confidence > 0
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # Draw rectangle around the word
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color rectangle


    # Step 6: Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(blurred)

    # Output the extracted text
    print("Extracted Text:")
    print(text)


    # Step 7: Display the processed image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
