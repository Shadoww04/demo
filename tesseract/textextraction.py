from PIL import Image
import pytesseract

# Path to the image you want to process
image_path = ''

# Open the image using Pillow
img = Image.open(image_path)

# Use pytesseract to extract text
text = pytesseract.image_to_string(img)

# Print the extracted text
print("Extracted Text:")
print(text)
