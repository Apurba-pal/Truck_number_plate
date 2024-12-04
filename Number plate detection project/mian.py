import pytesseract
from PIL import Image

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path if necessary

# Load the image from the file
image_path = 'test_tesseract.png'  # Replace with your image file path
image = Image.open(image_path)

# Use pytesseract to extract text
extracted_text = pytesseract.image_to_string(image)

# Print extracted text (optional)
print("Extracted Text:\n", extracted_text)

# Save the extracted text to a file
with open('output.txt', 'w') as file:
    file.write(extracted_text)

print("Text has been saved to output.txt")
