#Online tutorial applied to the pdf

from pytesseract import Output
import pytesseract
import cv2
from pdf2image import convert_from_path
import numpy as np

# Path to the PDF file
pdf_filename = 'KaranisSample.pdf'

# Convert PDF to images
pages = convert_from_path(pdf_filename, dpi=300)  # Higher DPI for better OCR accuracy

# Process each page
for page_number, page in enumerate(pages, start=1):
    # Convert PIL Image to NumPy array (compatible with OpenCV)
    image = np.array(page)

    # Convert RGB to BGR (OpenCV uses BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Perform OCR with Tesseract
    results = pytesseract.image_to_data(image, output_type=Output.DICT)
    count = 0
    for i in results['conf']:
        if i > 70:
            count = count + 1
    print(count/len(results['conf']))

    # Iterate through OCR results
    for i in range(len(results['text'])):
        # Extract bounding box and confidence
        x = results['left'][i]
        y = results['top'][i]
        w = results['width'][i]
        h = results['height'][i]
        text = results['text'][i]
        conf = int(results['conf'][i])

        # Draw bounding boxes for high-confidence text
        if conf > 60:
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    # Display the processed page
    window_name = f'Page {page_number}'
    cv2.imshow(window_name, image)

    # Wait for the Escape key (ASCII 27) to be pressed
    while True:
        key = cv2.waitKey(0)  # Wait indefinitely for a key press
        if key == 27:  # Check if the key is ESC
            break

    cv2.destroyAllWindows()  # Close the current window after ESC is pressed
