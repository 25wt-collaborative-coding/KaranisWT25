#online tutorial with a box around things it doesn't know 

import cv2
import pytesseract
from pytesseract import Output

# Load the image
filename = 'image5.png'
image = cv2.imread(filename)

class Color:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform OCR with bounding box information
results = pytesseract.image_to_data(gray, output_type=Output.DICT)

# Loop through results
for i in range(len(results['text'])):
    # Extract bounding box and confidence
    x = results['left'][i]
    y = results['top'][i]
    w = results['width'][i]
    h = results['height'][i]
    text = results['text'][i]
    conf = int(results['conf'][i])


    # Analyze confidence levels
    if conf < 60:  # Low confidence indicates possible handwriting
        color = (0, 0, 255)  # Red for handwritten
        print(Color.UNDERLINE + text + Color.END, end=" ")
    else:
        color = (0, 255, 0)  # Green for machine-printed
        print(text, end=" ")

    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the processed image
cv2.imshow('Identified Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
