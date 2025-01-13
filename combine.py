import cv2
import pytesseract
import os
import numpy as np
from pytesseract import Output

def isolate_handwritten_text_by_contours(file_name):
    """
    Isolate handwritten text from an image by analyzing contours for irregularity and non-uniformity.

    Args:
        file_name (str): Path to the input image.

    Returns:
        tuple: Processed image with handwritten contours highlighted, and a list of identified contours.
    """
    # Load the image in grayscale
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"The file '{file_name}' could not be loaded as an image.")

    # Step 1: Preprocessing - Adaptive Threshold
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 3)

    # Step 2: Find Contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 3: Filter Contours Based on Irregularity and Uniformity
    handwritten_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:  # Avoid division by zero
            continue

        # Irregularity metrics
        compactness = (4 * np.pi * area) / (perimeter**2)  # Circularity metric
        bounding_rect = cv2.boundingRect(contour)
        aspect_ratio = bounding_rect[2] / bounding_rect[3] if bounding_rect[3] != 0 else 0

        # Filtering conditions:
        # - Small to medium-sized areas
        # - Irregular contours (low compactness)
        # - Non-uniform aspect ratios
        if 20 < area < 2000 and compactness < 0.7 and 0.2 < aspect_ratio < 5:
            handwritten_contours.append(contour)

    # Step 4: Highlight Handwritten Contours
    handwritten_mask = np.zeros_like(binary_image)
    cv2.drawContours(handwritten_mask, handwritten_contours, -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original grayscale image
    handwritten_text_image = cv2.bitwise_and(image, image, mask=handwritten_mask)

    return handwritten_text_image, handwritten_contours


# Main Processing
filename = 'image5.png'
image = cv2.imread(filename)
if image is None:
    raise FileNotFoundError(f"The file '{filename}' could not be loaded.")

# Isolate handwritten text using contour analysis
handwritten_image, handwritten_contours = isolate_handwritten_text_by_contours(filename)

# Visualize the contours on the original image
# Perform OCR with pytesseract
results = pytesseract.image_to_data(image, output_type=Output.DICT)

# Iterate through OCR results
for i in range(0, len(results['text'])):
    #draw a box around the word
    x = results['left'][i]
    y = results['top'][i]
    w = results['width'][i]
    h = results['height'][i]
    text = results['text'][i]
    conf = int(results['conf'][i])
    print(conf)

    # Draw rectangles and display text for high-confidence results
    if conf > 50:
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        
contoured_image = image.copy()
for contour in handwritten_contours:
    x, y, w, h = cv2.boundingRect(contour)
    #print(x*h)
    print(w*h)
    if (w*h) > 650:
        cv2.rectangle(contoured_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for machine text
    else:
        cv2.rectangle(contoured_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for handwritten

# Display the Results
cv2.imshow("Original Image", image)
cv2.imshow("Isolated Handwritten Text", handwritten_image)
cv2.imshow("Contours Highlighted", contoured_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



''' To implement:
collect words (black boxes) and count green/red boxes inside. If the majority of boxes are green, recognize as 
machine text and report the word back. Otherwise, recognize as handwritten and flag. Testing.py has sort of
what this should look like in the end. '''
