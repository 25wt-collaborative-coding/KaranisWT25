import cv2
import pytesseract
from pytesseract import Output
import numpy as np

def preprocess_and_extract_text(filename):
    """
    Preprocess the image using OpenCV for improved OCR text extraction.

    Args:
        filename (str): Path to the input image.

    Returns:
        None
    """
    # Load the image
    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"Unable to load the image file: {filename}")

    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Noise Reduction (Gaussian Blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Adaptive Thresholding (to handle varying backgrounds)
    binary_image = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )

    # Step 4: Morphological Operations (to clean up noise)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Step 5: Perform OCR using pytesseract
    results = pytesseract.image_to_data(cleaned, output_type=Output.DICT)

    # Step 6: Draw Bounding Boxes and Display Extracted Text
    for i in range(len(results["text"])):
        # Extract OCR data
        x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
        text = results["text"][i].strip()
        conf = int(results["conf"][i])

        # Skip if text is empty or confidence is low
        if not text or conf < 56:
            continue

        # Print extracted text
        print(f"Text: {text} | Confidence: {conf}")

        # Draw a bounding box around detected text
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the original image with bounding boxes
    cv2.imshow("Processed Image with OCR Results", image)
    cv2.imshow("Preprocessed Binary Image", cleaned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Test the function with an image file
filename = "image5.png"
preprocess_and_extract_text(filename)
