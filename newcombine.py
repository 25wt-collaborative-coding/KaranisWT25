import cv2
import pytesseract
import numpy as np
from pytesseract import Output

class TextStyle:
    """Class to handle text styles for terminal output."""
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def preprocess_image(file_name):
    """
    Preprocess the image for improved OCR accuracy.
    Args:
        file_name (str): Path to the input image.
    Returns:
        tuple: The preprocessed grayscale and binary images.
    """
    # Grayscale image
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"The file '{file_name}' could not be loaded as an image.")

    # Denoising and smoothing
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Adaptive thresholding
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 3)
    return image, binary_image

def isolate_handwritten_text_by_contours(binary_image):
    """
    Identify and isolate handwritten text using contour analysis.
    Args:
        binary_image (numpy.ndarray): Binary image for contour analysis.
    Returns:
        list: List of handwritten contours.
    """
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        # Filtering conditions
        if 20 < area < 2000 and compactness < 0.7 and 0.2 < aspect_ratio < 5:
            handwritten_contours.append(contour)
    return handwritten_contours

def process_image_for_ocr(file_name):
    """
    Process an image, applying OCR to detect text while identifying handwritten text.
    Args:
        file_name (str): Path to the input image.
    Returns:
        None
    """
    # Step 1: Preprocess the image
    gray_image, binary_image = preprocess_image(file_name)

    # Step 2: Isolate handwritten contours
    handwritten_contours = isolate_handwritten_text_by_contours(binary_image)

    # Step 3: OCR processing
    results = pytesseract.image_to_data(gray_image, output_type=Output.DICT)

    # Prepare final image
    final_image = cv2.imread(file_name)
    final_transcript = []

    for i in range(len(results["text"])):
        x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
        text = results["text"][i].strip()
        conf = int(results["conf"][i])

        if not text:
            continue

        # Determine if the text overlaps with handwritten contours
        is_handwritten = False
        for contour in handwritten_contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            if x >= cx and y >= cy and x + w <= cx + cw and y + h <= cy + ch:
                is_handwritten = True
                break

        # Handle confidence levels and classification
        if is_handwritten:
            color = (0, 0, 255)  # Red for handwritten
            final_transcript.append(f"{TextStyle.BOLD}{TextStyle.UNDERLINE}{text}{TextStyle.END}")
        elif conf > 50:
            color = (0, 255, 0)  # Green for high-confidence machine-printed text
            final_transcript.append(text)
        else:
            color = (255, 0, 0)  # Blue for low-confidence text
            final_transcript.append(f"{TextStyle.BOLD}{TextStyle.UNDERLINE}{text}{TextStyle.END}")

        # Draw bounding box
        cv2.rectangle(final_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(final_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Step 4: Draw bounding boxes for handwritten text
    for contour in handwritten_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for handwritten

    # Print transcript
    print("\n=== Final Transcript ===")
    print(" ".join(final_transcript))

    # Display results
    cv2.imshow("Original Image", cv2.imread(file_name))
    cv2.imshow("Binary Image", binary_image)
    cv2.imshow("Final Image with OCR and Handwritten Identification", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run the process
filename1 = 'image6.png'
filename2 = 'image5.png'

process_image_for_ocr(filename1)
process_image_for_ocr(filename2)
