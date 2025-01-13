#Online tutorial followed verbatum

from PIL import Image
import pytesseract
import numpy as np
import cv2

# Load the first image
filename = 'image.png'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # Read as grayscale to process correctly
if img is None:
    raise FileNotFoundError(f"File '{filename}' not found or could not be read.")

# Normalize the image
norm_img = np.zeros((img.shape[0], img.shape[1]))  # Create a blank image of the same size
img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

# Apply threshold and blur
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)  # Apply binary threshold
img = cv2.GaussianBlur(img, (1, 1), 0)  # Apply Gaussian Blur

# Use Tesseract to extract text
text = pytesseract.image_to_string(img)
print(f"Text from {filename}:\n{text}")

# Load the second image
filename = 'image1.png'
img2 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
if img2 is None:
    raise FileNotFoundError(f"File '{filename}' not found or could not be read.")

# Use Tesseract to extract text from the second image without preprocessing
text = pytesseract.image_to_string(img2)
print(f"Text from {filename}:\n{text}")