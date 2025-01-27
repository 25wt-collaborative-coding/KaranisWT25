import cv2
import numpy as np
import tensorflow as tf
import pytesseract

letter_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# Load your pre-trained model
model = tf.keras.models.load_model('text_recognition_model.keras')

# Define constants (same as in the training code)
TARGET_HEIGHT = 32
TARGET_WIDTH = 128

def resize_image(image, target_height, target_width):
    """
    Resizes an image to the specified height while maintaining aspect ratio,
    then pads or crops it to fit the specified width.
    """
    h, w = image.shape[:2]  # Get the original height and width of the image
    scale = target_height / h  # Calculate the scaling factor based on height
    new_width = int(w * scale)  # Calculate the new width based on aspect ratio

    # Resize the image to maintain the aspect ratio with the target height
    resized = cv2.resize(image, (new_width, target_height))

    # If the new width is larger than the target width, crop the image
    if new_width > target_width:
        start_x = (new_width - target_width) // 2  # Crop from the center
        resized = resized[:, start_x:start_x + target_width]
    elif new_width < target_width:
        # Pad the image if the new width is smaller than the target width
        pad_width = target_width - new_width
        padded = cv2.copyMakeBorder(resized, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0)
        return padded
    else:
        return resized


letters = []

# Load the image
img = cv2.imread("image13.png")

# Create a copy of the original image for drawing purposes
img_copy = img.copy()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image (optional, might remove this step for alignment)
# This is not part of your model's preprocessing pipeline, so you can comment this out if needed
# thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Use pytesseract to get character bounding boxes, NOT predictions
items = pytesseract.image_to_boxes(img)

detected = ""

# Iterate over the bounding boxes detected by pytesseract
for detail in items.splitlines():
    values = detail.split()
    letter, x1, y1, x2, y2 = values[:5]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Pytesseract uses (0,0) at bottom-left; invert y-coordinates for OpenCV
    y1 = img.shape[0] - y1
    y2 = img.shape[0] - y2

    # Add padding to the bounding box
    padding = 2  # Adjust this padding value as needed
    x1 = max(0, x1 - padding)  # Ensure x1 doesn't go out of bounds
    x2 = min(img.shape[1], x2 + padding)  # Ensure x2 doesn't go out of bounds
    # Add extra padding to the bounding box height
    vertical_padding = -5  # Adjust this value as needed to include more height
    y1 = max(0, y1 - vertical_padding)  # Extend upwards
    y2 = min(img.shape[0], y2 + vertical_padding)  # Extend downwards

    width = x2 - x1
    height = y2 - y1

    #Dont count commas, noise, etc (small bounding box)
   #if width < 20 and height < 30:
       #continue

    # Extract the letter image from the bounding box
    letter_img = img[y2:y1, x1:x2]  # Invert y-coordinates for slicing

    # Preprocess the letter image: Convert to grayscale and normalize
    gray_letter_img = cv2.cvtColor(letter_img, cv2.COLOR_BGR2GRAY)

    # Resize the letter image to match the model's input size (32x128)
    resized_letter_img = resize_image(gray_letter_img, TARGET_HEIGHT, TARGET_WIDTH)

    # Normalize the resized letter image by dividing by 255
    resized_letter_img = resized_letter_img / 255.0

    # Reshape the image to match the model's expected input shape
    resized_letter_img = np.expand_dims(resized_letter_img, axis=-1)  # Add channel dimension (grayscale)
    resized_letter_img = np.expand_dims(resized_letter_img, axis=0)  # Add batch dimension (1 image)

    # Visualize each resized letter image before passing it to the model
    cv2.imshow("Letter Image", resized_letter_img[0, :, :, 0])  # Show the grayscale image
    cv2.waitKey(0)  # Wait for key press to move to the next letter

    # Predict the letter using your model (not pytesseract)
    prediction = model.predict(resized_letter_img)
    predicted_char = letter_list[np.argmax(prediction)]  # From your model
    detected += predicted_char  # Append only model predictions
    print(f"Bounding box: {x1, y1, x2, y2}")
    print(f"Model prediction probabilities: {prediction}")
    print(f"Predicted character: {predicted_char}")

    color = (0, 0, 255)
    cv2.rectangle(img_copy, (x1, img.shape[0] - y1), (x2, img.shape[0] - y2), color, 2)  # Adjust y-coordinates for display
    # Annotate the original image with the predicted letter
    cv2.putText(img_copy, predicted_char, (x1, img.shape[0] - y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# Print the final detected text (model predictions only)
print("Detected: " + detected)

# Show the image with bounding boxes and annotated predicted letters
cv2.imshow("Detected Letters", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
