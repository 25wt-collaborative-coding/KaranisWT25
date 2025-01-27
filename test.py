import pytesseract
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your trained model
model = load_model('text_recognition_model.keras')

# Load image
image_path = 'image.png'
image = cv2.imread(image_path)

# Convert to grayscale for pytesseract to process
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding or edge detection if necessary
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Get bounding boxes for text using pytesseract (without OCR prediction)
boxes = pytesseract.image_to_boxes(thresh)

# Prepare for drawing and recognizing each box
letter_predictions = []
for box in boxes.splitlines():
    b = box.split()
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    
    # Crop the region for the letter
    letter_img = thresh[y:h, x:w]
    
        # Resize letter image to match model's input
    letter_img_resized = cv2.resize(letter_img, (128, 128))  # Resize to expected dimensions
    letter_img_resized = letter_img_resized.astype('float32') / 255.0  # Normalize
    letter_img_resized = np.expand_dims(letter_img_resized, axis=-1)  # Add channel dimension
    letter_img_resized = np.expand_dims(letter_img_resized, axis=0)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(letter_img_resized)
    
    # Get the character prediction (e.g., class with highest probability)
    predicted_char = np.argmax(prediction)
    letter_predictions.append(predicted_char)

    # Optionally, draw the bounding box around each letter on the image
    cv2.rectangle(image, (x, h), (w, y), (0, 255, 0), 2)  # Green box
    
# Now print the predictions
print("Predicted characters:", letter_predictions)

# Show the image with bounding boxes (for visualization)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Turn off axis numbers
plt.show()
