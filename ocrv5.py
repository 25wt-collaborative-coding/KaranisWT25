import cv2
import pytesseract
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model("text_recognition_model.keras")

# Define constants
TARGET_HEIGHT = 32
TARGET_WIDTH = 128
letter_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# Label encoder for decoding predictions
label_encoder = LabelEncoder()
label_encoder.fit(letter_list)

def resize_image(image, target_height, target_width):
    h, w = image.shape
    scale = target_height / h
    new_width = int(w * scale)
    resized = cv2.resize(image, (new_width, target_height))

    if new_width < target_width:
        pad_width = target_width - new_width
        padded = cv2.copyMakeBorder(resized, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0)
        return padded
    else:
        return resized[:, :target_width]

def preprocess_image(input_image):
    """
    Preprocesses the input image (converts to grayscale, applies binarization and noise reduction).
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding to increase contrast
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to remove noise (optional)
    kernel = np.ones((3, 3), np.uint8)
    denoised_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
    
    return denoised_image

def predict_letter(letter_image):
    """
    Predicts the letter from the preprocessed image using the trained model.
    """
    normalized_image = letter_image / 255.0  # Normalize to [0, 1]
    input_image = np.expand_dims(normalized_image, axis=(0, -1))  # Add batch and channel dimensions
    prediction = model.predict(input_image)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def recognize_text_from_image(image_path):
    """
    Recognizes text from an input image using pytesseract for segmentation and the trained model for recognition.
    """
    # Read the input image
    img_bgr = cv2.imread(image_path)
    
    # Preprocess the image for better segmentation
    preprocessed_img = preprocess_image(img_bgr)
    
    # Use pytesseract to detect individual characters' bounding boxes
    # We are using a slightly different config for better segmentation
    details = pytesseract.image_to_boxes(preprocessed_img, config='--psm 6')  # psm 6: Assume a single uniform block of text
    
    recognized_text = ""
    
    # Loop over each detected character
    for detail in details.splitlines():
        print(f"Detected Detail: {detail}")  # Print the detail to inspect its structure
        
        try:
            # Split the detail by space and ensure there are only 5 elements
            parts = detail.split()
            
            # Ensure there are exactly 5 parts before unpacking
            if len(parts) == 5:
                char, x1, y1, x2, y2 = parts
            else:
                # If there are extra values, ignore them
                char, x1, y1, x2, y2 = parts[:5]
            
            # Convert the coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract the letter region from the image
            letter_image = img_bgr[y1:y2, x1:x2]
            
            # Preprocess the letter
            preprocessed_image = preprocess_image(letter_image)
            
            # Resize the letter image to the model's expected input size
            resized_image = resize_image(preprocessed_image, TARGET_HEIGHT, TARGET_WIDTH)
            
            # Predict the letter using the trained model
            predicted_letter = predict_letter(resized_image)
            
            # Append the predicted letter to the result
            recognized_text += predicted_letter
        except ValueError as e:
            print(f"Skipping invalid box: {detail} ({e})")
    
    return recognized_text

# Example usage
if __name__ == "__main__":
    image_path = "./image8.png"  # Replace with your image file path
    recognized_text = recognize_text_from_image(image_path)
    print(f"Recognized Text: {recognized_text}")
