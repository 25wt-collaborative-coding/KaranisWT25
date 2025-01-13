#Online tutorial but special

from pytesseract import Output
import pytesseract
import cv2

filename = 'image5.png'
image = cv2.imread(filename)

# Verify the image is loaded
if image is None:
    raise FileNotFoundError(f"File '{filename}' not found or could not be read.")

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
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

cv2.imshow('Image Window', image)

# Wait for the Escape key (ASCII 27) to be pressed
while True:
    key = cv2.waitKey(0)  # Wait indefinitely for a key press
    if key == 27:  # Check if the key is ESC
        break

cv2.destroyAllWindows()  # Close all OpenCV windows
