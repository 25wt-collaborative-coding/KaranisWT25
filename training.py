import cv2
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define constants
TARGET_HEIGHT = 32
TARGET_WIDTH = 128
letter_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


# Resize function
def resize_image(image, target_height, target_width):
    """
    Resizes an image to the specified height while maintaining aspect ratio,
    then pads or crops it to fit the specified width.
    """
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

# Prepare dataset
train_images, train_labels = [], []
test_images, test_labels = [], []

for letter in letter_list:
    path = f"./dataset/{letter}"
    if not os.path.exists(path):
        print(f"Directory {path} does not exist. Skipping.")
        continue

    dir_list = os.listdir(path)

    # Remove system files like ".DS_Store"
    dir_list = [file for file in dir_list if not file.startswith(".")]

    # Shuffle the file list for randomness
    random.shuffle(dir_list)

    # Calculate the split index
    split_index = 8

    # Split the data into training and test sets
    train_files = dir_list[split_index:]
    test_files = dir_list[:split_index]

    # Process training files
    for file_name in train_files:
        image_path = os.path.join(path, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Could not read {image_path}. Skipping.")
            continue

        # Resize and normalize
        resized_image = resize_image(image, TARGET_HEIGHT, TARGET_WIDTH)
        normalized_image = resized_image / 255.0

        # Add to training set
        train_images.append(normalized_image)
        train_labels.append(letter)

    # Process test files
    for file_name in test_files:
        image_path = os.path.join(path, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Could not read {image_path}. Skipping.")
            continue

        # Resize and normalize
        resized_image = resize_image(image, TARGET_HEIGHT, TARGET_WIDTH)
        normalized_image = resized_image / 255.0

        # Add to test set
        test_images.append(normalized_image)
        test_labels.append(letter)

# Convert lists to NumPy arrays
train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Print summary
print(f"Training set: {len(train_images)} images")
print(f"Test set: {len(test_images)} images")

#CNN
def create_model(input_shape, num_classes):
    model = models.Sequential([
        # Convolutional and pooling layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),

        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (32, 128, 1)  # (Height, Width, Channels)
num_classes = len(letter_list)  # Number of classes (26 letters)
model = create_model(input_shape, num_classes)
model.summary()

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Reshape images
train_images = train_images[..., np.newaxis]  # Add channel dimension
test_images = test_images[..., np.newaxis]

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=5,        # Rotate images slightly
    width_shift_range=0.1,   # Shift images horizontally (10% of the width)
    height_shift_range=0.1,  # Shift images vertically (10% of the height)
    shear_range=0.1,         # Shear transformations
    zoom_range=0.1           # Zoom in/out
)

# Normalize images directly in the generator
datagen.fit(train_images)

# Fit the model using the augmented data, run the model
history = model.fit(
    datagen.flow(train_images, train_labels_encoded, batch_size=32),  # Augmented data generator
    epochs=175, 
    validation_data=(test_images, test_labels_encoded)  # Validation data remains unaugmented
)


# Run model on all test images
for i in range(len(test_images)-1):
    sample_image = test_images[i]  
    sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension
    prediction = model.predict(sample_image)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    print(f"Predicted Label: {predicted_label[0]}")
    print(f"Actual Label: {test_labels[i]}")

test_loss, test_accuracy = model.evaluate(test_images, test_labels_encoded)
print(f"Test accuracy: {test_accuracy:.2f}")

