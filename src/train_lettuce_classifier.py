import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# Paths to your data
fresh_dir = "/Users/karthiknaik/Documents/Gen AI/M.Tech/lettuce_contamination_project/data/aug_fresh_water"
contaminated_dir = "/Users/karthiknaik/Documents/Gen AI/M.Tech/lettuce_contamination_project/data/aug_contaminated"

IMG_SIZE = 128  # Resize images to 128x128

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return images, labels

# Load data
fresh_images, fresh_labels = load_images_from_folder(fresh_dir, 0)
cont_images, cont_labels = load_images_from_folder(contaminated_dir, 1)

X = np.array(fresh_images + cont_images)
y = np.array(fresh_labels + cont_labels)

# Normalize images
X = X / 255.0

# One-hot encode labels
y = to_categorical(y, 2)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build a simple CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=16)

# Save the model
os.makedirs("../models", exist_ok=True)
model.save("../models/lettuce_classifier.h5")

print("✅ Model training complete! Model saved to models/lettuce_classifier.h5") 