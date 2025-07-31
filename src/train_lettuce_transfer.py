import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Paths
fresh_dir = "../data/aug_fresh_water"
contaminated_dir = "../data/aug_contaminated"
IMG_SIZE = 128

def load_images_from_folder(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return images, labels

fresh_images, fresh_labels = load_images_from_folder(fresh_dir, 0)
cont_images, cont_labels = load_images_from_folder(contaminated_dir, 1)
X = np.array(fresh_images + cont_images) / 255.0
y = to_categorical(np.array(fresh_labels + cont_labels), 2)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Transfer learning model
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=16)
os.makedirs("../models", exist_ok=True)
model.save("../models/lettuce_transfer_classifier.h5")
print("✅ Transfer learning model training complete! Model saved to models/lettuce_transfer_classifier.h5") 