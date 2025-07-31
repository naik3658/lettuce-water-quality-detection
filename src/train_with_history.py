import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Set paths
fresh_dir = "../data/aug_fresh_water"
contaminated_dir = "../data/aug_contaminated"
IMG_SIZE = 128

def load_images_from_folder(folder, label):
    """Load images from folder and assign labels"""
    images, labels = [], []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
    return images, labels

print("🔄 Loading images...")
fresh_images, fresh_labels = load_images_from_folder(fresh_dir, 0)
cont_images, cont_labels = load_images_from_folder(contaminated_dir, 1)

print(f"✅ Loaded {len(fresh_images)} fresh water images")
print(f"✅ Loaded {len(cont_images)} contaminated images")

# Prepare data
X = np.array(fresh_images + cont_images) / 255.0
y = to_categorical(np.array(fresh_labels + cont_labels), 2)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"📊 Training set: {X_train.shape[0]} images")
print(f"📊 Validation set: {X_val.shape[0]} images")

# Build model with transfer learning
print("🏗️ Building model...")
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("📋 Model Summary:")
model.summary()

# Train model
print("🚀 Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    validation_data=(X_val, y_val),
    batch_size=16,
    verbose=1
)

# Save model
os.makedirs("../models", exist_ok=True)
model.save("../models/lettuce_transfer_classifier.h5")

# Save training history
with open("../models/training_history.pkl", 'wb') as f:
    pickle.dump(history.history, f)

print("✅ Model and training history saved!")

# Plot training curves
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
plt.title('Model Accuracy During Training', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../models/training_curves.png", dpi=300, bbox_inches='tight')
plt.show()

# Print final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("\n" + "="*50)
print("📈 FINAL TRAINING RESULTS")
print("="*50)
print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print("="*50)

# Find best epoch
best_epoch = np.argmax(history.history['val_accuracy']) + 1
best_val_acc = max(history.history['val_accuracy'])
print(f"🏆 Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at epoch {best_epoch}")

print("\n✅ Training complete! Check '../models/training_curves.png' for the plots.") 