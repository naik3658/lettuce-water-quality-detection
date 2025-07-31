import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = 128
model = tf.keras.models.load_model("../models/lettuce_classifier.h5")

# Folders to evaluate
folders = [
    ("../data/aug_fresh_water", 0),      # 0 = Fresh Water
    ("../data/aug_contaminated", 1)      # 1 = Contaminated
]

X = []
y_true = []

for folder, label in folders:
    for fname in os.listdir(folder):
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            X.append(img)
            y_true.append(label)

X = np.array(X)
y_true = np.array(y_true)

# Predict
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy
accuracy = np.mean(y_pred == y_true)
print(f"Overall accuracy: {accuracy*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_names = ["Fresh Water", "Contaminated"]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names)) 