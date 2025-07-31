import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
model = tf.keras.models.load_model("../models/lettuce_classifier.h5")

# Path to a test image (change this to any image you want to test)
test_img_path = "../data/fresh_water/h3.jpg"  # test a fresh water image

img = cv2.imread(test_img_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
class_idx = np.argmax(prediction)
class_names = ["Fresh Water", "Contaminated"]

print(f"Prediction: {class_names[class_idx]} (Confidence: {prediction[0][class_idx]:.2f})") 