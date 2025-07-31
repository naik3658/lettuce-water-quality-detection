import matplotlib.pyplot as plt
import pickle
import os

# If you saved the history object during training, load and plot it
history_path = '../models/lettuce_classifier_history.pkl'

if os.path.exists(history_path):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    # Plot accuracy
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # Plot loss
    plt.subplot(1,2,2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No training history found. To enable evaluation plots, save the history object during training using pickle.\nExample in your training script after model.fit():\n\nimport pickle\nwith open('../models/lettuce_classifier_history.pkl', 'wb') as f:\n    pickle.dump(history.history, f)\n") 