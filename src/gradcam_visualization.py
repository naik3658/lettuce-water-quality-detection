import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import os

class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        
        # If no layer specified, use the last convolutional layer
        if self.layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    self.layer_name = layer.name
                    break
    
    def compute_heatmap(self, image, class_index=None):
        """Compute Grad-CAM heatmap for the given image"""
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            class_channel = predictions[:, class_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """Overlay heatmap on the original image"""
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert image to RGB if it's BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Overlay heatmap on image
        output = heatmap * alpha + image_rgb * (1 - alpha)
        output = output / output.max()
        return output

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess image for model input"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_resized = cv2.resize(img, target_size)
    img_array = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img, img_resized, img_batch

def create_gradcam_visualization(model_path, image_path, output_path=None):
    """Create Grad-CAM visualization for a single image"""
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess image
    original_img, resized_img, img_batch = load_and_preprocess_image(image_path)
    
    # Create GradCAM
    grad_cam = GradCAM(model)
    
    # Get prediction
    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    class_names = ["Fresh Water", "Contaminated"]
    
    # Compute heatmap
    heatmap = grad_cam.compute_heatmap(img_batch, predicted_class)
    
    # Overlay heatmap on original image
    overlay = grad_cam.overlay_heatmap(original_img, heatmap)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original Image\nPrediction: {class_names[predicted_class]}\nConfidence: {confidence:.3f}', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n(Red = High Attention)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Grad-CAM Overlay\n(Red Areas = Model Focus)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Grad-CAM visualization saved to: {output_path}")
    
    plt.show()
    
    return predicted_class, confidence, class_names[predicted_class]

def create_multiple_gradcam_visualizations():
    """Create Grad-CAM visualizations for multiple test images"""
    model_path = "../models/lettuce_transfer_classifier.h5"
    
    # Test images from your dataset
    test_images = [
        ("../data/fresh_water/h1.jpg", "Fresh Water Sample"),
        ("../data/fresh_water/h3.jpg", "Fresh Water Sample 2"),
        ("../data/contaminated/ba1.jpg", "Contaminated Sample"),
        ("../data/contaminated/ba8.jpg", "Contaminated Sample 2")
    ]
    
    results = []
    
    for i, (image_path, description) in enumerate(test_images):
        if os.path.exists(image_path):
            try:
                output_path = f"../models/gradcam_sample_{i+1}.png"
                predicted_class, confidence, class_name = create_gradcam_visualization(
                    model_path, image_path, output_path
                )
                results.append({
                    'image': description,
                    'predicted_class': class_name,
                    'confidence': confidence,
                    'output_path': output_path
                })
                print(f"✅ Processed {description}: {class_name} ({confidence:.3f})")
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
        else:
            print(f"⚠️ Image not found: {image_path}")
    
    return results

if __name__ == "__main__":
    print("🔍 Creating Grad-CAM Visualizations...")
    print("This will show which parts of the lettuce images the model focuses on for classification.")
    
    # Create output directory
    os.makedirs("../models", exist_ok=True)
    
    # Generate multiple visualizations
    results = create_multiple_gradcam_visualizations()
    
    print("\n" + "="*60)
    print("📊 GRAD-CAM VISUALIZATION RESULTS")
    print("="*60)
    for result in results:
        print(f"Image: {result['image']}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Output: {result['output_path']}")
        print("-" * 40)
    
    print("\n🎯 Key Insights:")
    print("- Red areas in heatmaps show where the model focuses")
    print("- This helps understand the model's decision-making process")
    print("- Useful for validating model behavior and feature importance")
    print("="*60) 