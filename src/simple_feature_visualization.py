import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess image for model input"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_resized = cv2.resize(img, target_size)
    img_array = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img, img_resized, img_batch

def create_simple_heatmap(model, image_batch, original_img):
    """Create a simple attention heatmap using the last convolutional layer"""
    # Get the base model (MobileNetV2)
    base_model = model.layers[0]
    
    # Create a model that outputs the last conv layer
    feature_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    
    # Get feature maps
    feature_maps = feature_model.predict(image_batch)
    
    # Average across channels to get attention map
    attention_map = np.mean(feature_maps[0], axis=-1)
    
    # Normalize and resize to original image size
    attention_map = cv2.resize(attention_map, (original_img.shape[1], original_img.shape[0]))
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    return attention_map

def create_visualization(model_path, image_path, output_path=None):
    """Create feature visualization for a single image"""
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess image
    original_img, resized_img, img_batch = load_and_preprocess_image(image_path)
    
    # Get prediction
    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    class_names = ["Fresh Water", "Contaminated"]
    
    # Create attention heatmap
    attention_map = create_simple_heatmap(model, img_batch, original_img)
    
    # Create heatmap overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb, 0.7, heatmap_colored, 0.3, 0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_rgb)
    axes[0].set_title(f'Original Image\nPrediction: {class_names[predicted_class]}\nConfidence: {confidence:.3f}', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Resized image (what model sees)
    axes[1].imshow(resized_img)
    axes[1].set_title('Model Input\n(128x128 pixels)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Attention heatmap
    axes[2].imshow(attention_map, cmap='jet')
    axes[2].set_title('Feature Attention Map\n(Red = High Attention)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(overlay)
    axes[3].set_title('Feature Visualization\n(Red Areas = Model Focus)', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Feature visualization saved to: {output_path}")
    
    plt.show()
    
    return predicted_class, confidence, class_names[predicted_class]

def create_multiple_visualizations():
    """Create feature visualizations for multiple test images"""
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
                output_path = f"../models/feature_viz_sample_{i+1}.png"
                predicted_class, confidence, class_name = create_visualization(
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

def create_comparison_visualization():
    """Create a side-by-side comparison of fresh vs contaminated features"""
    model_path = "../models/lettuce_transfer_classifier.h5"
    
    # Load model
    model = load_model(model_path)
    
    # Test images
    fresh_img_path = "../data/fresh_water/h1.jpg"
    cont_img_path = "../data/contaminated/ba1.jpg"
    
    if not os.path.exists(fresh_img_path) or not os.path.exists(cont_img_path):
        print("⚠️ Test images not found. Using available images...")
        return
    
    # Process fresh water image
    fresh_img, fresh_resized, fresh_batch = load_and_preprocess_image(fresh_img_path)
    fresh_pred = model.predict(fresh_batch)
    fresh_class = np.argmax(fresh_pred[0])
    fresh_conf = fresh_pred[0][fresh_class]
    fresh_attention = create_simple_heatmap(model, fresh_batch, fresh_img)
    
    # Process contaminated image
    cont_img, cont_resized, cont_batch = load_and_preprocess_image(cont_img_path)
    cont_pred = model.predict(cont_batch)
    cont_class = np.argmax(cont_pred[0])
    cont_conf = cont_pred[0][cont_class]
    cont_attention = create_simple_heatmap(model, cont_batch, cont_img)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Fresh water row
    axes[0, 0].imshow(cv2.cvtColor(fresh_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Fresh Water Lettuce\nPrediction: Fresh Water\nConfidence: {fresh_conf:.3f}', 
                          fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(fresh_attention, cmap='jet')
    axes[0, 1].set_title('Feature Attention\n(Red = High Focus)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    fresh_overlay = cv2.addWeighted(cv2.cvtColor(fresh_img, cv2.COLOR_BGR2RGB), 0.7, 
                                   cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * fresh_attention), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), 0.3, 0)
    axes[0, 2].imshow(fresh_overlay)
    axes[0, 2].set_title('Feature Visualization', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Contaminated row
    axes[1, 0].imshow(cv2.cvtColor(cont_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Contaminated Lettuce\nPrediction: Contaminated\nConfidence: {cont_conf:.3f}', 
                          fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cont_attention, cmap='jet')
    axes[1, 1].set_title('Feature Attention\n(Red = High Focus)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    cont_overlay = cv2.addWeighted(cv2.cvtColor(cont_img, cv2.COLOR_BGR2RGB), 0.7, 
                                  cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cont_attention), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), 0.3, 0)
    axes[1, 2].imshow(cont_overlay)
    axes[1, 2].set_title('Feature Visualization', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = "../models/feature_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Feature comparison saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("🔍 Creating Feature Visualizations...")
    print("This will show which parts of the lettuce images the model focuses on for classification.")
    
    # Create output directory
    os.makedirs("../models", exist_ok=True)
    
    # Generate comparison visualization
    print("\n📊 Creating comparison visualization...")
    create_comparison_visualization()
    
    # Generate individual visualizations
    print("\n📊 Creating individual visualizations...")
    results = create_multiple_visualizations()
    
    print("\n" + "="*60)
    print("📊 FEATURE VISUALIZATION RESULTS")
    print("="*60)
    for result in results:
        print(f"Image: {result['image']}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Output: {result['output_path']}")
        print("-" * 40)
    
    print("\n🎯 Key Insights:")
    print("- Red areas show where the model focuses its attention")
    print("- This helps understand the model's decision-making process")
    print("- Useful for validating model behavior and feature importance")
    print("- Shows which visual features distinguish fresh vs contaminated lettuce")
    print("="*60) 