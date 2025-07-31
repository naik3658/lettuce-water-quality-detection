# 🥬 AI-Powered Lettuce Water Quality Detection

## 📋 Project Overview

This project implements an AI-based system to detect whether lettuce was grown using contaminated water or fresh water through visual analysis. The system uses deep learning with transfer learning to classify lettuce images and achieve 93.48% accuracy.

## 🎯 Problem Statement

Traditional methods for detecting water contamination in vegetables require expensive laboratory testing and are destructive to the produce. This project addresses this challenge by developing a non-destructive, AI-based system that can quickly assess water quality through visual analysis of lettuce leaves.

## 🏗️ Architecture

- **Model:** MobileNetV2 with transfer learning
- **Input:** 128x128 RGB images
- **Output:** Binary classification (Fresh Water / Contaminated)
- **Accuracy:** 93.48% validation accuracy
- **Framework:** TensorFlow/Keras
- **Interface:** Streamlit web application

## 📁 Project Structure

```
lettuce_contamination_project/
├── data/
│   ├── fresh_water/          # Fresh water lettuce images
│   ├── contaminated/         # Contaminated lettuce images
│   ├── aug_fresh_water/      # Augmented fresh water images
│   └── aug_contaminated/     # Augmented contaminated images
├── models/
│   ├── lettuce_transfer_classifier.h5  # Trained model
│   ├── training_history.pkl            # Training history
│   ├── training_curves.png             # Training curves
│   └── feature_comparison.png          # Feature visualization
├── src/
│   ├── organize_lettuce_images.py      # Data organization script
│   ├── augment_lettuce_images.py       # Data augmentation
│   ├── train_with_history.py           # Model training with curves
│   ├── simple_feature_visualization.py # Feature visualization
│   ├── batch_evaluate_lettuce_model.py # Model evaluation
│   ├── predict_lettuce_image.py        # Single image prediction
│   ├── lettuce_predict_app.py          # Streamlit web app
│   └── create_ppt_screenshots.py       # PPT visualization generation
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── Speaker_Notes_PPT.md               # Presentation speaker notes
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11 (TensorFlow compatibility)
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/lettuce-water-quality-detection.git
cd lettuce-water-quality-detection
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Data Setup

1. **Organize your lettuce images:**
   - Place fresh water lettuce images in `data/fresh_water/`
   - Place contaminated lettuce images in `data/contaminated/`

2. **Run data augmentation:**
```bash
cd src
python augment_lettuce_images.py
```

### Model Training

1. **Train the model:**
```bash
python train_with_history.py
```

2. **View training results:**
```bash
python plot_training_curves.py
```

### Model Evaluation

1. **Evaluate on test images:**
```bash
python batch_evaluate_lettuce_model.py
```

2. **Test single image:**
```bash
python predict_lettuce_image.py
```

### Web Application

1. **Run Streamlit app:**
```bash
streamlit run lettuce_predict_app.py
```

2. **Open browser:** Navigate to `http://localhost:8501`

## 📊 Results

### Model Performance
- **Validation Accuracy:** 93.48%
- **Training Accuracy:** 99.45%
- **Best Epoch:** 7
- **Dataset Size:** 228 images (182 training, 46 validation)

### Feature Visualization
The model learns to focus on:
- Leaf texture and color patterns
- Visual anomalies (spots, discoloration)
- Growth patterns and structural features

## 🛠️ Usage Examples

### Single Image Prediction
```python
from src.predict_lettuce_image import predict_image

# Predict a single image
result = predict_image("path/to/lettuce_image.jpg")
print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Batch Evaluation
```python
from src.batch_evaluate_lettuce_model import evaluate_model

# Evaluate model on test dataset
accuracy, confusion_matrix = evaluate_model()
print(f"Overall Accuracy: {accuracy:.2f}%")
```

## 📈 Training Curves

The model shows smooth learning progression without overfitting:
- Training and validation accuracy curves follow similar patterns
- Early convergence at epoch 7
- Stable performance throughout training

## 🔍 Feature Visualization

The AI attention maps show what the model focuses on:
- **Red areas:** High attention (model focus)
- **Blue areas:** Low attention
- **Fresh water:** Focuses on healthy leaf patterns
- **Contaminated:** Detects visual anomalies

## 🎯 Applications

- **Agricultural Quality Control:** Farmers can assess irrigation water quality
- **Food Safety Inspection:** Rapid screening of produce
- **Consumer Protection:** Verify vegetable safety
- **Research:** Study contamination effects on plant appearance

## 🔬 Technical Details

### Model Architecture
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning:** Leverages features from millions of images
- **Custom Layers:** Dense layers (128 → 64 → 2 neurons)
- **Regularization:** Dropout (0.2) to prevent overfitting

### Data Augmentation
- Horizontal/Vertical flips
- Random brightness/contrast
- Rotation (±30 degrees)
- Gaussian noise

### Training Parameters
- **Epochs:** 15
- **Batch Size:** 16
- **Optimizer:** Adam
- **Loss Function:** Categorical crossentropy
- **Validation Split:** 20%

## 📚 References

1. Zhou, Y., et al. (2023). "Detection of lead and cadmium in lettuce using fluorescence hyperspectral imaging and machine learning." *Food Chemistry*, 405, 134-142.

2. Chen, Y., et al. (2022). "Rapid detection of cadmium in lettuce leaves using FTIR spectroscopy and machine learning." *Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy*, 268, 120-135.

3. Li, J., et al. (2021). "Detection of pesticide residues in vegetables using visible/near-infrared hyperspectral imaging and chemometrics." *Food Analytical Methods*, 14(8), 1567-1578.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Karthik Naik**
- **Student ID:** 2024PCP5320
- **Program:** CSE M.Tech
- **Guide:** Dr. Namita Mittal
- **Institution:** MNIT Jaipur

## 🙏 Acknowledgments

- Dr. Namita Mittal for guidance and supervision
- MNIT Jaipur for academic support
- Open source community for tools and libraries

## 📞 Contact

For questions or collaboration, please contact:
- **Email:** [your-email@example.com]
- **GitHub:** [@yourusername]

---

**⭐ Star this repository if you find it helpful!** 