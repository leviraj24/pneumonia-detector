# 🩻🤖 Pneumonia Detection & Explainable AI System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)
![GUI](https://img.shields.io/badge/GUI-Tkinter-lightgrey.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Chest%20X--Ray-brightgreen.svg?logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📘 Overview

This Python-based application uses a state-of-the-art deep learning model (EfficientNet-B0) to analyze chest X-ray images and predict whether they indicate pneumonia. It includes an interactive GUI built with Tkinter and leverages PyTorch for model implementation.

The system emphasizes explainability and continuous learning:
- Generates Grad-CAM heatmaps to show what parts of the X-ray influenced the prediction
- Incorporates user feedback to improve accuracy over time through active learning

## 🎯 Key Features

- **Efficient Model**: Pre-trained EfficientNet-B0 (ImageNet), fine-tuned on pneumonia X-rays
- **Explainable AI**: Grad-CAM visualizations highlight decision-making regions
- **Interactive GUI**: User-friendly Tkinter app to analyze images, view predictions, and provide feedback
- **Active Learning**: Misclassified images are stored and can be used for retraining
- **Robust Evaluation**: Tracks AUC, Accuracy, Sensitivity, and Specificity
- **Educational Focus**: A proof-of-concept for medical AI with explainability

## 🖼️ Project Showcase

Screenshots of the application in action:

### Main Interface
<img width="1912" height="1023" alt="image" src="https://github.com/user-attachments/assets/51df94c1-15c8-457b-8413-67cd549e77c1" />


### Selecting an Image
<img width="1919" height="1016" alt="image" src="https://github.com/user-attachments/assets/a2373aa5-91e3-4ee5-bbd5-e27d04b34a96" />

### Model Prediction
<img width="1913" height="1013" alt="image" src="https://github.com/user-attachments/assets/0b76292a-5db5-40f9-8ca3-0ac98df9732d" />

### Grad-CAM Heatmap Explanation
<img width="1919" height="996" alt="image" src="https://github.com/user-attachments/assets/ec2a6dd8-e458-4876-a5cf-4976685a1e1b" />


### Feedback Option
<img width="527" height="294" alt="image" src="https://github.com/user-attachments/assets/93dcc5e4-7102-4721-b31f-8721d4a196e1" />


*(All images are hosted on GitHub)*

## 📁 Project Structure

```
pneumonia-detector/
├── data/                 # Dataset (
│   ├── train/            # Training set (downloaded from Kaggle)
│   ├── val/              # Validation set (downloaded from Kaggle)
│   └── feedback/         # Create this folder manually (for misclassified images)
├── models/               # Saved models
│   └── best_efficientnet_b0.pth   # Uploaded trained weights
├── src/                  # Core source code
│   ├── app.py            # GUI application
│   ├── datasets.py       # Data loading utilities
│   ├── train_advanced.py # Training engine
│   └── retrain_from_feedback.py # Active learning retraining
├── scripts/              # Utility scripts
│   ├── export_model.py
│   └── merge_corrections.py
├── README.md
└── requirements.txt
```

## 📊 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

### 📥 Download Dataset

1. Visit [Kaggle Dataset Page](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) (requires Kaggle account)
2. Download the dataset
3. Extract the `chest_xray` folder

### 📂 Setup Instructions

1. Place the extracted contents into the `data/` directory:
   ```
   data/
   ├── train/
   ├── val/
   └── test/ (optional)
   ```

2. **Important**: Manually create a `data/feedback/` folder:
   ```bash
   mkdir data/feedback
   ```
   This folder will store misclassified images for retraining.

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/leviraj24/pneumonia-detector.git
   cd pneumonia-detector
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚡ Usage Workflow

### Step 1: Train the model (Optional - Pre-trained model included)

```bash
python src/train_advanced.py
```

*Saves `best_efficientnet_b0.pth` in `models/`. Note: A pre-trained model is already included in this repo.*

### Step 2: Run the GUI

```bash
python src/app.py
```

1. **Browse Image** → Choose an X-ray
2. **Analyze Image** → Get prediction
3. **Show Explanation** → View Grad-CAM heatmap
4. **Correct / Incorrect** → Save feedback for retraining

### Step 3: Retrain with Feedback

```bash
python src/retrain_from_feedback.py
```

*Incorporates misclassified images from `data/feedback/` into training.*

## 📈 Model Performance

The model was trained on the Kaggle dataset and optimized for AUC (Area Under the Curve).

| Metric | Score | Description |
|--------|-------|-------------|
| **AUC** | ~1.00 | Distinguishes pneumonia vs normal cases |
| **Accuracy** | ~93–100% | Overall correct predictions |
| **Sensitivity** | ~100% | Pneumonia cases correctly identified |
| **Specificity** | ~87–100% | Normal cases correctly identified |

⚠️ **Note**: These scores are based on the Kaggle dataset validation split. Real-world clinical performance will likely be lower.

## 🛠️ Model Architecture

- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Weights**: `models/best_efficientnet_b0.pth` (included in this repo)
- **Training Strategy**: Two-phase fine-tuning (freeze → unfreeze layers)
- **Explainability**: Grad-CAM heatmaps highlight decision regions

## 📄 Dataset Attribution & Licensing

- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) by Paul Mooney, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Model Weights**: `models/best_efficientnet_b0.pth` trained on the above dataset and released under CC BY 4.0 (attribution required for redistribution)
- **Source Code**: Licensed under MIT License (see LICENSE file)

### Model Usage & Redistribution

These model weights are provided for research and educational use. Redistribution and commercial use are permitted under CC BY 4.0, but you must provide attribution to:
- The original dataset authors (Kermany, Zhang, & Goldbaum)
- This repository

## ⚠️ Medical Disclaimer

**This tool is for educational and research purposes only. It must not be used for real medical diagnosis or clinical decision-making. Always consult a licensed medical professional for healthcare decisions.**

This system has not been validated for clinical use and should not be relied upon for medical diagnosis.

## 📝 License

- **Source Code**: This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
- **Model Weights**: Released under CC BY 4.0 license
- **Dataset**: Chest X-Ray Images (Pneumonia) by Paul Mooney, licensed under CC BY 4.0

## 🙏 Acknowledgments

- **Dataset**: Kermany, Zhang, & Goldbaum (2018) – Kaggle Chest X-Ray Images
- **Frameworks**: PyTorch, Tkinter, and the open-source community
- **Model Architecture**: EfficientNet by Google Research

## 📧 Contact

- **Author**: Levi Raj
- **Email**: leviraj24@gmail.com
- **GitHub**: [pneumonia-detector](https://github.com/leviraj24/pneumonia-detector)

---

**Disclaimer**: No patient-identifying information is included in this repository. All data handling complies with privacy guidelines.
