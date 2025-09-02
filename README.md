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
- **Active Learning**: Learns from mistakes! Misclassified images are fed back into the training loop
- **Robust Evaluation**: Tracks AUC, Accuracy, Sensitivity, and Specificity
- **Educational Focus**: A proof-of-concept for medical AI with explainability

## 🖼️ Project Showcase

Screenshots of the application in action:

### Main Interface
![Main Interface](https://github.com/user-attachments/assets/0d8e7b74-7f0a-4ecb-9700-b17f74e955ca)

### Selecting an Image
![Selecting an Image](https://github.com/user-attachments/assets/1d98a643-fe48-4338-a6af-403e72ffc0bc)

### Model Prediction
![Model Prediction](https://github.com/user-attachments/assets/a706e8e5-3d6e-41d9-9633-0f59aa87df4a)

### Grad-CAM Heatmap Explanation
![Grad-CAM Heatmap](https://github.com/user-attachments/assets/df053d98-2466-41e8-8c85-e4dfa9b7a466)

### Feedback Option
![Feedback Option](https://github.com/user-attachments/assets/7658b1d1-c413-4d02-a32b-610b3f002389)

*(All images are hosted on GitHub)*

## 📁 Project Structure

```
pneumonia-detector/
├── data/                 # Datasets (ignored by Git)
│   ├── train/            # Training set
│   ├── val/              # Validation set
│   └── feedback/         # Misclassified images for retraining
├── models/               # Saved models (ignored by Git)
│   └── best_efficientnet_b0.pth
├── src/                  # Core source code
│   ├── app.py            # GUI application
│   ├── datasets.py       # Data loading utilities
│   ├── train_advanced.py # Training engine
│   └── retrain_from_feedback.py # Active learning retraining
├── venv/                 # Virtual environment (ignored by Git)
├── .gitignore
├── README.md
└── requirements.txt
```

## 📊 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

- 📥 **[Download Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)**
- 📂 **Setup**: Extract and place the `train/` and `val/` folders into `data/`. Optionally include `test/`.

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

### Step 1: Train the model
```bash
python src/train_advanced.py
```
*Saves `best_efficientnet_b0.pth` in `models/`.*

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
*Incorporates feedback images and improves the model.*

## 📈 Model Performance

The model is trained to optimize for AUC (Area Under the Curve).

| Metric | Score | Description |
|--------|-------|-------------|
| **AUC** | ~1.00 | Ability to distinguish between pneumonia and normal cases |
| **Accuracy** | ~93–100% | Overall percentage of correct predictions |
| **Sensitivity** | ~100% | Percentage of pneumonia cases correctly identified |
| **Specificity** | ~87–100% | Percentage of normal cases correctly identified |

⚠️ **Note**: These scores are based on the Kaggle dataset's validation split. Real-world clinical performance will likely be lower.

## 🛠️ Model Architecture

- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Strategy**: Two-phase fine-tuning (freeze → unfreeze layers)
- **Explainability**: Grad-CAM heatmaps highlight decision-making regions

## 📝 License

This project is licensed under the MIT License. See LICENSE for details.

## ⚠️ Medical Disclaimer

**This tool is for educational and research purposes only. It must not be used for real medical diagnosis. Always consult a licensed medical professional for healthcare decisions.**

## 🙏 Acknowledgments

- **Dataset**: Kermany, Zhang, & Goldbaum (2018) – Kaggle Chest X-Ray Images
- **Frameworks**: PyTorch, Tkinter, and the open-source community

## 📧 Contact

- **Author**: Levi Raj
- **Email**: leviraj24@gmail.com
- **GitHub**: [pneumonia-detector](https://github.com/leviraj24/pneumonia-detector)
"# pneumonia-detector" 
