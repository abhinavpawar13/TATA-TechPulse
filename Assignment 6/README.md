# Tata Technologies - TechPulse FY-26: Applied AI & ML
## Lab Statement 6: Traffic Sign Classification using CNN

**Course:** Applied AI & ML (TechPulse FY-26)  
**Track:** AI & ML | **Level:** Intermediate  
**Curriculum Unit:** Unit 3 – Deep Learning and Computer Vision (CNN architecture, Image preprocessing, Traffic sign recognition, Overfitting and dropout, Confusion matrix)  
**Lab Statement 6:** *Train a CNN to recognize traffic signs using the GTSRB dataset.*  
**Domain Focus:** Advanced Driver Assistance Systems (ADAS), Autonomous Perception & Camera-Based Traffic Sign Recognition (TSR)  

---

## 📌 1. Overview & Objectives

Traffic Sign Recognition (TSR) is an essential sub-system of modern ADAS and Level 2+/Level 3 autonomous vehicles (such as Tata Safari/Harrier ADAS suites). Onboard forward-facing cameras capture road imagery containing regulatory, warning, and informational signs under varying illumination, weather, and motion blur.

This lab implements an end-to-end Deep Convolutional Neural Network (CNN) in **PyTorch** to classify traffic signs according to the **German Traffic Sign Recognition Benchmark (GTSRB)** format.

### Key Learning Outcomes:
1. **Computer Vision & CNN Architectures**: Understand convolutional feature extractors, spatial hierarchy learning, pooling, and receptive fields.
2. **PyTorch Deep Learning Pipeline**: Construct modular `nn.Module` networks with `Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`, and `Dropout`.
3. **Regularization & Overfitting Prevention**: Deploy Batch Normalization and Dropout layers ($p=0.4$) to stabilize gradient updates and prevent feature co-adaptation.
4. **Model Optimization**: Train with Cross-Entropy Loss and Adam optimizer with $L_2$ weight decay.
5. **Comprehensive Evaluation**: Track train/val loss curves, compute confusion matrices, and evaluate class-level prediction confidences.

---

## 📁 2. Project Directory Structure

```text
lab6_traffic_sign_classification/
│
├── README.md                                      # Lab manual & technical documentation
├── traffic_signs_metadata.csv                     # Dataset metadata (1,200 samples across 10 classes)
├── traffic_sign_cnn.py                            # Complete PyTorch CNN training & evaluation pipeline
├── lab6_traffic_sign_classification.ipynb         # Interactive Jupyter Notebook
├── requirements.txt                               # Minimal dependencies
│
└── plots/                                         # Diagnostic visualizations (200 DPI)
    ├── 01_traffic_sign_samples_grid.png           # 32x32 RGB traffic sign samples per class
    ├── 02_cnn_training_loss_and_accuracy.png      # Epoch-by-epoch loss and accuracy curves
    ├── 03_confusion_matrix_traffic_signs.png      # 10-class confusion matrix
    └── 04_sample_predictions_with_confidence.png  # Test predictions with confidence percentages
```

---

## 🔬 3. CNN Architecture Specifications

```text
Input (3 x 32 x 32)
 │
 ├── Conv2D(3 -> 32, k=3, p=1) + BatchNorm2D(32) + ReLU + MaxPool2D(2x2) -> [32, 16, 16]
 ├── Conv2D(32 -> 64, k=3, p=1) + BatchNorm2D(64) + ReLU + MaxPool2D(2x2) -> [64, 8, 8]
 ├── Conv2D(64 -> 128, k=3, p=1) + BatchNorm2D(128) + ReLU + MaxPool2D(2x2) -> [128, 4, 4]
 │
 ├── Flatten (128 * 4 * 4 = 2048)
 ├── Dropout(p=0.40)
 ├── Dense(2048 -> 128) + ReLU
 ├── Dropout(p=0.30)
 └── Dense(128 -> 10 Classes)
```

---

## 📊 4. Experimental Results

- **Training Epochs:** 12 Epochs (Adam optimizer, $\text{lr}=10^{-3}$)
- **Peak Test Classification Accuracy:** **100.00%**
- **Final Validation Loss:** $< 0.0001$

---

## 🚀 5. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Full PyTorch CNN Training Pipeline
```bash
python traffic_sign_cnn.py
```

### Step 3: Open Interactive Jupyter Notebook
```bash
jupyter notebook lab6_traffic_sign_classification.ipynb
```
