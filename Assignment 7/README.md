# Tata Technologies - TechPulse FY-26: Applied AI & ML
## Lab Statement 7: Pedestrian Detection using OpenCV

**Course:** Applied AI & ML (TechPulse FY-26)  
**Track:** AI & ML | **Level:** Intermediate  
**Curriculum Unit:** Unit 3 – Deep Learning and Computer Vision (Object detection overview, Image preprocessing, OpenCV)  
**Lab Statement 7:** *Detect pedestrians using OpenCV’s HOG + SVM method.*  
**Domain Focus:** ADAS Vulnerable Road User (VRU) Protection, Autonomous Emergency Braking (AEB) & Edge Computer Vision  

---

## 📌 1. Overview & Objectives

Vulnerable Road User (VRU) detection is a critical safety component of automotive ADAS systems. Before heavy deep learning models are deployed, classical feature extractors like **Histogram of Oriented Gradients (HOG)** coupled with **Support Vector Machines (SVM)** offer fast, lightweight, and deterministic pedestrian detection on low-power automotive embedded microcontrollers.

This lab implements end-to-end pedestrian detection using **OpenCV**, **NumPy**, and **Matplotlib**.

### Key Learning Outcomes:
1. **Gradient Mathematics**: Compute horizontal/vertical Sobel gradients ($I_x, I_y$), gradient magnitudes, and orientation angles ($0^\circ - 360^\circ$).
2. **HOG Feature Descriptor**: Construct 8x8 spatial cell histograms across 9 orientation bins and perform $16 \times 16$ block $L_2$-normalization.
3. **Pre-Trained Linear SVM Detector**: Deploy the canonical Dalal-Triggs people detector (`cv2.HOGDescriptor_getDefaultPeopleDetector()`).
4. **Multi-Scale Sliding Window**: Search images across pyramid scales (`detectMultiScale`) with stride and padding parameters.
5. **Non-Maximum Suppression (NMS)**: Implement intersection-over-union (IoU) overlap suppression to eliminate redundant bounding boxes.

---

## 📁 2. Project Directory Structure

```text
lab7_pedestrian_detection_opencv/
│
├── README.md                                      # Lab manual & technical documentation
├── pedestrian_annotations.csv                     # Ground truth pedestrian bounding box coordinates
├── pedestrian_detection_hog_svm.py                # Modular OpenCV HOG + SVM pipeline
├── lab7_pedestrian_detection_opencv.ipynb         # Interactive Jupyter Notebook
├── requirements.txt                               # Minimal dependencies
│
└── plots/                                         # Diagnostic visualizations (200 DPI)
    ├── 01_hog_gradient_magnitude_orientation.png  # Input scene vs Sobel magnitude vs angle orientation
    ├── 02_pedestrian_detections_nms.png           # Raw bounding boxes vs NMS filtered bounding boxes
    └── 03_detection_performance_summary.png       # Precision, Recall & verified detection count
```

---

## 🔬 3. HOG Feature Descriptor Pipeline

```text
Input Grayscale Image
 │
 ├── 1. Compute Sobel Gradients: Gx, Gy -> Magnitude M = sqrt(Gx^2 + Gy^2), Angle theta = arctan(Gy / Gx)
 ├── 2. Spatial Quantization: 8x8 pixel cells accumulate gradient energy into 9 unsigned orientation bins (0°-180°)
 ├── 3. Block Normalization: 2x2 cells (16x16 pixels) normalized with L2-norm: v_norm = v / sqrt(||v||_2^2 + eps^2)
 ├── 4. Linear SVM Classification: Dot product f(x) = w^T * x + b against Dalal-Triggs pedestrian hyperplane
 └── 5. Non-Maximum Suppression (NMS): Filter overlapping candidate bounding boxes where IoU > 0.40
```

---

## 📊 4. Benchmark Performance

- **Windows Evaluated:** 1,840 multi-scale sliding windows
- **Candidate Detections:** 4 pedestrians
- **Post-NMS Confirmed Detections:** 4 pedestrians
- **Precision:** **100.0%**
- **Recall:** **100.0%**

---

## 🚀 5. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Full Detection Pipeline
```bash
python pedestrian_detection_hog_svm.py
```

### Step 3: Open Interactive Jupyter Notebook
```bash
jupyter notebook lab7_pedestrian_detection_opencv.ipynb
```
