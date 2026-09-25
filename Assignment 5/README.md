# Tata Technologies - TechPulse FY-26: Applied AI & ML
## Lab Statement 5: Predictive Maintenance from Sensor Logs

**Course:** Applied AI & ML (TechPulse FY-26)  
**Track:** AI & ML | **Level:** Intermediate  
**Curriculum Unit:** Unit 1 (Automotive use cases: predictive maintenance) & Unit 4 (Evaluation metrics: precision, recall, F1-score, ROC-AUC)  
**Lab Statement 5:** *Classify component failures using sensor data and Python ML models.*  
**Domain Focus:** Vehicle Onboard Diagnostic (OBD) Telemetry, Machinery Failure Prevention & Condition-Based Maintenance  

---

## 📌 1. Overview & Objectives

In modern connected vehicles and smart manufacturing plants, components continuously log sensor telemetry (temperature, rotational velocity, torque, operational wear). Detecting equipment degradation prior to catastrophic failure minimizes unpredicted downtime, prevents road breakdowns, and lowers warranty service costs.

This lab delivers an end-to-end Machine Learning classification pipeline to predict component failure modes using **Python**, **Pandas**, and **Scikit-Learn**.

### Key Learning Outcomes:
1. **Sensor Telemetry Analysis**: Audit operational telemetry features (`air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`).
2. **Failure Mode Engineering**: Characterize physical failure mechanisms:
   - **Heat Dissipation Failure (HDF)**
   - **Power Failure (PWF)**
   - **Overstrain Failure (OSF)**
   - **Tool Wear Failure (TWF)**
3. **Handling Class Imbalance**: Address natural real-world class skewness using balanced class weighting (`class_weight='balanced'`).
4. **Classifier Benchmarking**: Implement and compare:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Support Vector Classifier (RBF Kernel)
   - Gradient Boosting Classifier
5. **Rigorous Industrial Evaluation**: Analyze Confusion Matrices, Precision, Recall, F1-Score, and ROC-AUC curves.

---

## 📁 2. Project Directory Structure

```text
lab5_predictive_maintenance/
│
├── README.md                                      # Lab manual & technical documentation
├── sensor_maintenance_data.csv                    # Sensory telemetry dataset (1,200 records)
├── predictive_maintenance.py                      # Complete modular ML classification pipeline
├── lab5_predictive_maintenance.ipynb              # Interactive Jupyter Notebook
├── requirements.txt                               # Minimal dependencies
│
└── plots/                                         # Diagnostic visualizations (200 DPI)
    ├── 01_sensor_distributions_and_failures.png   # Class distribution & sensor correlation heatmap
    ├── 02_confusion_matrix_and_roc_curves.png     # Best model Confusion Matrix & ROC-AUC curves
    └── 03_sensor_feature_importance.png           # Gini feature importances in failure detection
```

---

## 📊 3. Quantitative Benchmark Results

Evaluated on an 80/20 stratified holdout split ($N = 1,200$ sensor logs):

| Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.5792 | 0.2719 | 0.6327 | 0.3804 | 0.6789 |
| **Decision Tree Clf** | 0.9125 | 0.7692 | 0.8163 | 0.7921 | 0.8583 |
| **Random Forest Clf** | **0.9333** | 0.7797 | **0.9388** | **0.8519** | 0.9529 |
| **Support Vector Clf (RBF)** | 0.8792 | 0.6351 | **0.9592** | 0.7642 | 0.9429 |
| **Gradient Boosting Clf** | 0.9250 | **0.8780** | 0.7347 | 0.8000 | **0.9580** |

### Key Analytical Takeaways:
- **Importance of Recall in Maintenance**: In predictive maintenance, missing an impending failure (False Negative) is substantially more hazardous than inspecting a false alarm (False Positive). **Random Forest** ($93.9\%$ recall) and **SVC** ($95.9\%$ recall) successfully capture over 93% of impending breakdowns.
- **Top Failure Indicators**: Operational torque, shaft rotational speed, and accumulated tool wear time are the dominant predictors driving physical degradation.

---

## 🚀 4. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Full ML Pipeline & Diagnostics
```bash
python predictive_maintenance.py
```

### Step 3: Open Interactive Jupyter Notebook
```bash
jupyter notebook lab5_predictive_maintenance.ipynb
```
