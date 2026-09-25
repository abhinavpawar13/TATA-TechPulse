# Tata Technologies - TechPulse FY-26: Applied AI & ML
## Lab Statement 9: Feature Importance Visualization

**Course:** Applied AI & ML (TechPulse FY-26)  
**Track:** AI & ML | **Level:** Intermediate  
**Curriculum Unit:** Unit 4 – AI fundamentals & application development (Feature selection and importance, Gradient Boosting and XGBoost, Model interpretability)  
**Lab Statement 9:** *Visualize and interpret feature importance in automotive datasets.*  
**Domain Focus:** Powertrain Efficiency, Vehicle Fuel Consumption Attribution & Explainable AI (XAI)  

---

## 📌 1. Overview & Objectives

In automotive engineering, complex black-box machine learning models (like Random Forests and Gradient Boosted Trees) can achieve high predictive accuracy. However, automotive safety certification and powertrain calibrations demand transparent model interpretability: engineers must understand *which* physical attributes drive fuel consumption and emissions predictions, and *how* changes in weight, displacement, and aerodynamic drag affect outputs.

This lab delivers a comprehensive comparative analysis of modern feature importance and Explainable AI (XAI) techniques using **Python**, **Scikit-Learn**, and **Matplotlib**.

### Key Learning Outcomes:
1. **Mean Decrease in Impurity (MDI / Gini Importance)**: Analyze tree-based variance reduction across internal split nodes.
2. **Permutation Feature Importance (PFI)**: Measure true out-of-sample performance drops on the holdout test set to avoid training-set cardinality bias.
3. **Mutual Information (MI)**: Capture non-parametric non-linear dependencies between inputs and targets.
4. **Partial Dependence Plots (PDP)**: Visualize the marginal non-linear response curves of top physical attributes on vehicle fuel consumption.
5. **Noise Immunity Testing**: Validate that spurious, uninformative random noise features are correctly identified and penalized down to $0.0\%$.

---

## 📁 2. Project Directory Structure

```text
lab9_feature_importance_visualization/
│
├── README.md                                      # Lab manual & technical documentation
├── automotive_features_dataset.csv                # Powertrain engineering dataset (1,000 records)
├── feature_importance_analysis.py                 # Modular XAI feature importance pipeline
├── lab9_feature_importance_visualization.ipynb    # Interactive Jupyter Notebook
├── requirements.txt                               # Minimal dependencies
│
└── plots/                                         # Diagnostic visualizations (200 DPI)
    ├── 01_mdi_gini_feature_importance.png         # Random Forest vs Gradient Boosting MDI bar charts
    ├── 02_permutation_importance_test_set.png     # Test-set permutation importance with error bars
    ├── 03_partial_dependence_plots.png            # PDP non-linear marginal response curves
    └── 04_feature_importance_methods_comparison.png # Comparison of MDI vs Permutation vs Mutual Info
```

---

## 📊 3. Feature Importance Methods Comparison

| Feature | MDI (Random Forest) % | Permutation Importance % | Mutual Information % | Physical Engineering Role |
| :--- | :---: | :---: | :---: | :--- |
| **`curb_weight_kg`** | **30.2%** | **42.4%** | 13.9% | Primary rolling resistance & inertial force |
| **`horsepower_hp`** | **34.3%** | **25.3%** | **33.4%** | Engine power demand & fuel flow rate |
| **`engine_displacement_cc`** | 17.3% | 16.2% | 31.7% | Volumetric cylinder displacement intake |
| **`aerodynamic_cd`** | 7.0% | 7.6% | 0.0% | High-speed aerodynamic resistance |
| **`frontal_area_m2`** | 6.7% | 6.9% | 0.0% | Frontal projected surface area |
| **`gear_count`** | 1.7% | 1.2% | 0.6% | Gearbox ratio efficiency |
| **`rolling_crr`** | 1.5% | 0.5% | 3.2% | Tire rolling resistance coefficient |
| **`cylinders`** | 0.2% | 0.0% | 12.2% | Discrete cylinder count (collinear with CC) |
| **`uninformative_noise`** | 1.3% | **0.0%** | 5.0% | Control noise feature (correctly pruned) |

---

## 🚀 4. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Full XAI & Feature Importance Pipeline
```bash
python feature_importance_analysis.py
```

### Step 3: Open Interactive Jupyter Notebook
```bash
jupyter notebook lab9_feature_importance_visualization.ipynb
```
