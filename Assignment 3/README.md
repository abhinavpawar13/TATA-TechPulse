[README.md](https://github.com/user-attachments/files/32665608/README.md)
# Tata Technologies - TechPulse FY-26: Applied AI & ML
## Lab Statement 3: Data Cleaning & Preprocessing Techniques

**Unit:** Unit 2 – Machine Learning & Applications  
**Lab Statement 3:** *Handle missing values, outliers, and scale features using Pandas and Scikit-learn.*  
**Domain Focus:** Automotive Telemetry & Vehicle Sales Analytics  

---

## 📌 1. Overview & Objectives

In industrial and automotive applications, real-world data captured from onboard ECUs, telemetry sensors, and dealership databases is rarely clean. It commonly contains sensor dropouts (missing values), measurement spikes or communication glitches (outliers), categorical designations, and features with varying units of measure.

This lab delivers an end-to-end, production-ready implementation of data cleaning and feature preprocessing using **Pandas**, **NumPy**, and **Scikit-Learn**.

### Key Learning Outcomes:
1. **Auditing Datasets**: Systematically detect null values, anomalies, and structural distribution skews.
2. **Missing Value Imputation**: Compare and deploy statistical imputation techniques (`SimpleImputer` with median and mode strategies, and multivariate `KNNImputer`).
3. **Outlier Detection & Capping**: Apply **Interquartile Range (IQR)** and **Z-Score** methods, performing **Winsorization (capping)** to preserve training size while neutralizing high-leverage anomaly points.
4. **Categorical Encoding**: Implement **One-Hot Encoding** (`OneHotEncoder` with `drop='first'`) for nominal features and **Ordinal Encoding** (`OrdinalEncoder`) with custom hierarchical domain ordering.
5. **Feature Scaling**: Understand the mathematical formulation and practical trade-offs of **StandardScaler**, **MinMaxScaler**, and **RobustScaler**.
6. **Leak-Free ML Pipeline**: Package transformations into a reusable Scikit-Learn `ColumnTransformer` and `Pipeline` fitted strictly on training data.

---

## 📁 2. Project Directory Structure

```text
lab3_data_cleaning_preprocessing/
│
├── README.md                                   # Comprehensive lab manual & documentation
├── generate_dataset.py                         # Generates synthetic automotive dataset (1,200 records)
├── data_cleaning_preprocessing.py              # End-to-end modular Python pipeline & visualization generator
├── create_notebook.py                          # Generator for the interactive Jupyter Notebook
├── lab3_data_cleaning_preprocessing.ipynb      # Fully executed interactive Jupyter Notebook
│
├── automotive_data_raw.csv                     # Raw dataset containing missing values and outliers
├── automotive_data_cleaned.csv                 # Cleaned business-readable dataset (post imputation & capping)
├── automotive_features_preprocessed.csv        # Scaled & encoded numerical matrix ready for ML models
├── automotive_preprocessing_pipeline.joblib    # Serialized production ColumnTransformer object
│
└── plots/                                      # High-resolution diagnostic visualizations
    ├── 01_missing_values_audit.png             # Breakdown of missing value percentages by feature
    ├── 02_outlier_treatment_boxplots.png       # Before vs After boxplots of outlier capping
    ├── 03_feature_scaling_comparison.png       # Probability density curves across scalers
    └── 04_correlation_matrix_cleaned.png       # Pearson correlation heatmap of treated features
```

---

## 🔬 3. Technical Methodology & Formulations

### 3.1 Missing Value Treatment
- **Numerical Features** (`engine_size_cc`, `horsepower_hp`, `mileage_kmpl`, `maintenance_score`):  
  Automotive telemetry distributions frequently exhibit skewness due to performance variations. The **Median** is chosen over the mean because it is resistant to extreme values:
  $$\text{Imputed Value} = \text{Median}(X)$$
- **Categorical Features** (`fuel_type`):  
  Imputed using the **Most Frequent (Mode)** category:
  $$\text{Imputed Value} = \operatorname{argmax}_c \sum \mathbb{I}(x_i = c)$$
- **Multivariate Imputation (`KNNImputer`)**:  
  For correlated attributes (e.g. curb weight and horsepower), values are estimated from the Euclidean distance of the $k$-nearest vehicle records ($k=5$).

### 3.2 Outlier Detection & Winsorization
- **Interquartile Range (IQR) Method**:
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
  $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
- **Treatment (Winsorization)**: Instead of discarding rows (which reduces training data and causes selection bias), values are capped:
  $$x_{\text{capped}} = \min(\max(x, \text{Lower Bound}), \text{Upper Bound})$$
- **Z-Score Method**: Evaluates standard deviations from the mean:
  $$z = \frac{x - \mu}{\sigma}, \quad \text{Flagged if } |z| > 3.0$$

### 3.3 Categorical Encoding
- **Nominal Variables** (`fuel_type`, `transmission`):  
  Transformed using `OneHotEncoder(drop='first', sparse_output=False)` to prevent the dummy variable trap (multicollinearity).
- **Ordinal Variables** (`vehicle_segment`, `owner_type`):  
  Rank-ordered mapping preserving domain relationships:
  - `vehicle_segment`: Hatchback (0) $\to$ Sedan (1) $\to$ SUV (2) $\to$ Luxury (3)
  - `owner_type`: First (0) $\to$ Second (1) $\to$ Third (2)

### 3.4 Feature Scaling Comparison

| Scaler | Mathematical Transformation | Resulting Distribution | Best Automotive Use-Cases |
| :--- | :--- | :--- | :--- |
| **StandardScaler** | $$z = \frac{x - \mu}{\sigma}$$ | $\mu = 0, \sigma = 1$ | Linear/Ridge Regression, PCA, Support Vector Machines |
| **MinMaxScaler** | $$x_{norm} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$ | Range $[0, 1]$ | Distance metrics (KNN), Neural Network activation layers |
| **RobustScaler** | $$x_{rob} = \frac{x - \text{Median}}{\text{IQR}}$$ | $\text{Median} = 0, \text{IQR} = 1$ | Sensor datasets with stubborn anomalies/heavy tails |

---

## 🚀 4. How to Run

### Prerequisites
Ensure Python 3.8+ or Anaconda is installed with the required libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib jupyter
```

### Option A: Run the End-to-End Python Script
Execute the modular script to process data, generate logs, and produce diagnostic plots:
```bash
cd lab3_data_cleaning_preprocessing
python data_cleaning_preprocessing.py
```
*(If using an Anaconda environment):*
```bash
& "C:\ProgramData\anaconda3\python.exe" data_cleaning_preprocessing.py
```

### Option B: Interactive Jupyter Notebook
Launch Jupyter Notebook or Jupyter Lab to interact with each stage:
```bash
jupyter notebook lab3_data_cleaning_preprocessing.ipynb
```
The notebook is pre-executed with interactive cell outputs and inline plots.

---

## 📊 5. Summary of Experimental Results

1. **Audit Phase**:
   - Identified missing values across 5 attributes: `maintenance_score` (10.0%), `horsepower_hp` (7.9%), `engine_size_cc` (5.4%), `mileage_kmpl` (4.9%), and `fuel_type` (4.0%).
   - Detected negative mileage readings ($-15.0$ kmpl) and extreme odometer readings ($>1,600,000$ km) caused by sensor glitches.
2. **Post-Imputation & Capping**:
   - Zero missing entries remain in `automotive_data_cleaned.csv`.
   - Outliers capped within the $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$ bounds, eliminating artificial leverage points while preserving all 1,200 records.
3. **Pipeline Serialization**:
   - The fitted `automotive_preprocessing_pipeline.joblib` is saved and ready for zero-leakage inference.

---

## 🔗 Alignment with Subsequent Labs
- **Lab 4: Vehicle Price Prediction**: The processed feature matrix (`automotive_features_preprocessed.csv`) directly serves as the feature input $X$ and target $y$ (`vehicle_price_lakhs`) for regression modeling (Linear, Ridge, Lasso, and Random Forest).
