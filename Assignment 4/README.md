# Tata Technologies - TechPulse FY-26: Applied AI & ML
## Lab Statement 4: Vehicle Price Prediction

**Course:** Applied AI & ML (TechPulse FY-26)  
**Track:** AI & ML | **Level:** Intermediate  
**Curriculum Unit:** Unit 2 – Machine Learning & Applications (Case study: Car price prediction, Regression, Decision Trees & Random Forests)  
**Lab Statement 4:** *Use regression models to estimate vehicle prices from structured data.*  
**Domain Focus:** Automotive Resale Valuation, Depreciation Analytics & Dealership Pricing  

---

## 📌 1. Overview & Objectives

In automotive sales and dealership analytics, estimating the fair market resale value of pre-owned vehicles is crucial for trade-ins, fleet remarketing, and consumer financing. Resale price is governed by compound economic depreciation over vehicle age, mileage wear, fuel type, transmission, and initial showroom price.

This lab implements an end-to-end Machine Learning regression pipeline to predict vehicle prices using **Python**, **Pandas**, and **Scikit-Learn**.

### Key Learning Outcomes:
1. **Automotive Depreciation Modeling**: Analyze exponential and multi-factor depreciation across vehicle age, odometer mileage, owner history, and fuel type.
2. **Feature Engineering & Transformation**: Build Scikit-Learn `ColumnTransformer` pipelines combining `StandardScaler` for numeric attributes and `OneHotEncoder(drop='first')` for categorical designations.
3. **Model Benchmarking**: Compare linear and non-linear regression models:
   - Multiple Linear Regression
   - Ridge Regression ($L_2$ penalty)
   - Lasso Regression ($L_1$ penalty)
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. **Metric Evaluation**: Evaluate models across **MAE (₹ Lakhs)**, **RMSE (₹ Lakhs)**, **MAPE (%)**, **$R^2$**, and **Adjusted $R^2$**.
5. **Feature Importance Interpretation**: Quantify the dominant factors driving resale valuations via Random Forest Gini MDI.

---

## 📁 2. Project Directory Structure

```text
lab4_vehicle_price_prediction/
│
├── README.md                                      # Lab manual & technical documentation
├── car_price_dataset.csv                          # Structured vehicle resale dataset (1,000 records)
├── vehicle_price_prediction.py                    # Complete modular ML regression pipeline
├── lab4_vehicle_price_prediction.ipynb            # Interactive Jupyter Notebook
├── requirements.txt                               # Minimal dependencies
│
└── plots/                                         # Diagnostic visualizations (200 DPI)
    ├── 01_price_distribution_and_correlation.png  # Resale price distribution & correlation heatmap
    ├── 02_scatter_key_features_vs_price.png       # Present price and vehicle age depreciation boxplots
    ├── 03_model_evaluation_comparison.png         # Bar charts comparing Test R² and Test MAE across models
    └── 04_feature_importance_ranking.png          # Gini feature importances in resale pricing
```

---

## 📊 3. Quantitative Benchmark Results

Evaluated on an 80/20 train/test holdout split ($N = 1,000$ vehicles):

| Model | Test MAE (₹ Lakhs) | Test RMSE (₹ Lakhs) | Test MAPE (%) | Test $R^2$ | Adjusted $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multiple Linear Regr** | 1.703 | 2.468 | 45.73% | 0.8165 | 0.8088 |
| **Ridge Regression ($L_2$)** | 1.698 | 2.468 | 45.15% | 0.8166 | 0.8089 |
| **Lasso Regression ($L_1$)** | 1.699 | 2.468 | 45.11% | 0.8166 | 0.8089 |
| **Decision Tree Regr** | 0.881 | 1.281 | 15.66% | 0.9506 | 0.9485 |
| **Random Forest Regr** | 0.581 | 0.889 | 10.73% | 0.9762 | 0.9752 |
| **Gradient Boosting Regr** | **0.461** | **0.695** | **8.93%** | **0.9854** | **0.9848** |

### Key Findings:
- **Non-Linear Ensembles Dominate**: Gradient Boosting Regressor ($R^2 = 0.9854$, $\text{MAE} = \text{₹}0.46\text{ Lakhs}$) significantly outperforms Multiple Linear Regression ($R^2 = 0.8165$) by capturing the exponential decay nature of vehicle depreciation.
- **Top Value Determinants**: Present showroom price accounts for >65% of price variance, followed closely by vehicle age (>20%) and odometer mileage.

---

## 🚀 4. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Full ML Pipeline & Diagnostics
```bash
python vehicle_price_prediction.py
```

### Step 3: Open Interactive Jupyter Notebook
```bash
jupyter notebook lab4_vehicle_price_prediction.ipynb
```
