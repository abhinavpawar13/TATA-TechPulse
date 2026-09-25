[README.md](https://github.com/user-attachments/files/32665481/README.md)
# Tata Technologies - TechPulse FY-26: Applied AI & ML
## Lab Statement 1: ML Model for Car Mileage Estimation

**Curriculum Units:** Unit 1 (Introduction to AI & ML) & Unit 2 (Machine Learning & Applications)  
**Lab Statement 1:** *Predict car mileage using regression and Python libraries.*  
**Dataset:** Inbuilt Seaborn `mpg` Automotive Benchmark Dataset (UCI Machine Learning Repository)  
**Domain Focus:** Automotive Powertrain Fuel Economy, Vehicle Efficiency & Emissions Estimation  

---

## 📌 1. Overview & Objectives

Fuel economy estimation is an essential pillar of automotive systems engineering. As automotive manufacturers such as Tata Motors optimize vehicle designs for internal combustion, hybrid, and alternative powertrain vehicles, predictive models enable engineers to estimate fuel consumption (Miles Per Gallon [MPG] and Kilometers Per Liter [km/L]) before costly physical wind-tunnel and dyno testing.

This laboratory delivers an end-to-end, production-grade implementation of **Car Mileage Estimation** using Python's scientific and machine learning ecosystem (**NumPy**, **Pandas**, **Scikit-Learn**, **SciPy**, **Matplotlib**, and **Seaborn**).

### Key Learning Outcomes:
1. **Auditing Automotive Telemetry**: Ingest the canonical inbuilt `mpg` dataset, inspect feature datatypes, and handle missing values (`horsepower`) using robust statistical median imputation.
2. **Exploratory Data Analysis (EDA)**: Characterize non-linear dependencies between vehicle weight, engine displacement, cylinder count, horsepower, acceleration, model year, and origin.
3. **Leak-Free ML Pipeline**: Build Scikit-Learn `ColumnTransformer` pipelines combining `StandardScaler` for continuous features and `OneHotEncoder(drop='first')` for categorical features, fitted strictly on training data.
4. **Comprehensive Model Comparison**: Formulate, train, and benchmark 9 regression models:
   - Baseline Mean Regressor (Dummy benchmark)
   - Simple Linear Regression (Weight $\to$ Mileage)
   - Multiple Linear Regression (MLR)
   - Polynomial Regression (Degree 2 with interactions)
   - Ridge Regression ($L_2$ Regularization with 5-fold CV)
   - Lasso Regression ($L_1$ Regularization with Feature Selection)
   - Decision Tree Regressor
   - Random Forest Regressor (Ensemble Bagging)
   - Gradient Boosting Regressor (Sequential Boosting)
5. **Statistical Metrics & Residual Diagnostics**: Evaluate models across **MAE**, **MSE**, **RMSE**, **MAPE (%)**, **$R^2$**, and **Adjusted $R^2$**, verifying Gauss-Markov assumptions (homoscedasticity and residual normality).
6. **Production Deployment**: Serialize the best pipeline using `joblib` and build an interactive inference tool for custom vehicle specifications.

---

## 📁 2. Project Directory Structure

```text
lab1_car_mileage_estimation/
│
├── README.md                                      # Comprehensive lab manual & technical documentation
├── auto_mpg.csv                                   # Inbuilt Seaborn MPG dataset cached locally
├── mileage_prediction.py                          # Full modular Python pipeline (EDA, training, evaluation, plots)
├── create_notebook.py                             # Automated Jupyter Notebook generator script
├── lab1_car_mileage_estimation.ipynb              # Fully structured interactive Jupyter Notebook
├── interactive_inference.py                       # CLI & programmatic inference tool with fuel cost calculator
├── model_benchmark_results.csv                    # Quantified test metrics across all 9 regression models
│
├── best_car_mileage_model.joblib                  # Serialized production Random Forest pipeline artifact
├── multiple_linear_regression_pipeline.joblib     # Serialized Multiple Linear Regression pipeline artifact
│
└── plots/                                         # High-resolution diagnostic figures (200 DPI)
    ├── 01_eda_correlation_mileage_distribution.png    # Target distribution histogram & Pearson correlation heatmap
    ├── 02_scatter_key_features_vs_mileage.png         # 4-panel regression scatter plots of physical features vs MPG
    ├── 03_simple_vs_multiple_linear_regression.png    # SLR fit line vs MLR actual vs predicted comparison
    ├── 04_model_performance_comparison.png            # Bar chart comparing Test R², MAE, and RMSE across models
    ├── 05_residual_diagnostics_homoscedasticity.png   # 4-quadrant diagnostic plot (Homoscedasticity, Q-Q, Error distribution)
    └── 06_feature_importance_coefficients.png         # Linear Regression standardized β weights vs RF Gini importances
```

---

## 🔬 3. Automotive Engineering Physics & Mathematical Formulations

### 3.1 Physics of Fuel Economy
Vehicle fuel consumption is governed by the total power required to overcome vehicle drag and inertia:

$$P_{\text{req}} = \left( F_{\text{rolling}} + F_{\text{aero}} + F_{\text{accel}} \right) \cdot v$$

- **Rolling Resistance**: $F_{\text{rolling}} = C_{rr} \cdot m \cdot g$ (directly proportional to vehicle curb weight $m$)
- **Aerodynamic Drag**: $F_{\text{aero}} = \frac{1}{2} \rho C_d A v^2$ (proportional to frontal area $A$ and speed squared)
- **Inertial Resistance**: $F_{\text{accel}} = m \cdot a$ (mass acceleration)

Because vehicle weight and engine displacement dominate both static rolling friction and the volumetric fuel intake per cylinder cycle, fuel economy ($\text{MPG}$) shows a characteristic inverse relationship with vehicle weight and displacement.

### 3.2 Unit Conversions
The lab evaluates models in US Miles Per Gallon ($\text{MPG}$) and reports equivalent SI Metric units in Kilometers Per Liter ($\text{km/L}$):

$$1 \text{ US MPG} = 0.4251437 \text{ km/L}$$
$$\text{Mileage (km/L)} = \text{Mileage (MPG)} \times 0.4251437$$

### 3.3 Regression Models Formulations

| Model | Mathematical Objective / Hypothesis | Automotive Domain Role |
| :--- | :--- | :--- |
| **Simple Linear Regression** | $\hat{y} = w \cdot x_{\text{weight}} + b$ | Establishes single-variable physical baseline using curb weight. |
| **Multiple Linear Regression** | $\hat{y} = w_0 + \sum_{j=1}^p w_j X_j$ | Integrates all displacement, horsepower, year, and origin dimensions. |
| **Polynomial Regression** | $\hat{y} = w_0 + \sum w_j X_j + \sum w_{jk} X_j X_k$ | Models non-linear deceleration and diminishing fuel returns. |
| **Ridge Regression ($L_2$)** | $\min_w \|y - Xw\|_2^2 + \alpha \|w\|_2^2$ | Mitigates high collinearity between displacement, cylinders & weight. |
| **Lasso Regression ($L_1$)** | $\min_w \frac{1}{2n} \|y - Xw\|_2^2 + \alpha \|w\|_1$ | Drives uninformative coefficients to zero for automated feature selection. |
| **Random Forest Regressor** | $\hat{y} = \frac{1}{B}\sum_{b=1}^B T_b(X)$ | Captures complex non-linear interactions without feature engineering. |

### 3.4 Evaluation Metrics

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
3. **Mean Absolute Percentage Error (MAPE %)**:
   $$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
4. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
5. **Adjusted $R^2$**:
   $$\bar{R}^2 = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}$$

---

## 📊 4. Experimental Benchmark Results

Evaluated on an 80/20 train/test holdout split ($N_{\text{train}} = 318, N_{\text{test}} = 80$):

| Model | Train MAE | Test MAE (MPG) | Test RMSE (MPG) | Test MAPE (%) | Train $R^2$ | Test $R^2$ | Adjusted $R^2$ | 5-Fold CV $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Mean)** | 6.685 | 5.955 | 7.347 | 30.42% | 0.0000 | -0.0040 | -0.1172 | -0.0193 ± 0.015 |
| **Simple Linear Regr (Weight)** | 3.359 | 3.118 | 3.859 | 14.26% | 0.6845 | 0.7230 | 0.7194 | 0.6738 ± 0.026 |
| **Multiple Linear Regr** | 2.605 | 2.288 | 2.888 | 11.63% | 0.8189 | 0.8449 | 0.8274 | 0.8050 ± 0.021 |
| **Polynomial Regr (Degree 2)** | 1.947 | 1.780 | 2.430 | 8.26% | 0.8855 | 0.8902 | 0.8332 | 0.8518 ± 0.032 |
| **Ridge Regression ($L_2$)** | 2.593 | 2.277 | 2.886 | 11.52% | 0.8185 | 0.8451 | 0.8277 | 0.8048 ± 0.020 |
| **Lasso Regression ($L_1$)** | 2.589 | 2.276 | 2.900 | 11.50% | 0.8179 | 0.8436 | 0.8260 | 0.8036 ± 0.019 |
| **Decision Tree Regressor** | 1.938 | 2.330 | 3.372 | 10.38% | 0.8888 | 0.7885 | 0.7647 | 0.7728 ± 0.047 |
| **Random Forest Regressor** | **1.018** | **1.583** | **2.144** | **7.14%** | **0.9681** | **0.9145** | **0.9049** | **0.8441 ± 0.034** |
| **Gradient Boosting Regressor** | 1.134 | 1.704 | 2.281 | 7.89% | 0.9666 | 0.9033 | 0.8924 | 0.8417 ± 0.036 |

### Key Analytical Takeaways:
- **Weight Explains >72% of Variance Alone**: The Simple Linear Regression model using only vehicle weight attains $R^2 = 0.723$, verifying that curb mass is the single most influential engineering design factor for vehicle fuel economy.
- **Multicollinearity Shrinkage**: In Multiple Linear Regression, high correlation between displacement, cylinders, and horsepower is stabilized by Ridge Regression ($\alpha = 2.0434$), yielding superior generalizability.
- **Superiority of Non-Linear Ensembles**: The **Random Forest Regressor** achieves the best overall performance ($R^2 = 0.9145$, $\text{MAE} = 1.58\text{ MPG}$, $\text{MAPE} = 7.14\%$), effectively modeling subtle non-linearities and technological efficiency gains over model years.

---

## 🚀 5. How to Run

### Step 1: Environment Setup
Ensure Python 3.8+ is installed with the required libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib scipy
```

### Step 2: Run Full Training & Diagnostic Pipeline
Executes data loading, median imputation, model benchmarking, diagnostic plot generation, and pipeline serialization:
```bash
python mileage_prediction.py
```

### Step 3: Run Interactive Vehicle Mileage Predictor
Predicts fuel economy and calculates annual fuel expenses for vehicle specifications:
```bash
python interactive_inference.py
```

### Step 4: Open Interactive Jupyter Notebook
Launch Jupyter Notebook to interactively explore each code step and visualizations:
```bash
jupyter notebook lab1_car_mileage_estimation.ipynb
```

---

## 🚘 6. Sample Interactive Inference Results

```text
================================================================================
Vehicle: Japanese Economy Compact (e.g., Honda Civic / Toyota Corolla)
Specs: cylinders=4, displacement=98 cu.in., HP=68, weight=2050 lbs, accel=16.2s, year=82, origin=japan
--> Estimated Mileage: 34.83 US MPG  |  14.81 km/L
--> Annual Fuel Need : 1,620.9 Liters (430.7 Gallons)
--> Annual Fuel Cost : INR 155,602.02 ($1,550.47)
--> Estimated CO2    : 3,744.2 kg CO2/year

--------------------------------------------------------------------------------
Vehicle: American Heavy Muscle / Utility V8 (e.g., Chevrolet / Ford)
Specs: cylinders=8, displacement=350 cu.in., HP=165, weight=4140 lbs, accel=12.0s, year=74, origin=usa
--> Estimated Mileage: 14.22 US MPG  |  6.04 km/L
--> Annual Fuel Need : 3,971.0 Liters (1,055.1 Gallons)
--> Annual Fuel Cost : INR 381,211.74 ($3,798.51)
--> Estimated CO2    : 9,172.9 kg CO2/year
================================================================================
```
