"""
mileage_prediction.py
---------------------
Tata Technologies - TechPulse FY-26: Applied AI & ML
Lab Statement 1: ML Model for Car Mileage Estimation
Predict car mileage using regression and Python libraries using the inbuilt 'mpg' dataset.

Curriculum Context:
- Unit 1: Introduction to AI & ML, ML lifecycle overview, Model evaluation basics, Python libraries.
- Unit 2: Machine Learning & Applications, Preprocessing, Missing values, Feature scaling and encoding,
          Linear Regression for mileage prediction, Decision Trees, Random Forests, Evaluating model performance.
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Publication aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.autolayout": True,
    "figure.dpi": 200,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

def calculate_adjusted_r2(r2, n_samples, n_features):
    """Calculates Adjusted R-squared: 1 - [(1 - R²)(n - 1) / (n - k - 1)]"""
    if n_samples <= n_features + 1:
        return np.nan
    return 1.0 - ((1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1))

def calculate_metrics(y_true, y_pred, n_features):
    """Computes comprehensive regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    r2 = r2_score(y_true, y_pred)
    adj_r2 = calculate_adjusted_r2(r2, len(y_true), n_features)
    
    return {
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "MAPE (%)": round(mape, 2),
        "R²": round(r2, 4),
        "Adjusted R²": round(adj_r2, 4)
    }

def run_mileage_pipeline(base_dir):
    print("=" * 80, flush=True)
    print("TATA TECHNOLOGIES - TECHPULSE FY-26: APPLIED AI & ML", flush=True)
    print("LAB STATEMENT 1: ML MODEL FOR CAR MILEAGE ESTIMATION", flush=True)
    print("DATASET: INBUILT SEABORN 'mpg' AUTOMOTIVE BENCHMARK DATASET", flush=True)
    print("=" * 80, flush=True)
    
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # Step 1: Ingest Inbuilt MPG Dataset
    # -------------------------------------------------------------
    local_csv = os.path.join(base_dir, "auto_mpg.csv")
    print("\n[1/8] Ingesting Inbuilt 'mpg' Dataset...", flush=True)
    
    try:
        df = sns.load_dataset("mpg")
        df.to_csv(local_csv, index=False)
        print(f"-> Loaded via seaborn.load_dataset('mpg') and cached locally to {local_csv}", flush=True)
    except Exception as e:
        print(f"-> Remote fetch failed ({e}), loading from local copy {local_csv}...", flush=True)
        df = pd.read_csv(local_csv)
        
    print(f"-> Total records: {df.shape[0]} | Columns: {df.shape[1]}", flush=True)
    print("-> Features:", list(df.columns), flush=True)
    
    # Dual target representation: MPG and km/L (SI metric: 1 US MPG = 0.4251437 km/L)
    df["kmpl"] = np.round(df["mpg"] * 0.4251437, 2)
    
    # Check missing values
    missing_info = df.isnull().sum()
    print("-> Missing value audit per feature:", flush=True)
    for col, count in missing_info.items():
        if count > 0:
            print(f"   * {col}: {count} missing records ({(count/len(df))*100:.2f}%)", flush=True)
            
    # Handle missing values: Horsepower has 6 missing values.
    # Following industrial best practice (Unit 2: Handling missing values), impute with median.
    median_hp = df["horsepower"].median()
    df["horsepower"] = df["horsepower"].fillna(median_hp)
    print(f"-> Imputed missing 'horsepower' values with median = {median_hp:.1f} HP.", flush=True)
    
    # -------------------------------------------------------------
    # Step 2: Exploratory Data Analysis & Visualizations
    # -------------------------------------------------------------
    print("\n[2/8] Performing Exploratory Data Analysis (EDA)...", flush=True)
    
    numeric_features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
    categorical_features = ["origin"]
    
    # Plot 1: Target Distribution and Pearson Correlation Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    sns.histplot(df["mpg"], kde=True, color="#1f77b4", ax=axes[0], bins=22, edgecolor="black", alpha=0.65)
    axes[0].axvline(df["mpg"].mean(), color="crimson", linestyle="--", linewidth=2, label=f"Mean: {df['mpg'].mean():.2f} MPG")
    axes[0].axvline(df["mpg"].median(), color="darkgreen", linestyle=":", linewidth=2, label=f"Median: {df['mpg'].median():.2f} MPG")
    axes[0].set_title("Distribution of Vehicle Mileage (MPG & km/L)", pad=10)
    axes[0].set_xlabel("Fuel Economy (Miles Per Gallon - MPG)")
    axes[0].set_ylabel("Frequency Count")
    axes[0].legend()
    
    # Secondary X-axis for km/L
    secax = axes[0].secondary_xaxis('top', functions=(lambda x: x * 0.4251437, lambda x: x / 0.4251437))
    secax.set_xlabel("Equivalent Fuel Economy (km/L)")
    
    # Correlation Matrix
    corr_cols = numeric_features + ["mpg"]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                linewidths=0.6, ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Pearson Correlation Heatmap (Auto MPG Features)", pad=10)
    
    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "01_eda_correlation_mileage_distribution.png")
    plt.savefig(plot1_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved EDA Distribution & Correlation plot: {plot1_path}", flush=True)
    
    # Plot 2: Key Automotive Engineering Predictors vs Mileage
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    scatter_specs = [
        ("weight", "Vehicle Weight (lbs)", axes[0, 0], "#2b5c8f"),
        ("displacement", "Displacement (cu. in.)", axes[0, 1], "#d95f02"),
        ("horsepower", "Horsepower (HP)", axes[1, 0], "#7570b3"),
        ("acceleration", "Acceleration 0-60 mph (sec)", axes[1, 1], "#1b9e77")
    ]
    
    for col, xlabel, ax, color in scatter_specs:
        sns.regplot(data=df, x=col, y="mpg", ax=ax,
                    scatter_kws={"alpha": 0.5, "color": color, "s": 28},
                    line_kws={"color": "crimson", "linewidth": 2, "label": "Linear Fit"},
                    order=1)
        r_val, _ = stats.pearsonr(df[col], df["mpg"])
        ax.set_title(f"{xlabel} vs Mileage (r = {r_val:.2f})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mileage (MPG)")
        ax.legend()
        
    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "02_scatter_key_features_vs_mileage.png")
    plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved Engineering Relationships plot: {plot2_path}", flush=True)
    
    # -------------------------------------------------------------
    # Step 3: Dataset Splitting & Preprocessing Architecture
    # -------------------------------------------------------------
    print("\n[3/8] Partitioning Dataset into Train (80%) and Test (20%)...", flush=True)
    X = df[numeric_features + categorical_features]
    y = df["mpg"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"-> Training Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples", flush=True)
    
    print("\n[4/8] Constructing Scikit-Learn Preprocessing Pipelines...", flush=True)
    
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    
    preprocessor.fit(X_train)
    cat_names = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_names
    print(f"-> Transformed feature space: {len(all_feature_names)} features: {all_feature_names}", flush=True)
    
    # -------------------------------------------------------------
    # Step 5: Regression Models Training & Benchmarking
    # -------------------------------------------------------------
    print("\n[5/8] Training Regression Models...", flush=True)
    
    models = {}
    
    # 1. Baseline Mean Regressor
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    models["Baseline (Mean)"] = dummy
    
    # 2. Simple Linear Regression (Weight -> MPG)
    slr_pipeline = Pipeline([
        ("select_weight", ColumnTransformer([
            ("weight_scale", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), ["weight"])
        ], remainder="drop")),
        ("regressor", LinearRegression())
    ])
    slr_pipeline.fit(X_train, y_train)
    models["Simple Linear Regr (Weight)"] = slr_pipeline
    
    # 3. Multiple Linear Regression (MLR)
    mlr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    mlr_pipeline.fit(X_train, y_train)
    models["Multiple Linear Regr"] = mlr_pipeline
    
    # 4. Polynomial Regression (Degree 2 with interactions on physical variables)
    poly_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False))
            ]), numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    poly_pipeline = Pipeline([
        ("preprocessor", poly_preprocessor),
        ("regressor", RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5))
    ])
    poly_pipeline.fit(X_train, y_train)
    models["Polynomial Regr (Degree 2)"] = poly_pipeline
    
    # 5. Ridge Regression (L2 Regularization)
    ridge_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5))
    ])
    ridge_pipeline.fit(X_train, y_train)
    best_ridge_alpha = ridge_pipeline.named_steps["regressor"].alpha_
    models["Ridge Regression (L2)"] = ridge_pipeline
    print(f"   * Optimal Ridge Alpha (CV): {best_ridge_alpha:.4f}", flush=True)
    
    # 6. Lasso Regression (L1 Regularization & Sparsity)
    lasso_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, max_iter=3000, random_state=42))
    ])
    lasso_pipeline.fit(X_train, y_train)
    best_lasso_alpha = lasso_pipeline.named_steps["regressor"].alpha_
    models["Lasso Regression (L1)"] = lasso_pipeline
    print(f"   * Optimal Lasso Alpha (CV): {best_lasso_alpha:.4f}", flush=True)
    
    # 7. Decision Tree Regressor (Depth pruned to prevent overfitting)
    dt_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(max_depth=4, min_samples_split=8, random_state=42))
    ])
    dt_pipeline.fit(X_train, y_train)
    models["Decision Tree Regr"] = dt_pipeline
    
    # 8. Random Forest Regressor (Ensemble)
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42, n_jobs=1))
    ])
    rf_pipeline.fit(X_train, y_train)
    models["Random Forest Regr"] = rf_pipeline
    
    # 9. Gradient Boosting Regressor
    gb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42))
    ])
    gb_pipeline.fit(X_train, y_train)
    models["Gradient Boosting Regr"] = gb_pipeline
    
    # -------------------------------------------------------------
    # Step 6: Model Evaluation & Benchmarking Matrix
    # -------------------------------------------------------------
    print("\n[6/8] Evaluating Models on Test Set & Cross-Validation...", flush=True)
    
    results = []
    predictions_dict = {}
    
    for name, model in models.items():
        y_test_pred = model.predict(X_test)
        predictions_dict[name] = y_test_pred
        
        n_feat = len(all_feature_names)
        if "Simple" in name:
            n_feat = 1
        elif "Polynomial" in name:
            n_feat = 27
            
        test_metrics = calculate_metrics(y_test, y_test_pred, n_features=n_feat)
        
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        # 5-Fold Cross Validation R2 on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        
        results.append({
            "Model": name,
            "Train MAE": round(train_mae, 3),
            "Test MAE": test_metrics["MAE"],
            "Test RMSE": test_metrics["RMSE"],
            "Test MAPE (%)": test_metrics["MAPE (%)"],
            "Train R²": round(train_r2, 4),
            "Test R²": test_metrics["R²"],
            "Adjusted R²": test_metrics["Adjusted R²"],
            "CV R² Mean": round(cv_scores.mean(), 4),
            "CV R² Std": round(cv_scores.std(), 4)
        })
        
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 105, flush=True)
    print("MODEL BENCHMARK RESULTS TABLE (TARGET: MPG)", flush=True)
    print("=" * 105, flush=True)
    print(results_df.to_string(index=False), flush=True)
    print("=" * 105, flush=True)
    
    results_csv = os.path.join(base_dir, "model_benchmark_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"-> Benchmark results saved to: {results_csv}", flush=True)
    
    # -------------------------------------------------------------
    # Step 7: Diagnostic Visualizations
    # -------------------------------------------------------------
    print("\n[7/8] Generating Diagnostic & Comparative Visualizations...", flush=True)
    
    # Plot 3: Simple vs Multiple Linear Regression
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Simple Linear Regression Fit
    weight_linspace = np.linspace(df["weight"].min(), df["weight"].max(), 200)
    slr_pred = slr_pipeline.predict(pd.DataFrame({"weight": weight_linspace}))
    
    axes[0].scatter(X_test["weight"], y_test, color="#3498db", alpha=0.6, label="Actual Test Vehicles")
    axes[0].plot(weight_linspace, slr_pred, color="crimson", linewidth=2.5,
                 label=f"SLR Fit Line (R² = {results_df.loc[results_df['Model']=='Simple Linear Regr (Weight)', 'Test R²'].values[0]:.3f})")
    axes[0].set_title("Simple Linear Regression: Weight vs Mileage", pad=10)
    axes[0].set_xlabel("Vehicle Weight (lbs)")
    axes[0].set_ylabel("Mileage (MPG)")
    axes[0].legend()
    
    # Multiple Linear Regression Actual vs Predicted
    mlr_pred = predictions_dict["Multiple Linear Regr"]
    axes[1].scatter(y_test, mlr_pred, color="#2ecc71", alpha=0.65, edgecolors="black", linewidth=0.5, label="Predictions")
    min_v = min(y_test.min(), mlr_pred.min()) - 1
    max_v = max(y_test.max(), mlr_pred.max()) + 1
    axes[1].plot([min_v, max_v], [min_v, max_v], color="crimson", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
    axes[1].set_title(f"Multiple Linear Regression: Actual vs Predicted (R² = {results_df.loc[results_df['Model']=='Multiple Linear Regr', 'Test R²'].values[0]:.3f})", pad=10)
    axes[1].set_xlabel("Actual Mileage (MPG)")
    axes[1].set_ylabel("Predicted Mileage (MPG)")
    axes[1].legend()
    
    plt.tight_layout()
    plot3_path = os.path.join(plots_dir, "03_simple_vs_multiple_linear_regression.png")
    plt.savefig(plot3_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved Simple vs Multiple Regression plot: {plot3_path}", flush=True)
    
    # Plot 4: Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    comp_df = results_df[results_df["Model"] != "Baseline (Mean)"].copy()
    bar_colors = ["#34495e", "#2980b9", "#16a085", "#8e44ad", "#d35400", "#e67e22", "#27ae60", "#2c3e50"]
    
    # Test R2
    axes[0].barh(comp_df["Model"], comp_df["Test R²"], color=bar_colors[:len(comp_df)], edgecolor="black", alpha=0.85)
    axes[0].set_title("Test R² Score (Higher is Better)")
    axes[0].set_xlim(0, 1.0)
    for i, v in enumerate(comp_df["Test R²"]):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold", fontsize=8)
        
    # Test MAE
    axes[1].barh(comp_df["Model"], comp_df["Test MAE"], color=bar_colors[:len(comp_df)], edgecolor="black", alpha=0.85)
    axes[1].set_title("Test MAE (MPG) (Lower is Better)")
    axes[1].set_xlim(0, max(comp_df["Test MAE"]) * 1.3)
    for i, v in enumerate(comp_df["Test MAE"]):
        axes[1].text(v + 0.04, i, f"{v:.2f}", va="center", fontweight="bold", fontsize=8)
        
    # Test RMSE
    axes[2].barh(comp_df["Model"], comp_df["Test RMSE"], color=bar_colors[:len(comp_df)], edgecolor="black", alpha=0.85)
    axes[2].set_title("Test RMSE (MPG) (Lower is Better)")
    axes[2].set_xlim(0, max(comp_df["Test RMSE"]) * 1.3)
    for i, v in enumerate(comp_df["Test RMSE"]):
        axes[2].text(v + 0.04, i, f"{v:.2f}", va="center", fontweight="bold", fontsize=8)
        
    plt.tight_layout()
    plot4_path = os.path.join(plots_dir, "04_model_performance_comparison.png")
    plt.savefig(plot4_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved Model Performance Comparison plot: {plot4_path}", flush=True)
    
    # Plot 5: 4-Quadrant Residual Diagnostics
    best_row = results_df.sort_values(by="Test R²", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_model = models[best_name]
    best_preds = predictions_dict[best_name]
    residuals = y_test.values - best_preds
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Residuals vs Fitted (Homoscedasticity)
    axes[0, 0].scatter(best_preds, residuals, color="#2980b9", alpha=0.6, edgecolors="black", linewidth=0.5)
    axes[0, 0].axhline(0, color="crimson", linestyle="--", linewidth=2)
    sorted_order = np.argsort(best_preds)
    rolling_mean = pd.Series(residuals[sorted_order]).rolling(window=15, center=True).mean()
    axes[0, 0].plot(best_preds[sorted_order], rolling_mean, color="orange", linewidth=2.5, label="Local Trend")
    axes[0, 0].set_title(f"Residuals vs Fitted Values ({best_name})", pad=8)
    axes[0, 0].set_xlabel("Fitted Values (Predicted MPG)")
    axes[0, 0].set_ylabel("Residuals (MPG)")
    axes[0, 0].legend()
    
    # Q-Q Plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[0, 1].plot(osm, osr, "o", color="#8e44ad", alpha=0.65, markersize=5)
    axes[0, 1].plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=2, label=f"Normal Fit (R² = {r**2:.3f})")
    axes[0, 1].set_title("Normal Q-Q Plot of Residuals", pad=8)
    axes[0, 1].set_xlabel("Theoretical Quantiles")
    axes[0, 1].set_ylabel("Sample Quantiles")
    axes[0, 1].legend()
    
    # Residual Histogram
    sns.histplot(residuals, kde=True, color="#27ae60", ax=axes[1, 0], bins=20, edgecolor="black", alpha=0.6)
    axes[1, 0].axvline(0, color="crimson", linestyle="--", linewidth=2, label="Mean Residual = 0.00")
    axes[1, 0].set_title(f"Residual Distribution (μ={residuals.mean():.2f}, σ={residuals.std():.2f})", pad=8)
    axes[1, 0].set_xlabel("Residual Error (Actual - Predicted MPG)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    
    # Actual vs Predicted with +/- 3 MPG band
    axes[1, 1].scatter(y_test, best_preds, color="#e67e22", alpha=0.65, edgecolors="black", linewidth=0.5, label="Predictions")
    min_b = min(y_test.min(), best_preds.min()) - 1
    max_b = max(y_test.max(), best_preds.max()) + 1
    axes[1, 1].plot([min_b, max_b], [min_b, max_b], "k--", linewidth=2, label="Ideal y = x")
    axes[1, 1].fill_between([min_b, max_b], [min_b - 3.0, max_b - 3.0], [min_b + 3.0, max_b + 3.0],
                            color="gray", alpha=0.15, label="±3.0 MPG Margin")
    axes[1, 1].set_title(f"Actual vs Predicted Mileage ({best_name})", pad=8)
    axes[1, 1].set_xlabel("Actual Mileage (MPG)")
    axes[1, 1].set_ylabel("Predicted Mileage (MPG)")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plot5_path = os.path.join(plots_dir, "05_residual_diagnostics_homoscedasticity.png")
    plt.savefig(plot5_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved Residual Diagnostics plot: {plot5_path}", flush=True)
    
    # Plot 6: Linear Regression Coefficients vs Random Forest Feature Importances
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # MLR Coefficients
    mlr_coeffs = mlr_pipeline.named_steps["regressor"].coef_
    coeff_series = pd.Series(mlr_coeffs, index=all_feature_names).sort_values()
    colors_c = ["#e74c3c" if c < 0 else "#2ecc71" for c in coeff_series]
    
    axes[0].barh(coeff_series.index, coeff_series.values, color=colors_c, edgecolor="black", alpha=0.85)
    axes[0].axvline(0, color="black", linestyle="-", linewidth=0.8)
    axes[0].set_title("Standardized Multiple Linear Regression Coefficients (β)", pad=10)
    axes[0].set_xlabel("Standardized Coefficient Value (Change in MPG per 1σ)")
    for i, v in enumerate(coeff_series.values):
        axes[0].text(v + (0.1 if v >= 0 else -0.5), i, f"{v:.2f}", va="center", fontsize=8, fontweight="bold")
        
    # RF Feature Importances
    rf_importances = rf_pipeline.named_steps["regressor"].feature_importances_
    rf_series = pd.Series(rf_importances, index=all_feature_names).sort_values(ascending=True)
    
    axes[1].barh(rf_series.index, rf_series.values, color="#3498db", edgecolor="black", alpha=0.85)
    axes[1].set_title("Random Forest Gini Feature Importances (MDI)", pad=10)
    axes[1].set_xlabel("Relative Importance Score")
    for i, v in enumerate(rf_series.values):
        axes[1].text(v + 0.005, i, f"{v*100:.1f}%", va="center", fontsize=8, fontweight="bold")
        
    plt.tight_layout()
    plot6_path = os.path.join(plots_dir, "06_feature_importance_coefficients.png")
    plt.savefig(plot6_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved Feature Importance & Coefficients plot: {plot6_path}", flush=True)
    
    # -------------------------------------------------------------
    # Step 8: Serialization & Inference Verification
    # -------------------------------------------------------------
    print("\n[8/8] Serializing Production Pipelines & Artifacts...", flush=True)
    
    # Best Model Artifact
    best_model_file = os.path.join(base_dir, "best_car_mileage_model.joblib")
    joblib.dump({
        "model_name": best_name,
        "pipeline": best_model,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_names": all_feature_names,
        "median_imputer_horsepower": median_hp,
        "target": "mpg"
    }, best_model_file)
    print(f"-> Best Model ('{best_name}') serialized to: {best_model_file}", flush=True)
    
    # MLR Pipeline Artifact
    mlr_model_file = os.path.join(base_dir, "multiple_linear_regression_pipeline.joblib")
    joblib.dump({
        "model_name": "Multiple Linear Regr",
        "pipeline": mlr_pipeline,
        "feature_names": all_feature_names,
        "coefficients": dict(zip(all_feature_names, mlr_coeffs)),
        "intercept": mlr_pipeline.named_steps["regressor"].intercept_
    }, mlr_model_file)
    print(f"-> Multiple Linear Regression pipeline serialized to: {mlr_model_file}", flush=True)
    
    # Verification test inference on a sample vehicle
    sample_car = pd.DataFrame([{
        "cylinders": 4,
        "displacement": 140.0,
        "horsepower": 90.0,
        "weight": 2400.0,
        "acceleration": 15.5,
        "model_year": 78,
        "origin": "japan"
    }])
    
    pred_mpg_val = best_model.predict(sample_car)[0]
    pred_kmpl_val = pred_mpg_val * 0.4251437
    
    print("\n" + "=" * 65, flush=True)
    print("VERIFICATION INFERENCE (SAMPLE VEHICLE: 4-CYL, 2400 LBS):", flush=True)
    print(f"-> Predicted Mileage: {pred_mpg_val:.2f} MPG ({pred_kmpl_val:.2f} km/L)", flush=True)
    print("=" * 65, flush=True)
    
    return {
        "best_model_name": best_name,
        "results_df": results_df,
        "features": all_feature_names
    }

if __name__ == "__main__":
    base_dir = r"C:\Users\abhin\.gemini\antigravity-ide\scratch\lab1_car_mileage_estimation"
    run_mileage_pipeline(base_dir)
    print("\nLab Statement 1 pipeline execution completed successfully!", flush=True)
