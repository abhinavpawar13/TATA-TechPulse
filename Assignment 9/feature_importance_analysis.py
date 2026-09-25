"""
feature_importance_analysis.py
------------------------------
Tata Technologies - TechPulse FY-26: Applied AI & ML
Lab Statement 9: Feature Importance Visualization
Visualize and interpret feature importance in automotive datasets.

Curriculum Context:
- Unit 4: AI fundamentals & application development (Feature selection and importance,
  Gradient Boosting and XGBoost, Model optimization, Interpretable AI)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, mean_absolute_error

# Aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "figure.autolayout": True,
    "figure.dpi": 200,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "semibold"
})

def generate_automotive_powertrain_dataset(n_samples=1000, random_seed=42):
    """Generates automotive engineering dataset for powertrain efficiency and emissions analysis."""
    np.random.seed(random_seed)
    
    curb_weight_kg = np.random.normal(1450.0, 220.0, n_samples)
    curb_weight_kg = np.clip(curb_weight_kg, 920.0, 2350.0)
    
    engine_cc = np.random.normal(1650.0, 380.0, n_samples)
    engine_cc = np.clip(engine_cc, 998.0, 3200.0)
    
    horsepower = engine_cc * 0.076 + np.random.normal(0, 10.0, n_samples)
    horsepower = np.clip(horsepower, 65.0, 280.0)
    
    aero_cd = np.random.normal(0.31, 0.02, n_samples)
    aero_cd = np.clip(aero_cd, 0.26, 0.38)
    
    frontal_area_m2 = np.random.normal(2.20, 0.15, n_samples)
    frontal_area_m2 = np.clip(frontal_area_m2, 1.85, 2.75)
    
    tire_rolling_crr = np.random.normal(0.009, 0.001, n_samples)
    
    cylinders = np.where(engine_cc < 1250, 3, np.where(engine_cc < 2400, 4, 6))
    
    gear_ratios_count = np.random.choice([5, 6, 7, 8], size=n_samples, p=[0.2, 0.45, 0.25, 0.1])
    
    # Random uninformative noise feature to test whether feature importance methods reject spurious features
    random_noise = np.random.normal(50.0, 15.0, n_samples)
    
    # Target: Fuel Consumption Rate in Liters per 100km (L/100km)
    # Physically governed by weight, displacement, aerodynamics, and transmission gears
    l_100km = (
        0.0035 * curb_weight_kg + 
        0.0018 * engine_cc + 
        0.0120 * horsepower + 
        8.20 * (aero_cd * frontal_area_m2) + 
        120.0 * tire_rolling_crr - 
        0.28 * gear_ratios_count + 
        np.random.normal(0, 0.25, n_samples)
    )
    l_100km = np.clip(l_100km, 4.0, 16.5)
    
    df = pd.DataFrame({
        "curb_weight_kg": np.round(curb_weight_kg, 1),
        "engine_displacement_cc": np.round(engine_cc, 0),
        "horsepower_hp": np.round(horsepower, 1),
        "aerodynamic_cd": np.round(aero_cd, 3),
        "frontal_area_m2": np.round(frontal_area_m2, 2),
        "rolling_crr": np.round(tire_rolling_crr, 4),
        "cylinders": cylinders,
        "gear_count": gear_ratios_count,
        "uninformative_noise": np.round(random_noise, 2),
        "fuel_consumption_l_100km": np.round(l_100km, 2)
    })
    return df

def run_feature_importance_pipeline(output_dir):
    print("=" * 80)
    print("TATA TECHNOLOGIES - TECHPULSE FY-26: APPLIED AI & ML")
    print("LAB STATEMENT 9: FEATURE IMPORTANCE VISUALIZATION & INTERPRETABILITY")
    print("=" * 80)
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Dataset Generation
    df = generate_automotive_powertrain_dataset(1000)
    csv_path = os.path.join(output_dir, "automotive_features_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated automotive engineering dataset with {len(df)} records.")
    print(f"Saved to: {csv_path}")
    
    feature_cols = [
        "curb_weight_kg", "engine_displacement_cc", "horsepower_hp",
        "aerodynamic_cd", "frontal_area_m2", "rolling_crr",
        "cylinders", "gear_count", "uninformative_noise"
    ]
    target_col = "fuel_consumption_l_100km"
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 2. Train High-Capacity Random Forest & Gradient Boosting Regressors
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    
    gb = GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    
    print(f"Model Fits: Random Forest R² = {rf.score(X_test, y_test):.4f} | Gradient Boosting R² = {gb.score(X_test, y_test):.4f}")
    
    # -------------------------------------------------------------
    # 3. Method 1: Mean Decrease in Impurity (MDI / Gini Importance)
    # -------------------------------------------------------------
    mdi_rf = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    mdi_gb = pd.Series(gb.feature_importances_, index=feature_cols).sort_values()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].barh(mdi_rf.index, mdi_rf.values, color="#2980b9", edgecolor="black", alpha=0.85)
    axes[0].set_title("Random Forest MDI Feature Importance")
    axes[0].set_xlabel("Relative Importance Score")
    for i, v in enumerate(mdi_rf.values):
        axes[0].text(v + 0.005, i, f"{v*100:.1f}%", va="center", fontsize=8.5, fontweight="bold")
        
    axes[1].barh(mdi_gb.index, mdi_gb.values, color="#27ae60", edgecolor="black", alpha=0.85)
    axes[1].set_title("Gradient Boosting MDI Feature Importance")
    axes[1].set_xlabel("Relative Importance Score")
    for i, v in enumerate(mdi_gb.values):
        axes[1].text(v + 0.005, i, f"{v*100:.1f}%", va="center", fontsize=8.5, fontweight="bold")
        
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "01_mdi_gini_feature_importance.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p1}")
    
    # -------------------------------------------------------------
    # 4. Method 2: Permutation Feature Importance (Evaluated on Test Holdout)
    # -------------------------------------------------------------
    print("Computing Permutation Feature Importance on holdout test set...")
    perm_res = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=42)
    perm_series = pd.Series(perm_res.importances_mean, index=feature_cols).sort_values()
    perm_std = pd.Series(perm_res.importances_std, index=feature_cols)[perm_series.index]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(perm_series.index, perm_series.values, xerr=perm_std.values, color="#e67e22", edgecolor="black", alpha=0.85, capsize=4)
    ax.set_title("Holdout Permutation Feature Importance (Drop in R² when Shuffled)")
    ax.set_xlabel("Mean Decrease in Test Performance (R² Score)")
    for i, v in enumerate(perm_series.values):
        ax.text(v + 0.01, i, f"+{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
        
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "02_permutation_importance_test_set.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p2}")
    
    # -------------------------------------------------------------
    # 5. Method 3: Partial Dependence Plots (PDP)
    # -------------------------------------------------------------
    print("Generating Partial Dependence Plots for top engineering drivers...")
    top_features = ["curb_weight_kg", "engine_displacement_cc", "horsepower_hp", "aerodynamic_cd"]
    fig, ax = plt.subplots(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(rf, X_test, top_features, ax=ax, grid_resolution=30)
    plt.suptitle("Partial Dependence Plots (Non-Linear Response Curves on Fuel Consumption)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "03_partial_dependence_plots.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p3}")
    
    # -------------------------------------------------------------
    # 6. Method 4: Comprehensive Comparison (MDI vs Permutation vs Mutual Information)
    # -------------------------------------------------------------
    mi_scores = pd.Series(mutual_info_regression(X_train, y_train, random_state=42), index=feature_cols)
    
    comp_df = pd.DataFrame({
        "MDI (RF) %": np.round((mdi_rf / mdi_rf.sum()) * 100, 1),
        "Permutation %": np.round((np.maximum(0, perm_series) / np.maximum(0, perm_series).sum()) * 100, 1),
        "Mutual Info %": np.round((mi_scores / mi_scores.sum()) * 100, 1)
    }).sort_values(by="Permutation %", ascending=False)
    
    print("\n" + "=" * 65)
    print("FEATURE IMPORTANCE TECHNIQUES COMPARISON:")
    print("=" * 65)
    print(comp_df.to_string())
    print("=" * 65)
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    comp_df.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85)
    ax.set_title("Cross-Comparison of Feature Importance Frameworks", pad=10)
    ax.set_ylabel("Relative Normalized Importance (%)")
    ax.set_xlabel("Automotive Features")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "04_feature_importance_methods_comparison.png")
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p4}")

if __name__ == "__main__":
    out = r"C:\Users\abhin\.gemini\antigravity-ide\scratch\lab9_feature_importance_visualization"
    run_feature_importance_pipeline(out)
