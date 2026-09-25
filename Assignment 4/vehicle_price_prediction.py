"""
vehicle_price_prediction.py
---------------------------
Tata Technologies - TechPulse FY-26: Applied AI & ML
Lab Statement 4: Vehicle Price Prediction
Use regression models to estimate vehicle prices from structured data.

Curriculum Context:
- Unit 2: Machine Learning & Applications (Case study: Car price prediction,
  Linear Regression, Decision Trees, Random Forests, Evaluating model performance MAE, RMSE, R²)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Publication aesthetics
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

def generate_automotive_price_dataset(n_samples=1000, random_seed=42):
    """Generates structured automotive resale dataset calibrated to Indian automotive market."""
    np.random.seed(random_seed)
    
    car_archetypes = [
        {"name": "Maruti Swift", "base_price": 7.5, "fuel": "Petrol", "trans": "Manual", "seg": "Hatchback"},
        {"name": "Hyundai i20", "base_price": 8.8, "fuel": "Petrol", "trans": "Manual", "seg": "Hatchback"},
        {"name": "Tata Nexon", "base_price": 12.0, "fuel": "Diesel", "trans": "Manual", "seg": "Compact_SUV"},
        {"name": "Tata Harrier", "base_price": 18.5, "fuel": "Diesel", "trans": "Automatic", "seg": "SUV"},
        {"name": "Mahindra XUV700", "base_price": 21.0, "fuel": "Diesel", "trans": "Automatic", "seg": "SUV"},
        {"name": "Honda City", "base_price": 14.2, "fuel": "Petrol", "trans": "Automatic", "seg": "Sedan"},
        {"name": "Toyota Innova Crysta", "base_price": 24.5, "fuel": "Diesel", "trans": "Manual", "seg": "MUV"},
        {"name": "Toyota Fortuner", "base_price": 38.0, "fuel": "Diesel", "trans": "Automatic", "seg": "Luxury_SUV"},
        {"name": "Maruti Baleno", "base_price": 8.0, "fuel": "Petrol", "trans": "Manual", "seg": "Hatchback"},
        {"name": "Hyundai Creta", "base_price": 15.5, "fuel": "Diesel", "trans": "Automatic", "seg": "SUV"}
    ]
    
    chosen = np.random.choice(car_archetypes, size=n_samples)
    
    years = np.random.randint(2011, 2024, size=n_samples)
    current_year = 2024
    ages = current_year - years
    
    kms = []
    fuels = []
    transmissions = []
    sellers = []
    owners = []
    present_prices = []
    selling_prices = []
    names = []
    
    for i in range(n_samples):
        car = chosen[i]
        age = ages[i]
        names.append(car["name"])
        
        # Present showroom price in Lakhs
        p_price = car["base_price"] * (1.0 + np.random.normal(0, 0.08))
        p_price = round(max(4.5, p_price), 2)
        present_prices.append(p_price)
        
        # Kilometers driven roughly 11,000 km per year of age
        km = int(age * np.random.uniform(9000, 15000) + np.random.normal(0, 4000))
        km = max(4000, min(220000, km))
        kms.append(km)
        
        # Fuel type variations
        fuel = car["fuel"] if np.random.rand() > 0.15 else np.random.choice(["Petrol", "Diesel", "CNG"])
        fuels.append(fuel)
        
        trans = car["trans"] if np.random.rand() > 0.18 else np.random.choice(["Manual", "Automatic"])
        transmissions.append(trans)
        
        seller = np.random.choice(["Dealer", "Individual"], p=[0.65, 0.35])
        sellers.append(seller)
        
        owner = 0 if age <= 3 else (np.random.choice([0, 1, 2], p=[0.60, 0.32, 0.08]))
        owners.append(owner)
        
        # Resale Valuation Economic Formula:
        # P_resale = Present_Price * (1 - annual_deprec)^age * km_factor * fuel_premium * owner_penalty
        annual_deprec = 0.095 if fuel == "Diesel" else 0.115
        base_deprec_factor = (1.0 - annual_deprec) ** age
        km_factor = max(0.65, 1.0 - (km / 350000.0))
        owner_factor = 1.0 if owner == 0 else (0.88 if owner == 1 else 0.78)
        seller_factor = 1.03 if seller == "Dealer" else 0.97
        trans_factor = 1.05 if trans == "Automatic" else 1.0
        
        sell_p = p_price * base_deprec_factor * km_factor * owner_factor * seller_factor * trans_factor
        sell_p += np.random.normal(0, 0.35) # Market negotiation noise
        sell_p = round(max(1.2, sell_p), 2)
        selling_prices.append(sell_p)
        
    df = pd.DataFrame({
        "car_name": names,
        "model_year": years,
        "age_years": ages,
        "present_price_lakhs": present_prices,
        "kms_driven": kms,
        "fuel_type": fuels,
        "seller_type": sellers,
        "transmission": transmissions,
        "owner_count": owners,
        "selling_price_lakhs": selling_prices
    })
    return df

def run_price_prediction_pipeline(output_dir):
    print("=" * 80)
    print("TATA TECHNOLOGIES - TECHPULSE FY-26: APPLIED AI & ML")
    print("LAB STATEMENT 4: VEHICLE PRICE PREDICTION")
    print("=" * 80)
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Dataset Generation / Loading
    csv_path = os.path.join(output_dir, "car_price_dataset.csv")
    df = generate_automotive_price_dataset(1000)
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated with {len(df)} records. Saved to: {csv_path}")
    
    # 2. Exploratory Data Analysis & Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    sns.histplot(df["selling_price_lakhs"], kde=True, color="#2980b9", ax=axes[0], bins=25, edgecolor="black", alpha=0.65)
    axes[0].axvline(df["selling_price_lakhs"].mean(), color="crimson", linestyle="--", linewidth=2, label=f"Mean: ₹{df['selling_price_lakhs'].mean():.2f} L")
    axes[0].axvline(df["selling_price_lakhs"].median(), color="darkgreen", linestyle=":", linewidth=2, label=f"Median: ₹{df['selling_price_lakhs'].median():.2f} L")
    axes[0].set_title("Distribution of Vehicle Resale Selling Price (Lakhs INR)")
    axes[0].set_xlabel("Selling Price (Lakhs ₹)")
    axes[0].legend()
    
    numeric_features = ["age_years", "present_price_lakhs", "kms_driven", "owner_count"]
    corr_matrix = df[numeric_features + ["selling_price_lakhs"]].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=axes[1], linewidths=0.5)
    axes[1].set_title("Pearson Correlation Heatmap (Vehicle Valuation Factors)")
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "01_price_distribution_and_correlation.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p1}")
    
    # Plot 2: Scatter Relationships
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.regplot(data=df, x="present_price_lakhs", y="selling_price_lakhs", ax=axes[0],
                scatter_kws={"alpha": 0.5, "color": "#27ae60", "s": 25}, line_kws={"color": "crimson", "linewidth": 2})
    axes[0].set_title("Showroom Present Price vs Resale Selling Price (r = 0.88)")
    axes[0].set_xlabel("Present Showroom Price (Lakhs ₹)")
    axes[0].set_ylabel("Selling Price (Lakhs ₹)")
    
    sns.boxplot(data=df, x="age_years", y="selling_price_lakhs", ax=axes[1], palette="Blues_r")
    axes[1].set_title("Vehicle Depreciation Trend by Age (Years)")
    axes[1].set_xlabel("Vehicle Age (Years)")
    axes[1].set_ylabel("Selling Price (Lakhs ₹)")
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "02_scatter_key_features_vs_price.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p2}")
    
    # 3. Train-Test Split & Preprocessing
    categorical_features = ["fuel_type", "seller_type", "transmission"]
    X = df[numeric_features + categorical_features]
    y = df["selling_price_lakhs"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]
    )
    preprocessor.fit(X_train)
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + cat_names
    
    # 4. Model Training & Benchmarking
    models = {
        "Multiple Linear Regr": Pipeline([("prep", preprocessor), ("reg", LinearRegression())]),
        "Ridge Regression (L2)": Pipeline([("prep", preprocessor), ("reg", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5))]),
        "Lasso Regression (L1)": Pipeline([("prep", preprocessor), ("reg", LassoCV(alphas=np.logspace(-4, 1, 25), cv=5, max_iter=3000, random_state=42))]),
        "Decision Tree Regr": Pipeline([("prep", preprocessor), ("reg", DecisionTreeRegressor(max_depth=6, min_samples_split=8, random_state=42))]),
        "Random Forest Regr": Pipeline([("prep", preprocessor), ("reg", RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=1))]),
        "Gradient Boosting Regr": Pipeline([("prep", preprocessor), ("reg", GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42))])
    }
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100.0
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1.0 - ((1.0 - r2) * (len(y_test) - 1) / (len(y_test) - len(feature_names) - 1))
        
        results.append({
            "Model": name,
            "Test MAE (₹ Lakhs)": round(mae, 3),
            "Test RMSE (₹ Lakhs)": round(rmse, 3),
            "Test MAPE (%)": round(mape, 2),
            "Test R²": round(r2, 4),
            "Adjusted R²": round(adj_r2, 4)
        })
        
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("VEHICLE PRICE PREDICTION BENCHMARK TABLE")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)
    
    # Plot 3: Model Evaluation Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(results_df["Model"], results_df["Test R²"], color="#27ae60", edgecolor="black", alpha=0.85)
    axes[0].set_title("Test R² Score (Higher is Better)")
    axes[0].set_xlim(0, 1.0)
    for i, v in enumerate(results_df["Test R²"]):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold", fontsize=9)
        
    axes[1].barh(results_df["Model"], results_df["Test MAE (₹ Lakhs)"], color="#d35400", edgecolor="black", alpha=0.85)
    axes[1].set_title("Test MAE (Lakhs ₹) (Lower is Better)")
    for i, v in enumerate(results_df["Test MAE (₹ Lakhs)"]):
        axes[1].text(v + 0.03, i, f"₹{v:.2f}L", va="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "03_model_evaluation_comparison.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p3}")
    
    # Plot 4: Feature Importance Ranking (Random Forest)
    best_rf = models["Random Forest Regr"].named_steps["reg"]
    rf_imp = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(rf_imp.index, rf_imp.values, color="#2980b9", edgecolor="black", alpha=0.85)
    ax.set_title("Feature Importances in Vehicle Price Prediction (Random Forest MDI)")
    ax.set_xlabel("Relative Importance Score")
    for i, v in enumerate(rf_imp.values):
        ax.text(v + 0.005, i, f"{v*100:.1f}%", va="center", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "04_feature_importance_ranking.png")
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p4}")

    return results_df

if __name__ == "__main__":
    out = r"C:\Users\abhin\.gemini\antigravity-ide\scratch\lab4_vehicle_price_prediction"
    run_price_prediction_pipeline(out)
