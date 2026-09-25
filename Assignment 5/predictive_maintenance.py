"""
predictive_maintenance.py
-------------------------
Tata Technologies - TechPulse FY-26: Applied AI & ML
Lab Statement 5: Predictive Maintenance from Sensor Logs
Classify component failures using sensor data and Python ML models.

Curriculum Context:
- Unit 1: Automotive use cases (Predictive maintenance)
- Unit 4: AI fundamentals & application development (Classification metrics: precision, recall, F1-score, ROC-AUC)
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score

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

def generate_sensor_maintenance_dataset(n_samples=1200, random_seed=42):
    """Generates synthetic sensor telemetry calibrated to NASA / AI4I predictive maintenance."""
    np.random.seed(random_seed)
    
    air_temp = np.random.normal(300.0, 2.0, n_samples) # Kelvin
    process_temp = air_temp + np.random.normal(10.0, 1.0, n_samples)
    rot_speed = np.random.normal(1530.0, 180.0, n_samples) # RPM
    rot_speed = np.clip(rot_speed, 1150.0, 2880.0)
    torque = np.random.normal(40.0, 10.0, n_samples) # Nm
    torque = np.clip(torque, 12.0, 78.0)
    tool_wear = np.random.uniform(0.0, 240.0, n_samples) # Minutes
    
    failures = np.zeros(n_samples, dtype=int)
    failure_reasons = ["No_Failure"] * n_samples
    
    for i in range(n_samples):
        # Physical failure modes:
        # 1. Heat Dissipation Failure (HDF): temp difference < 8.6 K and speed < 1380 rpm
        temp_diff = process_temp[i] - air_temp[i]
        if temp_diff < 8.6 and rot_speed[i] < 1380:
            failures[i] = 1
            failure_reasons[i] = "Heat_Dissipation_Failure"
            continue
            
        # 2. Power Failure (PWF): Power = Torque * (Speed * 2pi/60). If Power < 3500W or > 9000W
        power_w = torque[i] * (rot_speed[i] * (2 * np.pi / 60.0))
        if power_w < 3500.0 or power_w > 9000.0:
            failures[i] = 1
            failure_reasons[i] = "Power_Failure"
            continue
            
        # 3. Overstrain Failure (OSF): Tool wear * Torque exceeds limit
        if tool_wear[i] * torque[i] > 11500.0:
            failures[i] = 1
            failure_reasons[i] = "Overstrain_Failure"
            continue
            
        # 4. Tool Wear Failure (TWF): Wear time > 205 min
        if tool_wear[i] > 210.0 and np.random.rand() > 0.4:
            failures[i] = 1
            failure_reasons[i] = "Tool_Wear_Failure"
            continue
            
        # Occasional random sensor spike
        if np.random.rand() < 0.015:
            failures[i] = 1
            failure_reasons[i] = "Mechanical_Glitch"
            
    df = pd.DataFrame({
        "device_id": [f"ECU_SENS_{i+1001:05d}" for i in range(n_samples)],
        "air_temperature_k": np.round(air_temp, 2),
        "process_temperature_k": np.round(process_temp, 2),
        "rotational_speed_rpm": np.round(rot_speed, 1),
        "torque_nm": np.round(torque, 2),
        "tool_wear_min": np.round(tool_wear, 1),
        "failure_mode": failure_reasons,
        "failure": failures
    })
    return df

def run_predictive_maintenance_pipeline(output_dir):
    print("=" * 80)
    print("TATA TECHNOLOGIES - TECHPULSE FY-26: APPLIED AI & ML")
    print("LAB STATEMENT 5: PREDICTIVE MAINTENANCE FROM SENSOR LOGS")
    print("=" * 80)
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Dataset Generation / Loading
    csv_path = os.path.join(output_dir, "sensor_maintenance_data.csv")
    df = generate_sensor_maintenance_dataset(1200)
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated with {len(df)} records. Saved to: {csv_path}")
    print(f"Class breakdown: Normal = {(df['failure']==0).sum()} | Failures = {(df['failure']==1).sum()} ({(df['failure'].mean())*100:.1f}%)")
    
    feature_cols = ["air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
    X = df[feature_cols]
    y = df["failure"]
    
    # 2. Exploratory Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x="failure_mode", ax=axes[0], palette="Set2")
    axes[0].set_title("Component Failure Class Distribution (Sensory Logs)")
    axes[0].set_xlabel("Failure Mode")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=25)
    
    corr = df[feature_cols + ["failure"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1], square=True, linewidths=0.5)
    axes[1].set_title("Sensor Feature Correlation Matrix with Failure")
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "01_sensor_distributions_and_failures.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p1}")
    
    # 3. Train-Test Split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # 4. Model Training & Benchmarking
    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", random_state=42))]),
        "Decision Tree Clf": Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42))]),
        "Random Forest Clf": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=1))]),
        "Support Vector Clf (RBF)": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))]),
        "Gradient Boosting Clf": Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42))])
    }
    
    results = []
    roc_curves = {}
    conf_matrices = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, auc)
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
        
        results.append({
            "Classifier": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })
        
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("PREDICTIVE MAINTENANCE CLASSIFIER BENCHMARK TABLE")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)
    
    # Plot 2: Confusion Matrix of Best Model (Gradient Boosting / Random Forest)
    best_name = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Classifier"]
    cm = conf_matrices[best_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Normal", "Failure"], yticklabels=["Normal", "Failure"])
    axes[0].set_title(f"Confusion Matrix: {best_name}")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # ROC-AUC Curves
    for name, (fpr, tpr, auc_val) in roc_curves.items():
        axes[1].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_title("ROC-AUC Curves Comparison")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "02_confusion_matrix_and_roc_curves.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p2}")
    
    # Plot 3: Feature Importance (Random Forest)
    rf_imp = pd.Series(models["Random Forest Clf"].named_steps["clf"].feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(rf_imp.index, rf_imp.values, color="#e67e22", edgecolor="black", alpha=0.85)
    ax.set_title("Sensor Feature Importances in Failure Classification (Random Forest Gini)")
    ax.set_xlabel("Relative Importance Score")
    for i, v in enumerate(rf_imp.values):
        ax.text(v + 0.005, i, f"{v*100:.1f}%", va="center", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "03_sensor_feature_importance.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p3}")

    return results_df

if __name__ == "__main__":
    out = r"C:\Users\abhin\.gemini\antigravity-ide\scratch\lab5_predictive_maintenance"
    run_predictive_maintenance_pipeline(out)
