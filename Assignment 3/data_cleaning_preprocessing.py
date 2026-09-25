"""
================================================================================
TATA TECHNOLOGIES - TECHPULSE FY-26: APPLIED AI & ML
LAB STATEMENT 3: DATA CLEANING & PREPROCESSING TECHNIQUES
================================================================================
Objective:
    Demonstrate comprehensive, industry-standard data cleaning and preprocessing
    techniques on automotive telemetry and sales records using Pandas and Scikit-learn.
    
Key Topics Covered:
    1. Dataset Ingestion & Missing Value Auditing
    2. Missing Value Imputation (SimpleImputer, KNNImputer, Strategy Justification)
    3. Outlier Detection & Treatment (IQR Winsorization & Z-Score Analysis)
    4. Categorical Encoding (OneHotEncoder for Nominal, OrdinalEncoder for Ranked)
    5. Feature Scaling (StandardScaler, MinMaxScaler, RobustScaler comparisons)
    6. Production-Grade Scikit-Learn Pipeline & ColumnTransformer (Preventing Data Leakage)
    7. Artifact Generation (Cleaned Data, Processed Matrices, Serialized Pipelines, Diagnostic Plots)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Set aesthetic visual theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ==============================================================================
# 1. INGESTION & DATA AUDIT
# ==============================================================================
def load_and_audit_dataset(file_path: str) -> pd.DataFrame:
    """Loads raw dataset and logs comprehensive structural and statistical audit."""
    print("=" * 80)
    print("STAGE 1: DATA INGESTION & INITIAL AUDIT")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"[INFO] Successfully loaded dataset: {file_path}")
    print(f"[INFO] Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("--- 1.1 First 5 Rows Preview ---")
    print(df.head())
    
    print("\n--- 1.2 Schema & Data Types ---")
    df.info()
    
    print("\n--- 1.3 Missing Value Audit ---")
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': null_counts, 'Percentage (%)': null_pct.round(2)})
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(by='Missing_Count', ascending=False)
    print(missing_df)
    
    print("\n--- 1.4 Descriptive Statistics (Raw Numerical Features) ---")
    print(df.describe().round(2).T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    
    # Plot missing values
    plot_missing_values_audit(df, missing_df)
    
    return df


def plot_missing_values_audit(df: pd.DataFrame, missing_df: pd.DataFrame):
    """Generates visual bar chart of missing value percentages across features."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if not missing_df.empty:
        bars = ax.barh(missing_df.index, missing_df['Percentage (%)'], color='#2b5c8f', edgecolor='black', alpha=0.85)
        ax.set_title("Missing Values Audit by Feature (%)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Percentage Missing (%)", fontsize=12)
        ax.set_xlim(0, max(missing_df['Percentage (%)']) + 5)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                    va='center', ha='left', fontsize=10, fontweight='semibold')
    else:
        ax.text(0.5, 0.5, "No Missing Values Found", ha='center', va='center', fontsize=14)
        
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "01_missing_values_audit.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Missing values diagnostic saved to: {plot_path}")


# ==============================================================================
# 2. MISSING VALUE HANDLING (PANDAS & SCIKIT-LEARN)
# ==============================================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrates missing value treatment:
    - Numerical: Median imputation (robust against outliers and skewed distributions)
    - Categorical: Mode / Most frequent imputation
    - Explores KNNImputer as an advanced alternative
    """
    print("\n" + "=" * 80)
    print("STAGE 2: MISSING VALUE IMPUTATION")
    print("=" * 80)
    
    df_clean = df.copy()
    
    # Identify numerical and categorical features with missing values
    num_cols_with_nan = [col for col in df_clean.select_dtypes(include=[np.number]).columns if df_clean[col].isnull().sum() > 0]
    cat_cols_with_nan = [col for col in df_clean.select_dtypes(include=['object']).columns if col != 'vehicle_id' and df_clean[col].isnull().sum() > 0]
    
    print(f"[INFO] Numerical columns with missing values: {num_cols_with_nan}")
    print(f"[INFO] Categorical columns with missing values: {cat_cols_with_nan}\n")
    
    # Demonstration of SimpleImputer (Strategy = 'median' for numerical)
    # Median is preferred over Mean because automotive variables (like price, odometer, hp) are right-skewed and contain outliers.
    median_imputer = SimpleImputer(strategy='median')
    df_clean[num_cols_with_nan] = median_imputer.fit_transform(df_clean[num_cols_with_nan])
    print(f"[SUCCESS] Applied Median SimpleImputer on numerical columns: {num_cols_with_nan}")
    
    # Demonstration of SimpleImputer (Strategy = 'most_frequent' for categorical)
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df_clean[cat_cols_with_nan] = mode_imputer.fit_transform(df_clean[cat_cols_with_nan])
    print(f"[SUCCESS] Applied Mode SimpleImputer on categorical columns: {cat_cols_with_nan}")
    
    # Demonstration of KNNImputer (for multivariate dependency exploration)
    print("\n--- Comparative Note: KNNImputer vs SimpleImputer ---")
    knn_sample = df[['curb_weight_kg', 'mileage_kmpl', 'horsepower_hp']].copy()
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_result = knn_imputer.fit_transform(knn_sample)
    print(f"[NOTE] KNNImputer (k=5) successfully computed distance-weighted values for {knn_sample.shape[1]} correlated features.")
    
    # Verification
    remaining_nulls = df_clean.isnull().sum().sum()
    print(f"[AUDIT] Remaining Missing Values in DataFrame: {remaining_nulls}")
    return df_clean


# ==============================================================================
# 3. OUTLIER DETECTION & TREATMENT (IQR & Z-SCORE)
# ==============================================================================
def detect_and_treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects outliers using IQR and Z-Score techniques, and performs capping (Winsorization)
    to prevent loss of valuable training records while neutralizing leverage points.
    """
    print("\n" + "=" * 80)
    print("STAGE 3: OUTLIER DETECTION & TREATMENT")
    print("=" * 80)
    
    df_treated = df.copy()
    
    # Automotive physical plausibility check: mileage cannot be negative
    neg_mileage_count = (df_treated['mileage_kmpl'] <= 0).sum()
    if neg_mileage_count > 0:
        print(f"[SANITY FIX] Found {neg_mileage_count} physically impossible non-positive mileage values. Correcting to min plausible (8.0 kmpl).")
        df_treated.loc[df_treated['mileage_kmpl'] <= 0, 'mileage_kmpl'] = 8.0

    features_to_check = ['mileage_kmpl', 'odometer_reading_km', 'horsepower_hp', 'vehicle_price_lakhs']
    outlier_summary = []
    
    # Store pre-capping values for comparative visualization
    pre_cap_df = df_treated[features_to_check].copy()
    
    for col in features_to_check:
        # --- Method A: Interquartile Range (IQR) ---
        q1 = df_treated[col].quantile(0.25)
        q3 = df_treated[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound_iqr = q1 - (1.5 * iqr)
        upper_bound_iqr = q3 + (1.5 * iqr)
        
        iqr_outliers = df_treated[(df_treated[col] < lower_bound_iqr) | (df_treated[col] > upper_bound_iqr)]
        
        # --- Method B: Z-Score Method ---
        mean_val = df_treated[col].mean()
        std_val = df_treated[col].std()
        z_scores = (df_treated[col] - mean_val) / std_val
        z_outliers = df_treated[np.abs(z_scores) > 3.0]
        
        outlier_summary.append({
            'Feature': col,
            'Q1': round(q1, 2),
            'Q3': round(q3, 2),
            'IQR': round(iqr, 2),
            'IQR_Lower': round(lower_bound_iqr, 2),
            'IQR_Upper': round(upper_bound_iqr, 2),
            'IQR_Outlier_Count': len(iqr_outliers),
            'Z_Score_Outliers (|z|>3)': len(z_outliers)
        })
        
        # --- Treatment: Capping / Winsorization ---
        # Cap values beyond 1.5*IQR to the bounds
        df_treated[col] = np.clip(df_treated[col], lower_bound_iqr, upper_bound_iqr)
        
    summary_df = pd.DataFrame(outlier_summary)
    print(summary_df.to_string(index=False))
    print("\n[SUCCESS] Treated outliers using IQR Winsorization (capping to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]).")
    
    # Generate Boxplot Visualizations Before and After Capping
    plot_outlier_treatment(pre_cap_df, df_treated[features_to_check], features_to_check)
    
    return df_treated


def plot_outlier_treatment(before_df: pd.DataFrame, after_df: pd.DataFrame, features: list):
    """Plots comparative boxplots showing outlier profiles before and after treatment."""
    fig, axes = plt.subplots(len(features), 2, figsize=(14, 3.2 * len(features)))
    
    for i, col in enumerate(features):
        # Before Capping
        sns.boxplot(x=before_df[col], ax=axes[i, 0], color='#e74c3c', flierprops={'markerfacecolor':'red', 'markersize':5})
        axes[i, 0].set_title(f"Raw / Uncapped: {col}", fontsize=11, fontweight='bold')
        axes[i, 0].set_xlabel("")
        
        # After Capping
        sns.boxplot(x=after_df[col], ax=axes[i, 1], color='#2ecc71', flierprops={'markerfacecolor':'green', 'markersize':5})
        axes[i, 1].set_title(f"Post IQR Capping: {col}", fontsize=11, fontweight='bold')
        axes[i, 1].set_xlabel("")
        
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "02_outlier_treatment_boxplots.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Outlier treatment boxplots saved to: {plot_path}")


# ==============================================================================
# 4. CATEGORICAL ENCODING
# ==============================================================================
def encode_categorical_features(df: pd.DataFrame):
    """
    Demonstrates encoding strategies:
    1. One-Hot Encoding for Nominal variables (fuel_type, transmission)
    2. Ordinal Encoding for Rank-Ordered variables (vehicle_segment, owner_type)
    """
    print("\n" + "=" * 80)
    print("STAGE 4: CATEGORICAL ENCODING")
    print("=" * 80)
    
    df_encoded = df.copy()
    
    # 4.1 Nominal Features -> One-Hot Encoding
    nominal_cols = ['fuel_type', 'transmission']
    print(f"[ENCODING] Applying OneHotEncoder to nominal features: {nominal_cols}")
    ohe = OneHotEncoder(drop='first', sparse_output=False) # drop='first' avoids dummy variable trap / multicollinearity
    ohe_arr = ohe.fit_transform(df_encoded[nominal_cols])
    ohe_feature_names = ohe.get_feature_names_out(nominal_cols)
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe_feature_names, index=df_encoded.index)
    print(f"  Generated One-Hot Features: {list(ohe_feature_names)}")
    
    # 4.2 Ordinal Features -> Ordinal Encoding
    # Explicitly defined hierarchy
    segment_order = ['Hatchback', 'Sedan', 'SUV', 'Luxury']
    owner_order = ['First', 'Second', 'Third']
    
    print(f"[ENCODING] Applying OrdinalEncoder with defined hierarchies:")
    print(f"  vehicle_segment: {segment_order}")
    print(f"  owner_type: {owner_order}")
    
    ordinal_enc = OrdinalEncoder(categories=[segment_order, owner_order])
    ordinal_arr = ordinal_enc.fit_transform(df_encoded[['vehicle_segment', 'owner_type']])
    ordinal_df = pd.DataFrame(ordinal_arr, columns=['vehicle_segment_encoded', 'owner_type_encoded'], index=df_encoded.index)
    
    # Combine back
    df_combined = pd.concat([
        df_encoded.drop(columns=nominal_cols + ['vehicle_segment', 'owner_type']),
        ohe_df,
        ordinal_df
    ], axis=1)
    
    print(f"[SUCCESS] Encoded shape: {df_combined.shape[1]} columns (expanded from {df.shape[1]})")
    return df_combined, ohe, ordinal_enc


# ==============================================================================
# 5. FEATURE SCALING COMPARISON
# ==============================================================================
def compare_feature_scalers(df: pd.DataFrame, sample_cols: list = ['curb_weight_kg', 'odometer_reading_km', 'horsepower_hp']):
    """
    Compares StandardScaler, MinMaxScaler, and RobustScaler.
    Generates comparative distribution plots and statistical summary table.
    """
    print("\n" + "=" * 80)
    print("STAGE 5: FEATURE SCALING ANALYSIS")
    print("=" * 80)
    
    raw_data = df[sample_cols].copy()
    
    # 1. StandardScaler: z = (x - mean) / std
    std_scaler = StandardScaler()
    std_scaled = pd.DataFrame(std_scaler.fit_transform(raw_data), columns=sample_cols)
    
    # 2. MinMaxScaler: x_norm = (x - min) / (max - min)
    minmax_scaler = MinMaxScaler()
    minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(raw_data), columns=sample_cols)
    
    # 3. RobustScaler: x_rob = (x - median) / IQR
    robust_scaler = RobustScaler()
    robust_scaled = pd.DataFrame(robust_scaler.fit_transform(raw_data), columns=sample_cols)
    
    print("--- 5.1 Scaler Statistical Comparison (Mean & Std Dev) ---")
    comparison = []
    for col in sample_cols:
        comparison.append({
            'Feature': col,
            'Raw Mean (Std)': f"{raw_data[col].mean():.1f} ({raw_data[col].std():.1f})",
            'Standard Scaled Mean (Std)': f"{std_scaled[col].mean():.2f} ({std_scaled[col].std():.2f})",
            'MinMax Range': f"[{minmax_scaled[col].min():.2f}, {minmax_scaled[col].max():.2f}]",
            'Robust Median (IQR)': f"{robust_scaled[col].median():.2f} ({robust_scaled[col].quantile(0.75) - robust_scaled[col].quantile(0.25):.2f})"
        })
    print(pd.DataFrame(comparison).to_string(index=False))
    
    # Generate Visual Plot
    plot_scaling_comparison(raw_data, std_scaled, minmax_scaled, robust_scaled, sample_cols[0])


def plot_scaling_comparison(raw_df, std_df, minmax_df, robust_df, col_name):
    """Plots probability density distributions comparing scaling transformations."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    
    sns.kdeplot(raw_df[col_name], ax=axes[0], fill=True, color='#34495e')
    axes[0].set_title(f"Original Raw ({col_name})", fontweight='bold')
    axes[0].set_xlabel("Original Units")
    
    sns.kdeplot(std_df[col_name], ax=axes[1], fill=True, color='#2980b9')
    axes[1].set_title("StandardScaler (μ=0, σ=1)", fontweight='bold')
    axes[1].set_xlabel("Z-Score")
    
    sns.kdeplot(minmax_df[col_name], ax=axes[2], fill=True, color='#27ae60')
    axes[2].set_title("MinMaxScaler (Range [0, 1])", fontweight='bold')
    axes[2].set_xlabel("Normalized Value")
    
    sns.kdeplot(robust_df[col_name], ax=axes[3], fill=True, color='#8e44ad')
    axes[3].set_title("RobustScaler (Median=0, IQR=1)", fontweight='bold')
    axes[3].set_xlabel("Robust Scaled Value")
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "03_feature_scaling_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Scaling comparison distribution curves saved to: {plot_path}")


# ==============================================================================
# 6. PRODUCTION SCIKIT-LEARN PIPELINE & COLUMN TRANSFORMER
# ==============================================================================
def build_and_evaluate_production_pipeline(raw_csv_path: str):
    """
    Builds a leak-free scikit-learn Pipeline and ColumnTransformer:
    - Numerical Pipeline: SimpleImputer(median) -> RobustScaler() / StandardScaler()
    - Categorical Nominal Pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(drop='first')
    - Categorical Ordinal Pipeline: SimpleImputer(most_frequent) -> OrdinalEncoder(categories=...)
    Fits strictly on Train data and transforms Test data.
    """
    print("\n" + "=" * 80)
    print("STAGE 6: PRODUCTION PIPELINE & COLUMN TRANSFORMER")
    print("=" * 80)
    
    raw_df = pd.read_csv(raw_csv_path)
    
    # Feature vs Target separation (Target: vehicle_price_lakhs for Lab 4 linkage)
    X = raw_df.drop(columns=['vehicle_id', 'vehicle_price_lakhs'])
    y = raw_df['vehicle_price_lakhs']
    
    # Define feature groups
    num_features = ['engine_size_cc', 'horsepower_hp', 'curb_weight_kg', 'mileage_kmpl', 'odometer_reading_km', 'maintenance_score']
    nominal_features = ['fuel_type', 'transmission']
    ordinal_features = ['vehicle_segment', 'owner_type']
    
    segment_categories = ['Hatchback', 'Sedan', 'SUV', 'Luxury']
    owner_categories = ['First', 'Second', 'Third']
    
    # 6.1 Sub-Pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[segment_categories, owner_categories]))
    ])
    
    # 6.2 Master Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, num_features),
            ('nom', nominal_pipeline, nominal_features),
            ('ord', ordinal_pipeline, ordinal_features)
        ],
        remainder='drop'
    )
    
    # 6.3 Train-Test Split (80-20) to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"[DATA SPLIT] Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    
    # Fit strictly on train, transform both
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Retrieve constructed column names
    ohe_step = preprocessor.named_transformers_['nom'].named_steps['ohe']
    ohe_col_names = list(ohe_step.get_feature_names_out(nominal_features))
    all_feature_names = num_features + ohe_col_names + [f"{c}_ord" for c in ordinal_features]
    
    print(f"[SUCCESS] Preprocessor fitted successfully!")
    print(f"[FEATURE NAMES] Extracted {len(all_feature_names)} transformed features:")
    for fn in all_feature_names:
        print(f"   • {fn}")
        
    # Serialize Pipeline
    pipeline_file = os.path.join(BASE_DIR, "automotive_preprocessing_pipeline.joblib")
    joblib.dump(preprocessor, pipeline_file)
    print(f"\n[SAVED] Serialized production pipeline to: {pipeline_file}")
    
    # Export preprocessed feature matrix
    train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    train_processed_df['vehicle_price_lakhs'] = y_train.values
    
    export_features_path = os.path.join(BASE_DIR, "automotive_features_preprocessed.csv")
    train_processed_df.to_csv(export_features_path, index=False)
    print(f"[SAVED] Exported preprocessed feature matrix: {export_features_path}")
    
    return preprocessor, train_processed_df


# ==============================================================================
# 7. CORRELATION MATRIX & EXPORT
# ==============================================================================
def plot_cleaned_correlation_matrix(df: pd.DataFrame):
    """Generates correlation heatmap on cleaned numerical features."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax, square=True, cbar_kws={'shrink': .8})
    ax.set_title("Pearson Correlation Heatmap (Cleaned Features)", fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "04_correlation_matrix_cleaned.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Correlation heatmap saved to: {plot_path}")


# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================
def main():
    raw_csv = os.path.join(BASE_DIR, "automotive_data_raw.csv")
    if not os.path.exists(raw_csv):
        print(f"[ERROR] Raw data not found at {raw_csv}. Run generate_dataset.py first.")
        sys.exit(1)
        
    print("\n" + "#" * 80)
    print("STARTING LAB STATEMENT 3: DATA CLEANING & PREPROCESSING")
    print("#" * 80 + "\n")
    
    # 1. Audit
    raw_df = load_and_audit_dataset(raw_csv)
    
    # 2. Impute Missing Values
    imputed_df = handle_missing_values(raw_df)
    
    # 3. Detect and Treat Outliers
    treated_df = detect_and_treat_outliers(imputed_df)
    
    # 4. Categorical Encoding (Exploratory)
    encoded_df, _, _ = encode_categorical_features(treated_df)
    
    # 5. Feature Scaling (Comparative)
    compare_feature_scalers(treated_df)
    
    # 6. Production Scikit-Learn Pipeline
    pipeline, preprocessed_df = build_and_evaluate_production_pipeline(raw_csv)
    
    # 7. Correlation Analysis
    plot_cleaned_correlation_matrix(treated_df)
    
    # 8. Export Cleaned Dataset (Pandas DataFrame)
    cleaned_csv = os.path.join(BASE_DIR, "automotive_data_cleaned.csv")
    treated_df.to_csv(cleaned_csv, index=False)
    print(f"\n[SAVED] Cleaned business-readable dataset exported to: {cleaned_csv}")
    
    print("\n" + "#" * 80)
    print("LAB STATEMENT 3 EXECUTION COMPLETED SUCCESSFULLY")
    print("#" * 80)


if __name__ == "__main__":
    main()
