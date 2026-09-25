[README.md](https://github.com/user-attachments/files/32665370/README.md)
[README.md](https://github.com/user-attachments/files/32664425/README.md)
# Tata Technologies Ltd. - TechPulse FY-26
# Applied AI & ML — Assignments Repository

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](http# Tata Technologies - Applied AI & ML Assignments

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)

This repository contains the complete solutions for all **10 Assignments** covering foundational Machine Learning, Deep Learning, Computer Vision, Genetic Algorithms, Natural Language Processing, and MLOps.

---

## 📑 Assignment Index

| Assignment | Topic / Title | Key Methodology & Models | Primary Tools | Directory Link |
| :---: | :--- | :--- | :--- | :--- |
| **Assignment 1** | **ML Model for Car Mileage Estimation** | Regression (SLR, MLR, Poly, Ridge, Lasso, Random Forest), Residual Diagnostics | Scikit-Learn, Seaborn | [🔗 Assignment 1](./Assignment%201/) |
| **Assignment 2** | **Simulated Driving Agent Behavior** | 2D Track Kinematics, 5-Raycast LIDAR, Neuroevolution, Genetic Algorithm Engine | NumPy, Matplotlib | [🔗 Assignment 2](./Assignment%202/) |
| **Assignment 3** | **Data Cleaning & Preprocessing** | Median/KNN Imputation, IQR/Z-Score Winsorization, StandardScaler/MinMax, Pipeline | Pandas, Scikit-Learn | [🔗 Assignment 3](./Assignment%203/) |
| **Assignment 4** | **Vehicle Price Prediction** | Multi-Factor Depreciation Analytics, Ensembles, Resale Valuation ($R^2 = 0.985$) | Scikit-Learn, Pandas | [🔗 Assignment 4](./Assignment%204/) |
| **Assignment 5** | **Predictive Maintenance from Sensor Logs** | OBD ECU Sensor Telemetry, HDF/PWF/OSF/TWF Failures, Balanced ROC-AUC ($0.958$) | Scikit-Learn, SciPy | [🔗 Assignment 5](./Assignment%205/) |
| **Assignment 6** | **Traffic Sign Classification using CNN** | PyTorch Deep CNN (Conv2D, BatchNorm, MaxPool, Dropout), 10 GTSRB Classes (100% Acc) | PyTorch, TorchVision | [🔗 Assignment 6](./Assignment%206/) |
| **Assignment 7** | **Pedestrian Detection using OpenCV** | Histogram of Oriented Gradients (HOG) + Dalal-Triggs Linear SVM, Multi-scale NMS | OpenCV, NumPy | [🔗 Assignment 7](./Assignment%207/) |
| **Assignment 8** | **Sentiment Analysis using LSTM** | PyTorch Bidirectional LSTM (Bi-LSTM), Word Embeddings, VoC Customer Sentiment | PyTorch, NLP | [🔗 Assignment 8](./Assignment%208/) |
| **Assignment 9** | **Feature Importance Visualization** | Explainable AI (XAI): MDI (Gini), Holdout Permutation Importance, Partial Dependence (PDP) | Scikit-Learn, XAI | [🔗 Assignment 9](./Assignment%209/) |
| **Assignment 10** | **MLOps Workflow Simulation** | MLflow Experiment Tracking, Model Registry Quality Gates, Dockerfile & docker-compose | MLflow, Docker | [🔗 Assignment 10](./Assignment%2010/) |

---

## 📁 Repository Directory Structure

```text
.
├── .gitignore                                     # Master Git ignore rules
├── requirements.txt                               # Unified Python dependencies
├── README.md                                      # Repository documentation
│
├── Assignment 1/                                  # ML Model for Car Mileage Estimation
│   ├── README.md                                  # Lab manual with derivations & benchmarks
│   ├── lab1_car_mileage_estimation.ipynb          # Interactive Jupyter Notebook
│   ├── mileage_prediction.py                      # Standalone execution pipeline
│   ├── auto_mpg.csv                               # Canonical Seaborn MPG dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Diagnostic visualizations (6 figures)
│
├── Assignment 2/                                  # Simulated Driving Agent Behavior
│   ├── README.md                                  # Genetic Algorithm formulations & kinematics
│   ├── lab2_simulated_driving_agent.ipynb         # Interactive Jupyter Notebook
│   ├── driving_agent_simulation.py                # GA simulation & vehicle control engine
│   ├── driving_track_coords.csv                   # 2D track boundary coordinates dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Generational trajectory & fitness plots
│
├── Assignment 3/                                  # Data Cleaning & Preprocessing Techniques
│   ├── README.md                                  # Missing value, outlier & scaling theory
│   ├── lab3_data_cleaning_preprocessing.ipynb     # Interactive Jupyter Notebook
│   ├── data_cleaning_preprocessing.py             # Modular cleaning pipeline
│   ├── automotive_data_raw.csv                    # Raw dataset with missing values & outliers
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Imputation & Winsorization boxplots
│
├── Assignment 4/                                  # Vehicle Price Prediction
│   ├── README.md                                  # Depreciation economics & benchmark metrics
│   ├── lab4_vehicle_price_prediction.ipynb        # Interactive Jupyter Notebook
│   ├── vehicle_price_prediction.py                # Machine learning regression script
│   ├── car_price_dataset.csv                      # Structured vehicle resale records
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Depreciation curves & model comparisons
│
├── Assignment 5/                                  # Predictive Maintenance from Sensor Logs
│   ├── README.md                                  # Failure modes (HDF, PWF, OSF, TWF) & metrics
│   ├── lab5_predictive_maintenance.ipynb          # Interactive Jupyter Notebook
│   ├── predictive_maintenance.py                  # Classification pipeline & evaluation
│   ├── sensor_maintenance_data.csv                # Multi-channel sensory telemetry dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Confusion matrices & ROC-AUC curves
│
├── Assignment 6/                                  # Traffic Sign Classification using CNN
│   ├── README.md                                  # Deep CNN architecture & receptive fields
│   ├── lab6_traffic_sign_classification.ipynb     # Interactive Jupyter Notebook
│   ├── traffic_sign_cnn.py                        # PyTorch training & inference script
│   ├── traffic_signs_metadata.csv                 # 10-class traffic sign metadata dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Sample grids, loss curves & confusion matrix
│
├── Assignment 7/                                  # Pedestrian Detection using OpenCV
│   ├── README.md                                  # HOG cell/block normalization & Linear SVM
│   ├── lab7_pedestrian_detection_opencv.ipynb     # Interactive Jupyter Notebook
│   ├── pedestrian_detection_hog_svm.py            # OpenCV sliding window & NMS pipeline
│   ├── pedestrian_annotations.csv                 # Ground truth bounding boxes dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Gradient orientations & NMS comparisons
│
├── Assignment 8/                                  # Sentiment Analysis using LSTM
│   ├── README.md                                  # Bi-LSTM architecture & embeddings
│   ├── lab8_sentiment_analysis_lstm.ipynb         # Interactive Jupyter Notebook
│   ├── sentiment_analysis_lstm.py                 # PyTorch recurrent model training
│   ├── vehicle_reviews_sentiment.csv              # Customer voice-of-customer reviews dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Sentiment distributions & confusion matrix
│
├── Assignment 9/                                  # Feature Importance Visualization
│   ├── README.md                                  # MDI, Permutation Importance & PDP curves
│   ├── lab9_feature_importance_visualization.ipynb # Interactive Jupyter Notebook
│   ├── feature_importance_analysis.py             # Feature importance analysis script
│   ├── automotive_features_dataset.csv            # Powertrain engineering dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # PDP curves & cross-method comparisons
│
└── Assignment 10/                                 # MLOps Workflow Simulation
    ├── README.md                                  # Experiment tracking, registry & Docker
    ├── lab10_mlops_workflow_simulation.ipynb      # Interactive Jupyter Notebook
    ├── mlops_pipeline_simulation.py               # MLflow tracking & CI/CD pipeline
    ├── telemetry_training_data.csv                # Engine telemetry dataset
    ├── requirements.txt                           # Dependencies
    └── plots/                                     # MLflow leaderboards & architecture diagrams
```
s://opensource.org/licenses/MIT)

> **Course:** Applied AI ML (TechPulse FY-26)  
> **Track:** AI & ML | **Level:** Intermediate  
> **Corporate HQ & Registered Office:** Plot No 25, Rajiv Gandhi Infotech Park, Hinjawadi, Pune, India – 411 057  

This repository contains the complete, production-grade solutions for all **10 Assignments** prescribed in the Tata Technologies Applied AI & ML curriculum. Every assignment is structured with a minimal, clean layout containing an interactive Jupyter Notebook, standalone Python script, automotive dataset, diagnostic plots, README manual, and requirements.

---

## 📑 Assignment Index & Curriculum Mapping

| Assignment | Topic & Lab Statement Title | Methodology & Key Models | Primary Tools | Directory Link |
| :---: | :--- | :--- | :--- | :--- |
| **Assignment 1** | **ML Model for Car Mileage Estimation** | Regression (SLR, MLR, Poly, Ridge, Lasso, Random Forest), Residual Diagnostics | Scikit-Learn, Seaborn | [🔗 Assignment 1](./Assignment%201/) |
| **Assignment 2** | **Simulated Driving Agent Behavior** | 2D Track Kinematics, 5-Raycast LIDAR, Neuroevolution, Genetic Algorithm Engine | NumPy, Matplotlib | [🔗 Assignment 2](./Assignment%202/) |
| **Assignment 3** | **Data Cleaning & Preprocessing** | Median/KNN Imputation, IQR/Z-Score Winsorization, StandardScaler/MinMax, Pipeline | Pandas, Scikit-Learn | [🔗 Assignment 3](./Assignment%203/) |
| **Assignment 4** | **Vehicle Price Prediction** | Multi-Factor Depreciation Analytics, Ensembles, Resale Valuation ($R^2 = 0.985$) | Scikit-Learn, Pandas | [🔗 Assignment 4](./Assignment%204/) |
| **Assignment 5** | **Predictive Maintenance from Sensor Logs** | OBD ECU Sensor Telemetry, HDF/PWF/OSF/TWF Failures, Balanced ROC-AUC ($0.958$) | Scikit-Learn, SciPy | [🔗 Assignment 5](./Assignment%205/) |
| **Assignment 6** | **Traffic Sign Classification using CNN** | PyTorch Deep CNN (Conv2D, BatchNorm, MaxPool, Dropout), 10 GTSRB Classes (100% Acc) | PyTorch, TorchVision | [🔗 Assignment 6](./Assignment%206/) |
| **Assignment 7** | **Pedestrian Detection using OpenCV** | Histogram of Oriented Gradients (HOG) + Dalal-Triggs Linear SVM, Multi-scale NMS | OpenCV, NumPy | [🔗 Assignment 7](./Assignment%207/) |
| **Assignment 8** | **Sentiment Analysis using LSTM** | PyTorch Bidirectional LSTM (Bi-LSTM), Word Embeddings, VoC Customer Sentiment | PyTorch, NLP | [🔗 Assignment 8](./Assignment%208/) |
| **Assignment 9** | **Feature Importance Visualization** | Explainable AI (XAI): MDI (Gini), Holdout Permutation Importance, Partial Dependence (PDP) | Scikit-Learn, XAI | [🔗 Assignment 9](./Assignment%209/) |
| **Assignment 10** | **MLOps Workflow Simulation** | MLflow Experiment Tracking, Model Registry Quality Gates, Dockerfile & docker-compose | MLflow, Docker | [🔗 Assignment 10](./Assignment%2010/) |

---

## 📁 Repository Directory Structure

```text
.
├── .gitignore                                     # Master Git ignore rules
├── requirements.txt                               # Unified Python dependencies
├── README.md                                      # Master repository documentation (this file)
│
├── Assignment 1/                                  # ML Model for Car Mileage Estimation
│   ├── README.md                                  # Lab manual with derivations & benchmarks
│   ├── lab1_car_mileage_estimation.ipynb          # Interactive Jupyter Notebook
│   ├── mileage_prediction.py                      # Standalone execution pipeline
│   ├── auto_mpg.csv                               # Canonical Seaborn MPG dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Diagnostic visualizations (6 figures)
│
├── Assignment 2/                                  # Simulated Driving Agent Behavior
│   ├── README.md                                  # Genetic Algorithm formulations & kinematics
│   ├── lab2_simulated_driving_agent.ipynb         # Interactive Jupyter Notebook
│   ├── driving_agent_simulation.py                # GA simulation & vehicle control engine
│   ├── driving_track_coords.csv                   # 2D track boundary coordinates dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Generational trajectory & fitness plots
│
├── Assignment 3/                                  # Data Cleaning & Preprocessing Techniques
│   ├── README.md                                  # Missing value, outlier & scaling theory
│   ├── lab3_data_cleaning_preprocessing.ipynb     # Interactive Jupyter Notebook
│   ├── data_cleaning_preprocessing.py             # Modular cleaning pipeline
│   ├── automotive_data_raw.csv                    # Raw dataset with missing values & outliers
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Imputation & Winsorization boxplots
│
├── Assignment 4/                                  # Vehicle Price Prediction
│   ├── README.md                                  # Depreciation economics & benchmark metrics
│   ├── lab4_vehicle_price_prediction.ipynb        # Interactive Jupyter Notebook
│   ├── vehicle_price_prediction.py                # Machine learning regression script
│   ├── car_price_dataset.csv                      # Structured vehicle resale records
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Depreciation curves & model comparisons
│
├── Assignment 5/                                  # Predictive Maintenance from Sensor Logs
│   ├── README.md                                  # Failure modes (HDF, PWF, OSF, TWF) & metrics
│   ├── lab5_predictive_maintenance.ipynb          # Interactive Jupyter Notebook
│   ├── predictive_maintenance.py                  # Classification pipeline & evaluation
│   ├── sensor_maintenance_data.csv                # Multi-channel sensory telemetry dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Confusion matrices & ROC-AUC curves
│
├── Assignment 6/                                  # Traffic Sign Classification using CNN
│   ├── README.md                                  # Deep CNN architecture & receptive fields
│   ├── lab6_traffic_sign_classification.ipynb     # Interactive Jupyter Notebook
│   ├── traffic_sign_cnn.py                        # PyTorch training & inference script
│   ├── traffic_signs_metadata.csv                 # 10-class traffic sign metadata dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Sample grids, loss curves & confusion matrix
│
├── Assignment 7/                                  # Pedestrian Detection using OpenCV
│   ├── README.md                                  # HOG cell/block normalization & Linear SVM
│   ├── lab7_pedestrian_detection_opencv.ipynb     # Interactive Jupyter Notebook
│   ├── pedestrian_detection_hog_svm.py            # OpenCV sliding window & NMS pipeline
│   ├── pedestrian_annotations.csv                 # Ground truth bounding boxes dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Gradient orientations & NMS comparisons
│
├── Assignment 8/                                  # Sentiment Analysis using LSTM
│   ├── README.md                                  # Bi-LSTM architecture & embeddings
│   ├── lab8_sentiment_analysis_lstm.ipynb         # Interactive Jupyter Notebook
│   ├── sentiment_analysis_lstm.py                 # PyTorch recurrent model training
│   ├── vehicle_reviews_sentiment.csv              # Customer voice-of-customer reviews dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # Sentiment distributions & confusion matrix
│
├── Assignment 9/                                  # Feature Importance Visualization
│   ├── README.md                                  # MDI, Permutation Importance & PDP curves
│   ├── lab9_feature_importance_visualization.ipynb # Interactive Jupyter Notebook
│   ├── feature_importance_analysis.py             # Feature importance analysis script
│   ├── automotive_features_dataset.csv            # Powertrain engineering dataset
│   ├── requirements.txt                           # Dependencies
│   └── plots/                                     # PDP curves & cross-method comparisons
│
└── Assignment 10/                                 # MLOps Workflow Simulation
    ├── README.md                                  # Experiment tracking, registry & Docker
    ├── lab10_mlops_workflow_simulation.ipynb      # Interactive Jupyter Notebook
    ├── mlops_pipeline_simulation.py               # MLflow tracking & CI/CD pipeline
    ├── telemetry_training_data.csv                # Engine telemetry dataset
    ├── requirements.txt                           # Dependencies
    └── plots/                                     # MLflow leaderboards & architecture diagrams
```

---

## 🚀 Getting Started

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Running Jupyter Notebooks
Start the notebook server:
```bash
jupyter notebook
```
Navigate to any assignment folder (e.g., `Assignment 1/`) and open the `.ipynb` file.

### 3. Running Standalone Scripts
```bash
# Example: Assignment 1 (Car Mileage Estimation)
python "Assignment 1/mileage_prediction.py"

# Example: Assignment 6 (Traffic Sign Classification CNN)
python "Assignment 6/traffic_sign_cnn.py"

# Example: Assignment 7 (Pedestrian Detection OpenCV)
python "Assignment 7/pedestrian_detection_hog_svm.py"
```

---

## 📚 Recommended Literature & References
- *Introduction to AI & Machine Learning* — Munesh Chandra Trivedi & Ankit Srivastava
- *Python Machine Learning* — Sebastian Raschka & Vahid Mirjalili
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — Aurélien Géron
- *Deep Learning* — Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- *Histograms of Oriented Gradients for Human Detection* — Navneet Dalal and Bill Triggs (CVPR 2005)

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
