# Tata Technologies - TechPulse FY-26: Applied AI & ML
## Lab Statement 10: MLOps Workflow Simulation

**Course:** Applied AI & ML (TechPulse FY-26)  
**Track:** AI & ML | **Level:** Intermediate  
**Curriculum Unit:** Unit 5 – Generative AI, Prompt Engineering & MLOps Overview (Overview of MLOps, CI/CD in ML pipelines, Model versioning, MLflow introduction, Containerization using Docker)  
**Lab Statement 10:** *Build a CI/CD pipeline using MLflow and Docker for model deployment.*  
**Domain Focus:** Automotive Telemetry Serving, Automated Quality Gates & Production Containerization  

---

## 📌 1. Overview & Objectives

In production automotive systems (such as connected car fleet analytics and ECU calibration backends), Machine Learning models cannot remain confined to static notebooks. MLOps establishes automated practices for continuous experiment tracking, version control, staging quality gates, container packaging, and API deployment.

This lab implements an end-to-end MLOps workflow simulating **MLflow tracking**, **Automated Model Registry Promotion**, and **Docker containerization**.

### Key Learning Outcomes:
1. **Experiment Tracking with MLflow**: Record hyperparameter sweeps (estimators, depth, regularizers) and track metrics (MAE, RMSE, $R^2$, latency).
2. **Quality Gate Validation**: Implement automated gate criteria ($R^2 > 0.95$ and $\text{Latency} < 10\text{ms}$) to promote candidate models to Production.
3. **Containerization with Docker**: Package runtime environments and FastAPI microservices into a standalone `Dockerfile` and `docker-compose.yml`.
4. **CI/CD Lifecycle Architecture**: Design continuous integration pipelines testing data validation, unit testing, and automated deployment.

---

## 📁 2. Project Directory Structure

```text
lab10_mlops_workflow_simulation/
│
├── README.md                                      # Lab manual & technical documentation
├── telemetry_training_data.csv                    # Engine telemetry dataset (1,000 records)
├── mlops_pipeline_simulation.py                   # Complete MLOps tracking & deployment script
├── lab10_mlops_workflow_simulation.ipynb          # Interactive Jupyter Notebook
├── requirements.txt                               # Minimal dependencies
├── Dockerfile                                     # Production container build specification
├── docker-compose.yml                             # Container orchestration configuration
│
└── plots/                                         # Diagnostic visualizations (200 DPI)
    ├── 01_mlflow_experiment_run_comparison.png    # MLflow run metric comparison bar charts
    ├── 02_mlops_cicd_pipeline_architecture.png    # End-to-end CI/CD lifecycle diagram
    └── 03_model_latency_benchmark.png             # Microservice latency distribution
```

---

## 📊 3. MLflow Experiment Leaderboard

| Run ID | Model Name | Test $R^2$ | Test MAE | Test RMSE | Latency (ms) | Registry Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Run_001`** | **Linear Regression** | **0.9922** | **0.270** | **0.348** | **0.91 ms** | **PROMOTED TO PRODUCTION (v1.0)** |
| `Run_002` | Ridge Regularized | 0.9921 | 0.270 | 0.350 | 0.88 ms | Candidate (Staging) |
| `Run_005` | Gradient Boosting | 0.9854 | 0.338 | 0.476 | 1.03 ms | Candidate (Staging) |
| `Run_004` | Random Forest (Deep) | 0.9820 | 0.368 | 0.529 | 8.76 ms | Evaluated |
| `Run_003` | Random Forest (Shallow) | 0.9706 | 0.482 | 0.676 | 4.23 ms | Evaluated |

---

## 🚀 4. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Full MLOps Pipeline & Docker Simulation
```bash
python mlops_pipeline_simulation.py
```

### Step 3: Open Interactive Jupyter Notebook
```bash
jupyter notebook lab10_mlops_workflow_simulation.ipynb
```
