# NO₂ Prediction with Urban Amenities and Satellite Data

## 1. Project Overview
Building on our Homework 2 ML pipeline for forecasting monthly NO₂ concentrations across Philippine provinces, this version adds:

- `Docker containerization` for reproducible, immutable environments.
- `Apache Airflow orchestration` (official `apache/airflow:3.0.3` image) for automation.
- `MLflow tracking + registry` for experiment management and model versioning.
- `Drift Simulation and Detection (Evidently)` with branching retrain logic.

---

## 2. Repository Setup

Clone the repository and switch to the HW3 branch:

```bash
git clone git@github.com:erosestante/ac112e6b40ed278155f9ba01f3d16c384cac5674e1cea62f7c67f4cda328a01b_no2_prediction
cd ac112e6b40ed278155f9ba01f3d16c384cac5674e1cea62f7c67f4cda328a01b_no2_prediction
git checkout hw3-mlflow-drift
```

---

## 3. Execution Modes

This project can be run in two modes:

1. **Standalone pipeline** (local virtual environment + `python src/run_pipeline.py`)
2. **Airflow DAG pipeline** (containerized orchestration with `airflow dags test` or UI trigger)

---

### A) Standalone Pipeline

#### 1. Setup Instructions
```bash
# Activate local environment (created with uv/venv)
source .venv/bin/activate

# Run end-to-end pipeline
python src/run_pipeline.py
```

#### 2. MLflow Integration
- Tracking URI defaults to `http://localhost:5001` (port 5000 is occupied by AirPlay on macOS).
- Each run logs **parameters, metrics, SHAP plots, and models**.
- If thresholds are met, the model is **registered** in MLflow (`no2_xgb_cli`).
- Access UI: [http://localhost:5001](http://localhost:5001).

#### 3. Drift Detection
`run_pipeline.py` generates `train.csv`, `test.csv`, and `drifted_test.csv`.
- **Simulation:** numeric features shifted (+0.5σ), Gaussian noise (+0.3σ), variance amplification.
- **Detection:** Evidently compares distributions, logs results, and raises an error if drift is detected.
- **Outcome:** MLflow run tagged as `drift_detected_error` if flagged.

#### 4. Testing Instructions
```bash
# Run the pipeline
python src/run_pipeline.py

# Inspect outputs
ls reports/
cat reports/xgb_metrics.json

# Verify MLflow UI
open http://localhost:5001
```

---

### B) Airflow DAG Pipeline

#### 1. Setup Instructions
```bash
# Build and start the stack (env file already provided)
docker compose build
docker compose up -d

# Check services
docker compose ps
curl -I http://localhost:8080/health
```

- Airflow UI: [http://localhost:8080](http://localhost:8080) (login: airflow / airflow)
- MLflow UI: [http://localhost:5001](http://localhost:5001)

#### 2. DAG Tasks
1. **preprocess_data** – clean raw NO₂ and create drift test files
2. **feature_engineering** – add lag features and flags
3. **train_model** – baseline + XGBoost training
4. **evaluate_model** – summarize metrics
5. **drift_detection** – Evidently drift check
6. **branch_on_drift** – decide between `pipeline_complete` or `retrain_model`

#### 3. Drift Detection
- Same perturbation method as standalone.
- Reports saved in `/opt/airflow/data/reports`.
- If drift detected → DAG branches to `retrain_model`.

#### 4. Testing Instructions
```bash
# Test run without scheduler
docker compose exec airflow-webserver   airflow dags test ml_pipeline_dag 2025-08-02

# Trigger full run
docker compose exec airflow-webserver   airflow dags trigger ml_pipeline_dag
```

---

## 4. Folder Structure

```text
.
├── data/
│   ├── raw/                # Immutable NO₂ data
│   ├── processed/          # Preprocessed & engineered features
│   └── reports/            # Evaluation & drift reports
├── deploy/
│   ├── airflow/            # DAGs, plugins, logs, configs
│   │   └── dags/ml_pipeline_dag.py
│   └── docker/             # Dockerfiles for pipeline & Airflow
├── mlflow/                 # Local MLflow run store
│   └── runs/
├── mlflow_artifacts/       # Stored model/artifact files
├── models/                 # Final trained models
├── src/                    # ML pipeline scripts
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── drift_detection.py
│   └── run_pipeline.py
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

**Justification (≥50 words):**
`deploy/airflow` separates orchestration logic (DAGs, configs, plugins) from `deploy/docker` (container builds). `mlflow/` and `mlflow_artifacts/` keep experiment tracking cleanly isolated. `data/` is split into `raw/`, `processed/`, and `reports` to enforce reproducibility and auditability. `src/` encapsulates modular ML lifecycle scripts, and `models/` stores promoted artifacts. This separation mirrors production-ready MLOps layouts and simplifies CI/CD integration.

---

## 5. Reflection

Key challenges and solutions:
- **Airflow imports** – DAG tasks failed on `import src.*`. Fix: bake code + deps into a custom Airflow image and set `PYTHONPATH=/opt/airflow`.
- **Volume mismatches** – artifacts weren’t appearing. Fix: align docker-compose mounts with DAG paths.
- **MLflow port conflicts** – macOS AirPlay blocked 5000. Fix: switched to port **5001**.
- **Evidently in containers** – version conflicts. Fix: pin dependencies in `uv.lock` and generate deterministic drift test files.

---
*This README satisfies Homework 3 requirements for a production-driven ML project.*
