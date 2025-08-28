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
# 1) Install uv (once)
pip install uv

# 2) Create & activate a virtual env (Python 3.12+ recommended)
uv venv --python=3.12
source .venv/bin/activate

# 3) Install your package + deps from pyproject.toml
uv pip install -e .

# 4) (Optional) Install pre-commit hooks
pre-commit install

# 5) Start MLflow tracking server on :5001 (new terminal tab recommended)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 -p 5001
# Keep this running. Open http://localhost:5001 in your browser.

# 6) Run the pipeline (in another terminal tab with the venv activated)
python src/run_pipeline.py
```

#### 2. MLflow Integration

##### MLflow Tracking

- **Tracking server:** defaults to `http://localhost:5001` (set in `run_pipeline.py`).
- **UI:** open http://localhost:5001

**What gets logged per pipeline run**

- **Pipeline parameters (parent run `cli_pipeline`):**
  - `city_path`, `prov_path`, `parq_dir`, `code_version`
  - Drift flag: `test_drift_detected` (True/False)

- **Model parameters (child run `xgb_regression`):**
  - `learning_rate`
  - `max_depth`
  - `n_estimators`

- **Metrics**
  - **Baseline (prefixed `naive_`):** `naive_mae`, `naive_smape` *(and any others returned by `evaluate_regression`)*
  - **XGBoost (prefixed `xgb_`):** `xgb_mae`, `xgb_smape` *(used by the registration gate)*
  - **Drift/quality:** `test_overall_drift_score` (metric)

- **Artifacts**
  - Data: `data/processed/preprocessed.parquet`, `data/processed/features.parquet`
  - Reports: `reports/evaluation_results.json`, `reports/top10_shap.txt`
  - Figures: `figures/shap_bar.png`, `figures/shap_beeswarm.png`
  - Model: logged at `artifact_path="model"` (plus PyFunc wrapper)

- **Model registration**
  - If `xgb_mae < 4.0` **and** `xgb_smape < 10.0`, the model is registered as **`no2_xgb_cli`** via `runs:/<child_run_id>/model`.

- **UI access**: open **http://localhost:5001** to view experiments, runs, artifacts, and the Model Registry.
  - The pipeline run (`cli_pipeline`) appears as the **parent** run.
  - The XGBoost training (`xgb_regression`) appears as a **child** (nested) run under it.


#### 3. Drift Detection

**What gets generated (by `run_pipeline.py`):**
- `data/train.csv`, `data/test.csv`, and `data/drifted_test.csv`

**How we simulate drift (numeric-only):**
We create a synthetic “current” dataset (`drifted_test.csv`) from the original test split by perturbing only numeric feature columns (excluding targets and non-features like `date` and `region`). Concretely, we add a **+0.5σ mean shift** and **+0.3σ Gaussian noise** per numeric column (σ computed from the training set). To make drift clearly detectable, we also **amplify the top-variance features by 1.5×**. The transformation uses a fixed random seed (`42`) for reproducibility and maintains column order by reindexing to the training feature set.

**How detection works (Evidently + fallback):**
We run **Evidently’s `DataDriftPreset`** on the reference (`test.csv`) versus current (`drifted_test.csv`) data, extract per-feature `drift_score`s, and compute an `overall_drift_score` (mean of available scores). If Evidently indicates dataset-level drift (via `dataset_drift` or `drift_share > 0`), we flag drift. If Evidently fails (version/layout differences), we fall back to a simple **Cohen’s d** effect size per feature and flag drift when **any feature ≥ 0.5** (medium effect).

**Outputs & logging:**
- JSON report saved to: `reports/drift_report.json`
- In MLflow, the pipeline logs:
  - Param: `test_drift_detected` (True/False)
  - Metric: `test_overall_drift_score`
- If drift is detected, the run is **tagged** (`pipeline_final_status=drift_detected_error`, etc.) and the pipeline **raises an error** to halt downstream steps.

#### 4. Testing Instructions
```bash
# Run the pipeline
python src/run_pipeline.py

# Verify MLflow UI
open http://localhost:5001
```

---

### B) Airflow DAG Pipeline

#### 1. Setup Instructions
```bash
# Build + start everything
docker compose up -d --build

# Check containers
docker compose ps

# Webserver health (200 OK when ready)
curl -I http://localhost:8080/health

# MLflow health (optional)
curl -I http://localhost:5001
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

**Justification:**
`deploy/airflow` separates orchestration logic (DAGs, configs, plugins) from `deploy/docker` (container builds). `mlflow/` and `mlflow_artifacts/` keep experiment tracking cleanly isolated. `data/` is split into `raw/`, `processed/`, and `reports` to enforce reproducibility and auditability. `src/` encapsulates modular ML lifecycle scripts, and `models/` stores promoted artifacts. This separation mirrors production-ready MLOps layouts and simplifies CI/CD integration.

---

## 5. Reflection

### Key Challenges & Solutions

- **Airflow couldn’t import `src.*` / tasks failed**
  - **Cause:** `./src` wasn’t mounted into Airflow containers.
  - **Fix (in `x-airflow-common` so all Airflow services inherit it):**
    ```yaml
    volumes:
      - ${AIRFLOW_PROJ_DIR:-.}/dags:/opt/airflow/dags
      - ${AIRFLOW_PROJ_DIR:-.}/logs:/opt/airflow/logs
      - ${AIRFLOW_PROJ_DIR:-.}/config:/opt/airflow/config
      - ${AIRFLOW_PROJ_DIR:-.}/plugins:/opt/airflow/plugins
      - ./data:/opt/airflow/data
      - ./reports:/opt/airflow/reports
      - ./models:/opt/airflow/models
      - ./src:/opt/airflow/src        # ✅ code now importable
      - mlflow-artifacts:/mlflow       # ✅ shared MLflow artifacts
    ```

- **MLflow port conflict (macOS AirPlay on :5000)**
  - **Fix:** Use **:5001** end-to-end (Compose `5001:5001`, `MLFLOW_TRACKING_URI=http://mlflow:5001`, UI at `http://localhost:5001`).

---
*This README satisfies Homework 3 requirements for a production-driven ML project.*
