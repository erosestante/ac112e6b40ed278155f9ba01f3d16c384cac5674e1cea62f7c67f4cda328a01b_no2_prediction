# NO₂ Prediction with Urban Amenities and Satellite Data

## 1. Project Overview
Building on our Homework 1 ML pipeline for forecasting monthly NO₂ concentrations across Philippine provinces, this version adds:

- `Docker containerization` for reproducible, immutable environments. We start from a Python 3.12-slim base, install locked dependencies via UV, and bundle our ML code so anyone can clone and run without “it works on my machine” issues.

- `Apache Airflow orchestration` (official `apache/airflow:3.0.3` image) to automate preprocessing, feature engineering, training (baseline + XGBoost/Optuna), evaluation, and drift monitoring.

- `MLflow tracking + registry` for experiments, metrics, artifacts, and model versions.

- `Drift Simulation and Detection (Evidently)` with branching retrain logic

## 2. Setup Instructions

### Prerequisites

1. **Install Docker & Docker Compose**
   - **Docker Desktop** for macOS/Windows: https://www.docker.com/products/docker-desktop
   - **Docker Engine & Compose** for Linux via your package manager
2. **Verify versions**
   ```bash
   docker --version          # e.g. Docker version 24.x
   docker compose version    # e.g. Docker Compose version v2.x
3. **Ensure sufficient resources**
  - ≥2 CPUs, ≥4 GB RAM, ≥10 GB disk

4. **Clone the Repository**
    ```bash
    git clone git@github.com:erosestante/ac112e6b40ed278155f9ba01f3d16c384cac5674e1cea62f7c67f4cda328a01b_no2_prediction.
    cd ac112e6b40ed278155f9ba01f3d16c384cac5674e1cea62f7c67f4cda328a01b_no2_prediction
    git checkout hw3-mlflow-drift

### Folder Structure
```text
.
├── data/
│   ├── models/                        # Intermediate models (optional, e.g., cross-validation outputs)
│   ├── processed/                     # Feature-engineered datasets (e.g., parquet files)
│   └── raw/                           # Raw NO₂ data (CSV)
│       ├── PHL_HUCs_Monthly_NO2_2018_2025.csv
│       ├── PHL_Provinces_Monthly_NO2_2018_2025.csv
│       └── Archive/
│           └── features_quarter=YYYY-MM-DD/
│               ├── _SUCCESS
│               ├── _SUCCESS.crc
│               └── part-*.snappy.parquet
│
├── models/                            # Final trained XGBoost model artifacts (.pkl)
│
├── reports/                           # Pipeline-generated evaluation reports and plots
│
├── src/                               # ML pipeline scripts
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── run_pipeline.py
│
├── deploy/                            # Environment-specific configuration and orchestration
│   ├── airflow/                       # Airflow DAGs and settings
│   │   ├── config/                    # Airflow configs (e.g., connections, variables)
│   │   ├── dags/                      # DAG definitions
│   │   ├── logs/                      # Execution logs
│   │   └── plugins/                   # Custom Airflow plugins
│   └── docker/                        # Docker-related files
│       ├── Dockerfile                 # For containerizing the ML pipeline
│       ├── airflow.Dockerfile         # For running Airflow inside a container
│       └── .dockerignore              # Exclude unnecessary files from build context
│
├── .env                               # Local environment variables
├── .gitignore
├── .pre-commit-config.yaml            # Linting & formatting rules via pre-commit
├── .yamllint.yml                      # YAML linting config
├── docker-compose.yml                 # Multi-container orchestration (Airflow + ML pipeline)
├── pyproject.toml                     # Python dependencies (preferred over requirements.txt)
├── uv.lock                            # Locked package versions (used by `uv`)
├── requirements.txt                   # Optional pip-compatible dependency list
└── README.md                          # Project overview and usage guide

```
I separated `airflow` and `docker` under `deploy` because they serve different purposes—`airflow` handles orchestration and scheduling, while `docker` manages containerization. This separation keeps the project modular, maintainable, and easier to scale.
- `airflow/` handles DAGs, configs, and scheduling logic.
- `docker/` contains Dockerfiles for building the ML pipeline and orchestration environments.

##  3. Containerization of the ML Pipeline with Docker
1. Docker Set-up

The ML pipeline is containerized using a lightweight `python:3.12-slim` base image and `uv` for fast, reproducible dependency management. The Dockerfile is located at: `/deploy/docker/Dockerfile`. The image is optimized by removing unnecessary layers and tools, ensuring faster builds and smaller image size—ideal for CI/CD workflows.

2. Build the ML Docker Image
  To build the ML Docker Image, run this from the root of your project (replace <hashed_value> with your actual project hash or ID):
  ```bash
    docker build -f deploy/docker/Dockerfile -t <hashed_value>-ml-pipeline .
  ```

3. Run the pipeline
  Mount the local folders for data, models, and reports when running the container:
  ```bash
  docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/reports:/app/reports" \
  <hashed_value>-ml-pipeline
  ```

  This will execute the default command specified in the Dockerfile.
  ```bash
  python src/run_pipeline.py
  ```

## 4. Orchestratation with Airflow (DockerCompose)

This project runs Airflow using Docker Compose and a custom image built from `deploy/docker/airflow.Dockerfile`.
Compose wires up the webserver, scheduler, API server, Postgres, and Redis, and mounts your project folders for reproducible runs.

### `.env` (required)
Create a `.env` at the project root so Compose picks up consistent paths, image name, and user mapping:
```dotenv
AIRFLOW_PROJ_DIR=./deploy/airflow
AIRFLOW_UID=501                 # set to your host UID to avoid permission issues
AIRFLOW_IMAGE_NAME=custom-airflow:latest
```
- `AIRFLOW_PROJ_DIR` maps local deploy/airflow/{dags,logs,plugins,config} to /opt/airflow/... in containers.
- `AIRFLOW_UID` should match your local user (id -u on Linux).
- `AIRFLOW_IMAGE_NAME` is the tag used after building from deploy/docker/airflow.Dockerfile.

### Volumes (from docker-compose.yml)
- ./deploy/airflow/dags → /opt/airflow/dags
- ./deploy/airflow/plugins → /opt/airflow/plugins
- ./deploy/airflow/logs → /opt/airflow/logs
- ./deploy/airflow/config → /opt/airflow/config
- ./data → /opt/airflow/data (raw, processed, models, reports)
- ./reports → /reports, ./models → /models (optional host access)

### Bring Up the Stack
From the project root, run the following in the bash terminal
```bash
# Build the custom Airflow image defined in deploy/docker/airflow.Dockerfile
docker compose build

# Start services (Postgres, Redis, API server, Scheduler, etc.)
docker compose up -d

# Check services & follow logs
docker compose ps
```

### Access and Trigger
1. Open the Airflow UI
- Go to: http://localhost:8080
- Login: airflow / airflow (created by the airflow-init service)

2. Enable the DAG
- In the UI, find `ml_pipeline_taskflow_periods`
- Toggle the DAG ON (top-right of the DAG page)

3. Trigger a run (UI)
- Click the Play ▶️ button → Trigger DAG
- Watch progress in Graph View; click each task for Logs

## 5. Docker Integration

### A.Standalone ML image — for quick, end-to-end runs and CI smoke tests
Dockerfile: `deploy/docker/Dockerfile`

Purpose: Provide a self-contained runner for the entire pipeline so you (or CI) can execute python src/run_pipeline.py without installing Python or any libs on the host.

When to use this image:
- Quickly validate the pipeline on a teammate’s machine or CI.
- Iterate on feature engineering or model training with a simple, linear run.
- You don’t need orchestration (retries/scheduling) yet.

How it works
- Base: `python:3.12-slim` (small, fast to pull).
- Installs dependencies with uv using uv.lock → reproducible and deterministic.
- Default command runs: python src/run_pipeline.py.

Why it’s structured this way
- Layer caching: copy pyproject.toml and uv.lock before src/ so dependency layers are reused unless deps change.
- Single stage: keeps image small and simple → faster builds/pulls in CI/CD.
- Stateless: inputs/outputs live in bind-mounted volumes, not the image.

Data persistence (volumes at run time)
- `./data → /app/data`
- `./models → /app/models`
- `./reports → /app/reports`

### B.Custom Airflow image — for orchestration, retries, and scheduled runs
Dockerfile: `deploy/docker/airflow.Dockerfile`

Purpose: Extend `apache/airflow:3.0.3` and bake in your project code + dependencies so tasks can import src.* reliably inside Airflow containers.

When to use this image
- Need task-level observability (per-task logs, retries).
- Need scheduling (manual or cron-like).
- Want to parallelize pieces (e.g., train per period) and capture metrics/artifacts per task.

How it works
- Sets PYTHONPATH=/opt/airflow → src/ is importable in all tasks.
- Installs uv and apache-airflow-providers-docker.
- Runs uv pip install --system . to register your project in the container’s Python env.

Why is it necessary:
- Airflow runs tasks in separate processes/containers. If your project isn’t installed in the image (or mounted correctly), import src.* fails.
- Baking code+deps into the image ensures consistent environments across scheduler, workers, and CLI.

## 6. Airflow DAG
File: `deploy/airflow/dags/ml_pipeline_dag.py`

Style: TaskFlow API with a @task_group that expands over time periods (pre/during/post pandemic).

### A. Configuration
- schedule: `None` (manual) — easy to switch to cron later
- catchup: `False` — avoids backfilling old dates automatically
- retries: `1` with `retry_delay=5m` — basic resilience to transient failures
- tags: `ml`, `no2`, `xgboost`

There are in-container paths present in the DAG that must much the the compose volume Thus, if you change the mounts in `docker-compose.yml`, update these paths in the DAG so task read/write to the correct locations.

### B. Tasks (inputs → outputs)
1. `data_preprocessing`: Load, clean, and combine city-level and province-level NO₂ data, and merge them with external urban amenity features.
2. `feature_engineering`: Add machine learning-friendly features like lagged NO₂, seasonal indicators, and HUC flags.
3. `split_periods`: Separate the data into 3 time-based regimes to model behavior differences across events like the COVID-19 pandemic.
4. `model_training_group`: Train and evaluate a baseline and an XGBoost model for each period.
5. `evaluate`: Collect and summarize all results at the end of the DAG run.

### Dependency Chain
```nginx
data_preprocessing
  → feature_engineering
  → split_periods
  → model_training_group
  → evaluate
```

## 8. Reflection
During development, one major challenge was mounting the src/ folder into Airflow containers. Initially, I repeatedly rebuilt images, but the actual fix was simply restarting services with the correct mounted paths. Another difficulty was configuring Evidently for drift detection inside the container environment, which required careful handling of dependencies and ensuring drifted test sets were generated properly. These obstacles highlighted the importance of debugging with container restarts before rebuilding images and of incrementally validating each DAG task. Once resolved, the pipeline achieved full reproducibility with minimal friction between Docker, Airflow, and MLflow.

---
*This README satisfies course requirements for a production-driven ML project.*
