# deploy/airflow/dags/ml_pipeline_dag.py

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import mlflow
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

import sys
if "/opt/airflow" not in sys.path:
    sys.path.append("/opt/airflow")

# --- Project imports ---
from src.data_preprocessing import load_and_prepare_all
from src.feature_engineering import engineer_features
from src.model_training import train_naive_baseline, train_xgb_with_optuna
from src.evaluation import save_metrics
from src.drift_detection import detect_drift

# ---------- DAG Defaults ----------
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 7, 29),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ---------- Helper ----------
def time_based_split(df, date_col: str = "date", frac_train: float = 0.8):
    dates = pd.to_datetime(df[date_col]).sort_values().unique()
    split = int(len(dates) * frac_train)
    train_dates, val_dates = dates[:split], dates[split:]
    train = df[df[date_col].isin(train_dates)].copy()
    val   = df[df[date_col].isin(val_dates)].copy()  # 👈 use val_dates
    return train, val

with DAG(
    dag_id="ml_pipeline_dag",
    description="ML pipeline (NO2) with drift detection and branching",
    default_args=default_args,
    schedule=None,   # Airflow 3.x
    catchup=False,
    tags=["ml", "no2", "xgboost"],
) as dag:

    # 👇 MLflow URI configured once per DAG context
    mlflow.set_tracking_uri("http://mlflow:5000")

    # ---------- Core tasks ----------
    def _preprocess():
        import numpy as np
        import pandas as pd

        RAW_CITY  = Path("/opt/airflow/data/raw/PHL_HUCs_Monthly_NO2_2018_2025.csv")
        RAW_PROV  = Path("/opt/airflow/data/raw/PHL_Provinces_Monthly_NO2_2018_2025.csv")
        PARQ_DIR  = Path("/opt/airflow/data/raw/Archive")
        PROCESSED = Path("/opt/airflow/data/processed")
        DATA_DIR  = Path("/opt/airflow/data")

        PROCESSED.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 1) your original preprocessing
        df = load_and_prepare_all(RAW_CITY, RAW_PROV, PARQ_DIR)
        df.to_parquet(PROCESSED / "preprocessed.parquet")

        # 2) create clean + drifted CSVs used by drift_detection
        target_col = "NO2_umol_m2"
        date_col   = "date"

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        dates = np.sort(df[date_col].unique())
        split = int(0.8 * len(dates))
        train = df[df[date_col].isin(dates[:split])].reset_index(drop=True)
        test  = df[df[date_col].isin(dates[split:])].reset_index(drop=True)

        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]
        X_test  = test.drop(columns=[target_col])
        y_test  = test[target_col]

        # numeric-only drift (strong & obvious)
        non_features = {target_col, date_col, "region"}
        num_cols = [c for c in X_train.columns
                    if c not in non_features and np.issubdtype(X_train[c].dtype, np.number)]

        X_test_drifted = X_test.copy()
        if num_cols:
            stds = X_train[num_cols].astype(float).std(ddof=0)
            min_pos = stds[stds > 0].min()
            stds = stds.replace(0, min_pos if pd.notna(min_pos) else 1.0).fillna(1.0)

            rng   = np.random.default_rng(42)
            shift = 0.5 * stds.values                         # +0.5*std mean shift
            noise = rng.normal(0.0, 0.3 * stds.values,        # 0.3*std noise
                            size=(len(X_test_drifted), len(num_cols)))
            X_test_drifted.loc[:, num_cols] = (
                X_test_drifted.loc[:, num_cols].astype(float).values + shift + noise
            )

            # amplify on the top-variance 5 features
            topk = X_train[num_cols].var().sort_values(ascending=False).index[:5]
            X_test_drifted.loc[:, topk] = X_test_drifted.loc[:, topk].astype(float) * 1.5

        # save what the drift task expects
        (X_train.assign(**{target_col: y_train})).to_csv(DATA_DIR / "train.csv", index=False)
        (X_test.assign(**{target_col: y_test})).to_csv(DATA_DIR / "test.csv", index=False)
        (X_test_drifted.assign(**{target_col: y_test})).to_csv(DATA_DIR / "drifted_test.csv", index=False)

    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=_preprocess,
    )

    def _feature_engineer():
        PROCESSED = Path("/opt/airflow/data/processed")
        RAW_CITY  = Path("/opt/airflow/data/raw/PHL_HUCs_Monthly_NO2_2018_2025.csv")

        df = pd.read_parquet(PROCESSED / "preprocessed.parquet")
        df = engineer_features(df, RAW_CITY)
        df.to_parquet(PROCESSED / "features.parquet")

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=_feature_engineer,
    )

    def _train_model():
        PROCESSED = Path("/opt/airflow/data/processed")
        REPORTS   = Path("/opt/airflow/data/reports")
        MODELS    = Path("/opt/airflow/data/models")

        REPORTS.mkdir(parents=True, exist_ok=True)
        MODELS.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(PROCESSED / "features.parquet")
        df = df.dropna(subset=["NO2_prev_month"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

        train_df, val_df = time_based_split(df, "date", 0.8)

        # Baseline
        naive = train_naive_baseline(val_df) or {}
        save_metrics(naive, REPORTS / "naive_metrics.json")

        # XGBoost
        exclude = ["region", "date", "NO2_umol_m2", "NO2_prev_month"]
        model, xgb_metrics, *_ = train_xgb_with_optuna(
            train_df,
            val_df,
            exclude_cols=exclude,
            model_path=str(MODELS / "xgb.pkl"),
            figures_dir=str(REPORTS),
        )
        save_metrics(xgb_metrics, REPORTS / "xgb_metrics.json")

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    def _evaluate_model():
        REPORTS = Path("/opt/airflow/data/reports")
        xgb = json.loads((REPORTS / "xgb_metrics.json").read_text())
        print(f"XGB -> MAE={xgb.get('mae')}, SMAPE={xgb.get('smape')}")

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )

    def _drift_detection():
        DATA_DIR  = Path("/opt/airflow/data")
        REPORTS   = Path("/opt/airflow/data/reports")

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS.mkdir(parents=True, exist_ok=True)

        # Use the files written by preprocess_data()
        test_ref = DATA_DIR / "test.csv"           # undrifted
        test_cur = DATA_DIR / "drifted_test.csv"   # drifted

        # Fail fast if they don't exist (forces correct order and avoids weak dummy data)
        if not test_ref.exists() or not test_cur.exists():
            raise FileNotFoundError(
                f"Missing input files for drift detection. "
                f"Expected: {test_ref} and {test_cur}. "
                f"Make sure the preprocess task ran and saved them."
            )

        result = detect_drift(str(test_ref), str(test_cur))
        drift = bool(result.get("drift_detected"))
        overall = result.get("overall_drift_score")

        if drift:
            logging.info("⚠️ Data drift detected in test set! Model retraining required.")
        else:
            logging.info("✅ No drift detected in test set.")

        logging.info("Drift summary: drift_detected=%s, overall_drift_score=%s", drift, overall)

        # If detect_drift returned HTML locations, surface them too
        html_paths = result.get("html_paths") or []
        if html_paths:
            logging.info("Drift HTML report saved at: %s", html_paths)

        # Mirror into /opt/airflow/data/reports for the branch step
        src_json = Path("/opt/airflow/data/reports/drift_report.json")
        dst_json = REPORTS / "drift_report.json"
        if not src_json.exists():
            raise FileNotFoundError(f"Expected drift report at {src_json}, but it was not created.")
        dst_json.write_text(src_json.read_text())

    drift_detection = PythonOperator(
        task_id="drift_detection",
        python_callable=_drift_detection,
    )

    # ---------- Branching ----------
    def _branch_on_drift() -> str:
        REPORTS = Path("/opt/airflow/data/reports")
        drift_json = REPORTS / "drift_report.json"

        if not drift_json.exists():
            return "pipeline_complete"

        data = json.loads(drift_json.read_text())
        return "retrain_model" if data.get("drift_detected") else "pipeline_complete"

    branch_on_drift = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=_branch_on_drift,
    )

    # ---------- End tasks ----------
    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=_train_model,  # reuse same logic
    )

    pipeline_complete = EmptyOperator(task_id="pipeline_complete")

    # ---------- Dependencies ----------
    preprocess_data >> feature_engineering >> train_model >> evaluate_model \
        >> drift_detection >> branch_on_drift >> [retrain_model, pipeline_complete]
