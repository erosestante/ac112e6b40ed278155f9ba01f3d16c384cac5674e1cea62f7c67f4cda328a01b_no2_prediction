# deploy/airflow/dags/ml_pipeline_dag.py

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

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
        RAW_CITY  = Path("/opt/airflow/data/raw/PHL_HUCs_Monthly_NO2_2018_2025.csv")
        RAW_PROV  = Path("/opt/airflow/data/raw/PHL_Provinces_Monthly_NO2_2018_2025.csv")
        PARQ_DIR  = Path("/opt/airflow/data/raw/Archive")
        PROCESSED = Path("/opt/airflow/data/processed")

        df = load_and_prepare_all(RAW_CITY, RAW_PROV, PARQ_DIR)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED / "preprocessed.parquet")

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
        PROCESSED = Path("/opt/airflow/data/processed")
        DATA_DIR  = Path("/opt/airflow/data")
        REPORTS   = Path("/opt/airflow/data/reports")

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS.mkdir(parents=True, exist_ok=True)

        test_ref = DATA_DIR / "test.csv"
        test_cur = DATA_DIR / "drifted_test.csv"
        drift_json = REPORTS / "drift_report.json"

        # Create dummy test sets if missing
        if not test_ref.exists() or not test_cur.exists():
            df = pd.read_parquet(PROCESSED / "features.parquet")
            df = df.dropna(subset=["NO2_prev_month"]).reset_index(drop=True)
            df["date"] = pd.to_datetime(df["date"])
            _, val_df = time_based_split(df, "date", 0.8)

            val_df.to_csv(test_ref, index=False)
            drifted = val_df.copy()
            for c in drifted.select_dtypes("number").columns[:3]:
                drifted[c] = drifted[c] * 1.1
            drifted.to_csv(test_cur, index=False)

        detect_drift(str(test_ref), str(test_cur), output_path=str(drift_json))

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
