# deploy/airflow/dags/ml_pipeline_taskflow_periods.py

from airflow import DAG
from airflow.decorators import task, task_group
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from src.data_preprocessing import load_and_prepare_all
from src.feature_engineering import engineer_features
from src.model_training import train_naive_baseline, train_xgb_with_optuna
from src.evaluation import save_metrics
from src.run_pipeline import time_based_split

RAW_CITY  = Path("/opt/airflow/data/raw/PHL_HUCs_Monthly_NO2_2018_2025.csv")
RAW_PROV  = Path("/opt/airflow/data/raw/PHL_Provinces_Monthly_NO2_2018_2025.csv")
PARQ_DIR  = Path("/opt/airflow/data/raw/Archive")
PROCESSED = Path("/opt/airflow/data/processed")
REPORTS   = Path("/opt/airflow/data/reports")
MODELS    = Path("/opt/airflow/data/models")

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 7, 29),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_pipeline_dag",
    default_args=default_args,
    description="ML pipeline for NO2 prediction with taskflow API",
    schedule=None,  # Correct for Airflow 3.x
    catchup=False,
    tags=["ml", "no2", "xgboost"],
) as dag:

    @task()
    def data_preprocessing() -> str:
        df = load_and_prepare_all(RAW_CITY, RAW_PROV, PARQ_DIR)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        path = PROCESSED / "preprocessed.parquet"
        df.to_parquet(path)
        return str(path)

    @task()
    def feature_engineering(preprocessed_path: str) -> str:
        df = pd.read_parquet(preprocessed_path)
        df = engineer_features(df, RAW_CITY)
        path = PROCESSED / "features.parquet"
        df.to_parquet(path)
        return str(path)

    @task()
    def split_periods(feature_path: str) -> list[dict]:
        df = pd.read_parquet(feature_path)
        df = df.dropna(subset=["NO2_prev_month"])
        df["date"] = pd.to_datetime(df["date"])

        periods = {
            "pre-pandemic":    df[df.date <  "2020-03-01"],
            "during-pandemic": df[(df.date >= "2020-03-01") & (df.date < "2022-01-01")],
            "post-pandemic":   df[df.date >= "2022-01-01"],
        }

        output = []
        for name, subset in periods.items():
            if subset.empty:
                continue
            path = PROCESSED / f"{name}.parquet"
            subset.to_parquet(path)
            output.append({"name": name, "path": str(path)})

        return output

    @task_group()
    def model_training_group(periods: list[dict]) -> list[dict]:

        @task()
        def train(period_info: dict) -> dict:
            name = period_info["name"]
            path = period_info["path"]

            df = pd.read_parquet(path)
            train_df, val_df = time_based_split(df, date_col="date", frac_train=0.8)

            REPORTS.joinpath(name).mkdir(parents=True, exist_ok=True)
            MODELS.mkdir(parents=True, exist_ok=True)

            naive_metrics = train_naive_baseline(val_df) or {}
            save_metrics(naive_metrics, REPORTS / name / "naive_metrics.txt")

            model, xgb_metrics, _ = train_xgb_with_optuna(
                train_df, val_df,
                exclude_cols=["region", "date", "NO2_umol_m2", "NO2_prev_month"],
                model_path=str(MODELS / f"xgb_{name}.pkl"),
                figures_dir=str(REPORTS / name),
            )
            save_metrics(xgb_metrics, REPORTS / name / "xgb_metrics.txt")

            return {"period": name, "naive": naive_metrics, "xgb": xgb_metrics}

        return train.expand(period_info=periods)

    @task()
    def evaluate(results: list[dict]):
        df = pd.DataFrame(results).set_index("period")
        print("\n🎉 Final Evaluation Summary:\n", df.T.to_string())

    prepped = data_preprocessing()
    featured = feature_engineering(prepped)
    splits = split_periods(featured)
    trained = model_training_group(splits)
    evaluate(trained)
