# run_pipeline.py
# flake8: noqa: E402
# ruff: noqa: E402
from __future__ import annotations

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
import logging
import numpy as np
import pandas as pd

from src.data_preprocessing import load_and_prepare_all
from src.feature_engineering import engineer_features
from src.model_training import train_naive_baseline, train_xgb_with_optuna
from src.evaluation import save_metrics
from src.drift_detection import detect_drift


def time_based_split(df: pd.DataFrame, date_col: str = "date", frac_train: float = 0.8):
    dates = pd.to_datetime(df[date_col]).sort_values().unique()
    split = int(len(dates) * frac_train)
    train_dates, val_dates = dates[:split], dates[split:]
    train = df[df[date_col].isin(train_dates)].copy()
    val   = df[df[date_col].isin(val_dates)].copy()
    return train, val


def make_test_and_drift_files(df: pd.DataFrame, data_dir: Path, target_col="NO2_umol_m2", date_col="date"):
    """Create reference/test CSVs and a drifted test to trigger Evidently."""
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

    non_features = {target_col, date_col, "region"}
    num_cols = [c for c in X_train.columns
                if c not in non_features and np.issubdtype(X_train[c].dtype, np.number)]

    X_test_drifted = X_test.copy()
    if num_cols:
        stds = X_train[num_cols].astype(float).std(ddof=0)
        min_pos = stds[stds > 0].min()
        stds = stds.replace(0, min_pos if pd.notna(min_pos) else 1.0).fillna(1.0)

        rng   = np.random.default_rng(42)
        shift = 0.5 * stds.values
        noise = rng.normal(0.0, 0.3 * stds.values, size=(len(X_test_drifted), len(num_cols)))
        X_test_drifted.loc[:, num_cols] = (
            X_test_drifted.loc[:, num_cols].astype(float).values + shift + noise
        )

        topk = X_train[num_cols].var().sort_values(ascending=False).index[:5]
        X_test_drifted.loc[:, topk] = X_test_drifted.loc[:, topk].astype(float) * 1.5

    data_dir.mkdir(parents=True, exist_ok=True)
    (X_train.assign(**{target_col: y_train})).to_csv(data_dir / "train.csv", index=False)
    (X_test.assign(**{target_col: y_test})).to_csv(data_dir / "test.csv", index=False)
    (X_test_drifted.assign(**{target_col: y_test})).to_csv(data_dir / "drifted_test.csv", index=False)

    return data_dir / "test.csv", data_dir / "drifted_test.csv"


def main():
    # ---------- Logging ----------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

    # ---------- MLflow setup (single source of truth) ----------
    import mlflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "no2-experiment")
    public_base = os.getenv("MLFLOW_PUBLIC_URL", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logging.info("MLflow tracking URI: %s | experiment: %s", mlflow.get_tracking_uri(), experiment_name)

    # Start one run for the whole pipeline
    with mlflow.start_run(run_name="cli_pipeline") as active_run:
        run_id = active_run.info.run_id
        logging.info("MLflow run_id: %s", run_id)

        # ---------- Paths ----------
        CITY_PATH = Path("data/raw/PHL_HUCs_Monthly_NO2_2018_2025.csv")
        PROV_PATH = Path("data/raw/PHL_Provinces_Monthly_NO2_2018_2025.csv")
        PARQ_DIR  = Path("data/raw/Archive")
        PROCESSED = Path("data/processed")
        REPORTS   = Path("data/reports")
        MODELS    = Path("models")
        DATA_DIR  = Path("data")
        for p in (PROCESSED, REPORTS, MODELS):
            p.mkdir(parents=True, exist_ok=True)

        # Basic run metadata
        mlflow.log_params({
            "city_path": str(CITY_PATH),
            "prov_path": str(PROV_PATH),
            "parq_dir": str(PARQ_DIR),
            "code_version": os.getenv("GIT_COMMIT_SHA", "local"),
        })

        # ---------- 1) Load & preprocess ----------
        logging.info("1) Loading & preprocessing data...")
        df = load_and_prepare_all(CITY_PATH, PROV_PATH, PARQ_DIR)
        df.to_parquet(PROCESSED / "preprocessed.parquet", index=False)
        logging.info("   ➤ Combined dataframe shape: %s", df.shape)
        mlflow.log_metric("n_rows_raw", float(df.shape[0]))

        # ---------- 2) Feature engineering ----------
        logging.info("2) Engineering features...")
        df = engineer_features(df, CITY_PATH)
        df.to_parquet(PROCESSED / "features.parquet", index=False)
        logging.info("   ➤ After feature engineering: %s", df.shape)
        mlflow.log_metric("n_rows_features", float(df.shape[0]))

        # Clean like DAG
        df = df.dropna(subset=["NO2_prev_month"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

        # ---------- 3) Train/Eval on time split ----------
        train_df, val_df = time_based_split(df, "date", 0.8)
        logging.info("3) Train/Val shapes: train=%s, val=%s", train_df.shape, val_df.shape)
        if val_df.empty:
            logging.warning("No validation data; exiting.")
            mlflow.set_tag("pipeline_final_status", "no_validation_data")
            return

        # Baseline
        naive = train_naive_baseline(val_df) or {}
        save_metrics(naive, REPORTS / "naive_metrics.json")
        for k, v in naive.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"naive_{k}", float(v))

        # XGB with Optuna (must ensure it logs a model at artifact_path='model' OR we log it here)
        exclude = ["region", "date", "NO2_umol_m2", "NO2_prev_month"]
        model, xgb_metrics, top10, maybe_run_id = train_xgb_with_optuna(
            train_df, val_df,
            exclude_cols=exclude,
            model_path=str(MODELS / "xgb.pkl"),
            figures_dir=str(REPORTS),
        )

        # If the training function started its own run, prefer current run
        # Ensure artifacts are attached to *this* run:
        try:
            import mlflow.sklearn
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception as e:
            logging.warning("Could not log model artifact: %s", e)

        # Log metrics
        for k, v in (xgb_metrics or {}).items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"xgb_{k}", float(v))
        save_metrics(xgb_metrics or {}, REPORTS / "xgb_metrics.json")

        # Log SHAP top-10 as a text artifact for easy viewing
        try:
            top10_path = REPORTS / "top10_shap.txt"
            with open(top10_path, "w") as f:
                for name, val in (top10 or []):
                    f.write(f"{name}\t{val}\n")
            mlflow.log_artifact(str(top10_path), artifact_path="reports")
        except Exception as e:
            logging.warning("Could not log SHAP top10: %s", e)

        # Friendly links
        try:
            exp_id = active_run.info.experiment_id
            logging.info("View run:        %s/#/experiments/%s/runs/%s", public_base, exp_id, run_id)
            logging.info("View experiment: %s/#/experiments/%s", public_base, exp_id)
            mlflow.set_tag("run_link", f"{public_base}/#/experiments/{exp_id}/runs/{run_id}")
        except Exception as e:
            logging.warning("Could not set view links: %s", e)

        # ---------- 4) Drift detection ----------
        test_ref_path, test_cur_path = make_test_and_drift_files(df, DATA_DIR)
        logging.info("Drift inputs: %s vs %s", test_ref_path, test_cur_path)

        test_drift = detect_drift(str(test_ref_path), str(test_cur_path)) or {}
        logging.info("Test drift: %s", test_drift)
        mlflow.log_param("test_drift_detected", bool(test_drift.get("drift_detected")))
        if "overall_drift_score" in test_drift:
            mlflow.log_metric("test_overall_drift_score", float(test_drift["overall_drift_score"]))

        if bool(test_drift.get("drift_detected")):
            error_msg = "Data drift detected in test set! Model retraining required."
            mlflow.set_tag("pipeline_final_status", "drift_detected_error")
            mlflow.set_tag("pipeline_outcome", "drift_error_raised")
            mlflow.set_tag("drift_error_message", error_msg)
            raise ValueError(error_msg)

        # ---------- 5) Optional registration gate ----------
        MAE_THRESHOLD, SMAPE_THRESHOLD = 4.0, 10.0
        mae, smape = (xgb_metrics or {}).get("mae"), (xgb_metrics or {}).get("smape")
        meets_gate = (mae is not None and smape is not None and mae < MAE_THRESHOLD and smape < SMAPE_THRESHOLD)

        if meets_gate:
            model_name = os.getenv("MLFLOW_MODEL_NAME", "no2_xgb_cli")
            logging.info("✅ Threshold met. Registering model as '%s' ...", model_name)
            # This expects the artifact "model" to exist in this run:
            mlflow.register_model(f"runs:/{run_id}/model", model_name)
            mlflow.set_tag("pipeline_final_status", "registered")
        else:
            logging.info("❌ Threshold not met (mae=%s, smape=%s). Skipping registration.", mae, smape)
            mlflow.set_tag("pipeline_final_status", "completed_no_register")

        logging.info("🎉 Pipeline complete.")


if __name__ == "__main__":
    main()
