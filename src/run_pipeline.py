# run_pipeline.py
from __future__ import annotations
from pathlib import Path
import os
import logging
import numpy as np
import pandas as pd

from src.data_preprocessing import load_and_prepare_all
from src.feature_engineering import engineer_features
from src.model_training import train_naive_baseline, train_xgb_with_optuna
from src.evaluation import save_metrics
from src.drift_detection import detect_drift

# ---------- MLflow setup ----------
import mlflow

# Store everything in ./mlflow/runs
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlflow/runs")
mlflow.set_tracking_uri(tracking_uri)

# (optional) give a default experiment name so runs don’t go to "Default"
mlflow.set_experiment("no2-experiment")

def time_based_split(df: pd.DataFrame, date_col: str = "date", frac_train: float = 0.8):
    dates = pd.to_datetime(df[date_col]).sort_values().unique()
    split = int(len(dates) * frac_train)
    train_dates, val_dates = dates[:split], dates[split:]
    train = df[df[date_col].isin(train_dates)].copy()
    val   = df[df[date_col].isin(val_dates)].copy()
    return train, val


def make_test_and_drift_files(df: pd.DataFrame, data_dir: Path, target_col="NO2_umol_m2", date_col="date"):
    """Replicates the DAG’s stronger numeric-only drift so Evidently flags it."""
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

    # ---------- MLflow setup ----------
    import mlflow
    # When running on HOST, default to localhost:5001 (mapped port)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)

    public_base = os.getenv("MLFLOW_PUBLIC_URL", "http://localhost:5001")  # for friendly links

    logging.info("Starting NO₂ pipeline with MLflow at %s", tracking_uri)

    # ---------- Paths (host paths mirroring DAG mounts) ----------
    CITY_PATH = Path("data/raw/PHL_HUCs_Monthly_NO2_2018_2025.csv")
    PROV_PATH = Path("data/raw/PHL_Provinces_Monthly_NO2_2018_2025.csv")
    PARQ_DIR  = Path("data/raw/Archive")
    PROCESSED = Path("data/processed")
    REPORTS   = Path("data/reports")
    MODELS    = Path("models")
    DATA_DIR  = Path("data")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Load & preprocess ----------
    logging.info("1) Loading & preprocessing data...")
    df = load_and_prepare_all(CITY_PATH, PROV_PATH, PARQ_DIR)
    (PROCESSED / "preprocessed.parquet").write_bytes(df.to_parquet(index=False))
    logging.info("   ➤ Combined dataframe shape: %s", df.shape)

    # ---------- 2) Feature engineering ----------
    logging.info("2) Engineering features...")
    df = engineer_features(df, CITY_PATH)
    (PROCESSED / "features.parquet").write_bytes(df.to_parquet(index=False))
    logging.info("   ➤ After feature engineering: %s", df.shape)

    # Clean up like DAG
    df = df.dropna(subset=["NO2_prev_month"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # ---------- 3) Train/Eval on time split (single period for CLI) ----------
    train_df, val_df = time_based_split(df, "date", 0.8)
    logging.info("3) Train/Val shapes: train=%s, val=%s", train_df.shape, val_df.shape)
    if val_df.empty:
        logging.warning("No validation data; exiting.")
        return

    # Baseline
    naive = train_naive_baseline(val_df) or {}
    save_metrics(naive, REPORTS / "naive_metrics.json")
    logging.info("Baseline metrics: %s", naive)

    # XGB with Optuna
    exclude = ["region", "date", "NO2_umol_m2", "NO2_prev_month"]
    model, xgb_metrics, top10, run_id = train_xgb_with_optuna(
        train_df, val_df, exclude_cols=exclude,
        model_path=str(MODELS / "xgb.pkl"),
        figures_dir=str(REPORTS),
    )
    save_metrics(xgb_metrics, REPORTS / "xgb_metrics.json")
    logging.info("XGB metrics: %s", xgb_metrics)
    logging.info("Top 10 SHAP: %s", top10)

    # Log friendly MLflow links
    try:
        run = mlflow.get_run(run_id)
        exp_id = run.info.experiment_id
        logging.info("View run (host): %s/#/experiments/%s/runs/%s", public_base, exp_id, run_id)
        logging.info("View experiment (host): %s/#/experiments/%s", public_base, exp_id)
    except Exception as e:
        logging.warning("Could not print MLflow links: %s", e)

    # ---------- 4) Drift detection on test vs drifted_test ----------
    # Create files like in DAG so results match
    test_ref_path, test_cur_path = make_test_and_drift_files(df, DATA_DIR)
    logging.info("Drift inputs: %s vs %s", test_ref_path, test_cur_path)

    test_drift = detect_drift(str(test_ref_path), str(test_cur_path))
    logging.info("Test drift: %s", test_drift)
    mlflow.log_param("test_drift_detected", bool(test_drift.get("drift_detected")))
    mlflow.log_param("test_overall_drift_score", test_drift.get("overall_drift_score"))

    # If drift is detected on the test pair, raise the exact error string
    if bool(test_drift.get("drift_detected")):
        error_msg = "Data drift detected in test set! Model retraining required."
        # Optional: tag in MLflow for easy filtering
        try:
            mlflow.set_tag("pipeline_final_status", "drift_detected_error")
            mlflow.set_tag("pipeline_outcome", "drift_error_raised")
            mlflow.set_tag("drift_error_message", error_msg)
        except Exception as _:
            pass
        raise ValueError(error_msg)

    # ---------- 5) Optional registration gate (same thresholds as DAG logic) ----------
    MAE_THRESHOLD, SMAPE_THRESHOLD = 4.0, 10.0
    mae, smape = xgb_metrics.get("mae"), xgb_metrics.get("smape")
    meets_gate = (mae is not None and smape is not None and mae < MAE_THRESHOLD and smape < SMAPE_THRESHOLD)

    if meets_gate:
        model_name = "no2_xgb_cli"
        logging.info("✅ Threshold met. Registering model as '%s' ...", model_name)
        mlflow.register_model(f"runs:/{run_id}/model", model_name)
    else:
        logging.info("❌ Threshold not met (mae=%s, smape=%s). Skipping registration.", mae, smape)

    logging.info("🎉 Pipeline complete.")


if __name__ == "__main__":
    main()
