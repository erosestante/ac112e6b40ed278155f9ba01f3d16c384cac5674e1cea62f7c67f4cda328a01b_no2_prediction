# src/model_training.py

import os
import joblib
import numpy as np
import optuna
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional, Dict

from sklearn.base import RegressorMixin
from src.evaluation import evaluate_regression, save_metrics
# Import MLflow but DO NOT configure it at module scope
import mlflow
import mlflow.pyfunc

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---- PyFunc wrapper (unchanged) ----
class XGBPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a trained XGBoost model; reindexes inputs to the training feature order."""

    def __init__(self, feature_names: List[str]):
        self._feature_names = feature_names
        self._model = None

    def load_context(self, context):
        self._model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        import pandas as pd
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        X = (
            model_input.reindex(columns=self._feature_names, fill_value=0)
            .select_dtypes("number")
        )
        return self._model.predict(X)


def train_naive_baseline(
    val_df,
    target_col: str = "NO2_umol_m2",
    lag_col: str = "NO2_prev_month"
) -> Optional[Dict[str, float]]:
    v = val_df.dropna(subset=[lag_col])
    if v.empty:
        return None
    return evaluate_regression(v[target_col].values, v[lag_col].values)

def train_xgb_with_optuna(
    train_df,
    val_df,
    exclude_cols: List[str],
    target_col: str = "NO2_umol_m2",
    n_trials: int = 10,
    model_path: str = "models/xgb.pkl",
    figures_dir: str = "reports/figures",
    reports_dir: str = "reports",
) -> Tuple[RegressorMixin, Dict[str, float], List[Tuple[str, float]], str]:

    # === Data Prep ===
    X_train = train_df.drop(columns=exclude_cols).select_dtypes("number")
    y_train = train_df[target_col].values
    X_val   = val_df.drop(columns=exclude_cols).select_dtypes("number")
    y_val   = val_df[target_col].values
    X_val   = X_val.reindex(columns=X_train.columns, fill_value=0)

    if X_train.shape[0] < 10:
        raise ValueError("Not enough training rows for XGBoost")

    # === Hyperparameter search ===
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "verbosity": 0,
            "random_state": 42,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, verbose=False)
        preds = m.predict(X_val)
        return evaluate_regression(y_val, preds)["smape"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # === Best model ===
    best = study.best_params
    best.update({"verbosity": 0, "random_state": 42})
    model = xgb.XGBRegressor(**best)
    model.fit(X_train, y_train, verbose=False)

    # Persist locally
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # Metrics
    preds = model.predict(X_val)
    metrics: Dict[str, float] = evaluate_regression(y_val, preds)

    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, Path(reports_dir) / "evaluation_results.json")

    # SHAP + features
    explainer = shap.Explainer(model, X_train)
    shap_vals = explainer(X_val)
    importances = np.abs(shap_vals.values).mean(axis=0)
    feats = X_train.columns.tolist()
    top10 = sorted(zip(feats, importances), key=lambda x: x[1], reverse=True)[:10]

    # Plots
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_vals, X_val, plot_type="bar", feature_names=feats, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_bar.png"))
    plt.close()

    shap.summary_plot(shap_vals, X_val, feature_names=feats, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_beeswarm.png"))
    plt.close()

    # === MLflow logging ===
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("NO2-forecast")

    with mlflow.start_run(run_name="xgb_regression") as run:
        run_id = run.info.run_id

        # Params (only key ones)
        mlflow.log_params({
            "learning_rate": best["learning_rate"],
            "n_estimators":  best["n_estimators"],
            "max_depth":     best["max_depth"],
        })

        # Metrics
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        # Artifacts (let MLflow copy into /mlflow/artifacts internally)
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifacts(figures_dir, artifact_path="figures")
        mlflow.log_artifact(str(Path(reports_dir) / "evaluation_results.json"),
                            artifact_path="reports")

        # PyFunc wrapper
        pyfunc = XGBPyfuncWrapper(feature_names=feats)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pyfunc,
            artifacts={"model_path": model_path},
        )

    return model, metrics, top10, run_id
