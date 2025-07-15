import os
import joblib
import numpy as np
import optuna
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin
from typing import Tuple, List

from evaluation import evaluate_regression
from typing import Optional, Dict


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
    n_trials: int = 30,
    model_path: str = "models/xgb.pkl",
    figures_dir: str = "reports"
) -> Tuple[RegressorMixin, dict[str, float], List[Tuple[str, float]]]:
    X_train = train_df.drop(columns=exclude_cols).select_dtypes("number")
    y_train = train_df[target_col].values
    X_val   = val_df.drop(columns=exclude_cols).select_dtypes("number")
    y_val   = val_df[target_col].values

    # align columns
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    if X_train.shape[0] < 10:
        raise ValueError("Not enough training rows for XGBoost")

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
        return evaluate_regression(y_val, preds)["SMAPE"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    best.update({"verbosity": 0, "random_state": 42})
    model = xgb.XGBRegressor(**best)
    model.fit(X_train, y_train, verbose=False)

    # persist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # metrics
    preds = model.predict(X_val)
    metrics = evaluate_regression(y_val, preds)

    # SHAP
    explainer = shap.Explainer(model, X_train)
    shap_vals = explainer(X_val)
    importances = np.abs(shap_vals.values).mean(axis=0)
    feats = X_train.columns.tolist()
    top10 = sorted(zip(feats, importances), key=lambda x: x[1], reverse=True)[:10]

    # plots
    os.makedirs(figures_dir, exist_ok=True)
    shap.summary_plot(shap_vals, X_val, plot_type="bar", feature_names=feats, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_bar.png"))
    plt.close()

    shap.summary_plot(shap_vals, X_val, feature_names=feats, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_beeswarm.png"))
    plt.close()

    return model, metrics, top10
