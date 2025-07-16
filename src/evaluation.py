import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict[str, float]:
    """
    Compute MAE, RMSE, and SMAPE.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)**0.5  # RMSE is sqrt of MSE
    smape = np.mean(
        np.abs(y_true - y_pred)
        / ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-3)
    )
    return {"MAE": mae, "RMSE": rmse, "SMAPE": smape * 100}  # SMAPE as percentage

def save_metrics(
    metrics: dict[str, float],
    out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
