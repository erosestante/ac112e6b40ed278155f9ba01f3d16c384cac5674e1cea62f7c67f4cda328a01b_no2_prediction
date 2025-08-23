from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict[str, float]:
    """
    Return exactly two metrics:
    - mae
    - smape (percentage)
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    smape = float(np.mean(
        np.abs(y_true - y_pred) /
        ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-3)
    ) * 100.0)
    return {"mae": mae, "smape": smape}

def save_metrics(
    metrics: dict[str, float],
    out_path: Path | str = "reports/evaluation_results.json"
) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
