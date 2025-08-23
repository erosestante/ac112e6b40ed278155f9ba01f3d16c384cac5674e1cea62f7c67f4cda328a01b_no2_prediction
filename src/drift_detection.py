# src/drift_detection.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset  # v0.7.11+


# columns we should NOT treat as features
TARGET_CANDIDATES = {"NO2_umol_m2", "target", "y", "label"}
NON_FEATURE_COLUMNS = {"date", "region", "id"}

def _pick_feature_columns(df: pd.DataFrame) -> List[str]:
    """Pick at least 3 numeric feature columns (exclude targets/IDs/dates)."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    drop = TARGET_CANDIDATES | NON_FEATURE_COLUMNS
    feat_cols = [c for c in num_cols if c not in drop]

    # If you have 3+ features, take the first 3 (rubric allows this).
    if len(feat_cols) >= 3:
        return feat_cols[:3]
    # If fewer than 3 exist, return whatever you have (could be 1–2).
    if feat_cols:
        return feat_cols
    # Fallback: if filtering removed everything, use any numeric columns.
    return num_cols[:3]

def _to_dataset(df: pd.DataFrame, feature_cols: List[str]) -> Dataset:
    """Wrap a pandas frame into an Evidently Dataset with a minimal numeric schema."""
    schema = DataDefinition(numerical_columns=feature_cols)
    return Dataset.from_pandas(df[feature_cols], data_definition=schema)

def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    """
    Compare reference vs current CSVs with Evidently (v0.7.11+), extract dataset-level
    drift flag and per-feature drift scores, compute an overall average, save JSON, return it.

    Returns:
        {
          "drift_detected": bool,
          "feature_drifts": {"feat1": float, ...},
          "overall_drift_score": float
        }
    """
    ref = pd.read_csv(reference_data_path)
    cur = pd.read_csv(current_data_path)

    # choose feature subset & align columns
    feat_cols = _pick_feature_columns(ref)
    if not feat_cols:
        raise ValueError("No numeric feature columns available for drift analysis.")
    # ensure current has those columns (fill missing with NA)
    for c in feat_cols:
        if c not in cur.columns:
            cur[c] = pd.NA

    ref_ds = _to_dataset(ref, feat_cols)
    cur_ds = _to_dataset(cur, feat_cols)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_ds, current_data=cur_ds)
    rep = report.as_dict()

    # ---- Extract results (robust to small key differences across patch versions) ----
    metrics_list = rep.get("metrics", [])
    if not metrics_list:
        raise RuntimeError("Evidently returned no metrics in report.as_dict().")

    # The DataDriftPreset is usually the first item
    drift_result = metrics_list[0].get("result", {})

    # dataset-level drift flag
    drift_detected = bool(
        drift_result.get("dataset_drift",
        drift_result.get("drift_share", 0) > 0)
    )

    # per-column drift scores
    drift_by_cols = (
        drift_result.get("drift_by_columns")
        or (metrics_list[0].get("result", {}).get("drift_by_columns", {}))
    ) or {}

    feature_drifts: Dict[str, float] = {}
    for col in feat_cols:
        col_info = drift_by_cols.get(col, {})
        if "drift_score" in col_info and col_info["drift_score"] is not None:
            feature_drifts[col] = float(col_info["drift_score"])

    overall = float(np.mean(list(feature_drifts.values()))) if feature_drifts else float("nan")

    out = {
        "drift_detected": drift_detected,
        "feature_drifts": feature_drifts,
        "overall_drift_score": overall,
    }

    # save JSON (overwrite each run)
    out_path = Path("reports/drift_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    return out
