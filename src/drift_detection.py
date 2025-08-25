# src/drift_detection.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import logging

# --- Evidently imports with fallbacks (works on 0.7.11 and older layouts) ---
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    NEW_API = True
except ModuleNotFoundError:
    from evidently import Report  # type: ignore
    from evidently.presets import DataDriftPreset  # type: ignore
    NEW_API = False

TARGET_CANDIDATES = {"NO2_umol_m2", "target", "y", "label"}
NON_FEATURE_COLUMNS = {"date", "region", "id"}

def _pick_feature_columns(df: pd.DataFrame) -> List[str]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    drop = {"NO2_umol_m2", "target", "y", "label", "date", "region", "id"}
    feat_cols = [c for c in num_cols if c not in drop]
    return feat_cols or num_cols  # use all numeric features available

def _write_json_all_places(payload: Dict[str, Any]) -> None:
    txt = json.dumps(payload, indent=2, sort_keys=True)
    for p in [
        Path("reports/drift_report.json"),
        Path("/opt/airflow/reports/drift_report.json"),
        Path("/opt/airflow/data/reports/drift_report.json"),
    ]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)
        except Exception:
            pass

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)
    if pooled <= 0:
        return float("nan")
    return float(abs((np.mean(b) - np.mean(a)) / np.sqrt(pooled)))

def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    ref = pd.read_csv(reference_data_path)
    cur = pd.read_csv(current_data_path)

    feat_cols = _pick_feature_columns(ref)
    if not feat_cols:
        raise ValueError("No numeric feature columns available for drift analysis.")
    for c in feat_cols:
        if c not in cur.columns:
            cur[c] = pd.NA

    # --- Try Evidently first (compute + HTML); if serialization fails, fall back ---
    rep_dict: Dict[str, Any] | None = None
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref[feat_cols], current_data=cur[feat_cols])

        # Save HTML (supported across versions)
        try:
            Path("/opt/airflow/reports").mkdir(parents=True, exist_ok=True)
            report.save_html("/opt/airflow/reports/drift_report.html")  # best-effort
        except Exception:
            pass

        # Get full JSON dict (handle API variations)
        try:
            rep_dict = json.loads(report.json())              # new API
        except Exception:
            try:
                tmp = Path("/opt/airflow/reports/_drift_full.json")
                report.save_json(str(tmp))                    # mid-era API
                rep_dict = json.loads(tmp.read_text())
            except Exception:
                rep_dict = getattr(report, "as_dict", lambda: None)()  # very old API

    except Exception:
        rep_dict = None

    # --- If we got Evidently output, extract dataset flag + per-column scores ---
    if isinstance(rep_dict, dict) and rep_dict.get("metrics"):
        metrics_list = rep_dict.get("metrics", [])
        drift_result = (metrics_list[0] or {}).get("result", {}) if metrics_list else {}

        drift_detected = bool(
            drift_result.get("dataset_drift",
                             drift_result.get("drift_share", 0) > 0)
        )
        drift_by_cols = drift_result.get("drift_by_columns") or {}
        feature_drifts = {
            col: float(info["drift_score"])
            for col, info in drift_by_cols.items()
            if isinstance(info, dict) and info.get("drift_score") is not None
        }
        # keep only chosen feat_cols if Evidently returned more
        feature_drifts = {c: v for c, v in feature_drifts.items() if c in feat_cols}
        overall = float(np.mean(list(feature_drifts.values()))) if feature_drifts else float("nan")

        # Logging Block
        if drift_detected:
            logging.info("⚠️ Data drift detected in test set! Model retraining required.")
        else:
            logging.info("✅ No drift detected in test set.")

        out = {
            "drift_detected": drift_detected,
            "feature_drifts": feature_drifts,
            "overall_drift_score": overall,
            "source": "evidently",
        }
        _write_json_all_places(out)
        return out

    # --- Fallback: simple Cohen's d proxy so the pipeline keeps moving ---
    feature_drifts = {
        col: _cohens_d(ref[col].to_numpy(dtype=float), cur[col].to_numpy(dtype=float))
        for col in feat_cols
    }
    vals = [v for v in feature_drifts.values() if not np.isnan(v)]
    overall = float(np.mean(vals)) if vals else float("nan")
    drift_detected = any(v >= 0.5 for v in vals)  # medium effect size heuristic

    out = {
        "drift_detected": drift_detected,
        "feature_drifts": feature_drifts,
        "overall_drift_score": overall,
        "source": "fallback_cohens_d",
    }
    _write_json_all_places(out)
    return out
