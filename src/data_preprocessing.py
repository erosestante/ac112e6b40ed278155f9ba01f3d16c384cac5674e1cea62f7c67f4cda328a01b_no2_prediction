##data_preprocessing.py

import os
import re
import glob
import pandas as pd
from pathlib import Path
from typing import List, Optional
import numpy as np

def load_csv(path: Path, region_col: str, date_col: str = "date") -> pd.DataFrame:
    """
    Load a CSV, rename its region column to "region", and parse the date.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={region_col: "region"})
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def combine_datasets(city: pd.DataFrame, prov: pd.DataFrame, raw_col: str = "NO2") -> pd.DataFrame:
    """
    Stack city & province NO₂, convert to µmol/m².
    """
    df = pd.concat([
        city[["region", "date", raw_col]],
        prov[["region", "date", raw_col]],
    ], ignore_index=True)
    df = df.rename(columns={raw_col: "NO2_umol_m2"})
    # assume original units were mol/m²
    df["NO2_umol_m2"] *= 1e6
    return df

def get_quarter_start(date: pd.Timestamp) -> pd.Timestamp:
    q = ((date.month - 1) // 3) * 3 + 1
    return pd.Timestamp(year=date.year, month=q, day=1)

def add_quarter_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df["quarter_date"] = df[date_col].apply(get_quarter_start)
    return df

def load_external_features(PARQUET_DIR, region_col="region") -> pd.DataFrame:
    # Find all parquet files recursively
    all_parquet_files = glob.glob(os.path.join(PARQUET_DIR, '**/*.parquet'), recursive=True)

    dfs = []
    for file in all_parquet_files:
        # Extract date from file path
        match = re.search(r"features_quarter=(\d{4}-\d{2}-\d{2})", file)
        if match:
            quarter_date = pd.to_datetime(match.group(1))
        else:
            quarter_date = pd.NaT

        # Read parquet file
        df = pd.read_parquet(file)
        df["quarter_date"] = quarter_date
        dfs.append(df)

    # Combine all files into one DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df.rename(columns={"gadm": region_col})



def merge_features(
    main_df: pd.DataFrame,
    feats_df: pd.DataFrame,
    drop_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Left‐merge external features, drop rows where all external features are NaN,
    then fill remaining NaNs with 0.
    """
    drop_cols = drop_cols or ["\tUncategorized", "uncat__OTHER", "date"]
    feats = feats_df.drop(columns=[c for c in drop_cols if c in feats_df], errors="ignore")

    # normalize keys
    for df in (main_df, feats):
        df["region"] = df["region"].str.strip().str.lower()
        df["quarter_date"] = pd.to_datetime(df["quarter_date"])

    merged = main_df.merge(feats, on=["region", "quarter_date"], how="left")
    ext_cols = [c for c in feats.columns if c not in ("region", "quarter_date")]
    merged = merged.loc[~merged[ext_cols].isna().all(axis=1)].fillna(0)
    return merged

def load_and_prepare_all(
    city_path: Path,
    prov_path: Path,
    parquet_dir: Path,
    city_region_col: str = "adm3_en",
    prov_region_col: str = "adm2_en"
) -> pd.DataFrame:
    city = load_csv(city_path, region_col=city_region_col)
    prov = load_csv(prov_path, region_col=prov_region_col)
    df = combine_datasets(city, prov)
    df = add_quarter_date(df)
    ext = load_external_features(parquet_dir)
    df = merge_features(df, ext)
    # drop intermediate columns
    return df.drop(columns=["NO2", "quarter_date"], errors="ignore")


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "NO2_umol_m2",
    date_col: str = "date",
    save_dir: Path = Path("data"),
    frac_train: float = 0.8,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # ---- time-based split ----
    dates = df[date_col].sort_values().unique()
    split = int(len(dates) * frac_train)
    train = df[df[date_col].isin(dates[:split])].reset_index(drop=True)
    test  = df[df[date_col].isin(dates[split:])].reset_index(drop=True)

    # ---- features/targets ----
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test  = test.drop(columns=[target_col])
    y_test  = test[target_col]

    # numeric feature cols (exclude obvious non-features)
    non_features = {target_col, date_col, "region"}
    num_cols = [c for c in X_train.columns
                if c not in non_features and np.issubdtype(X_train[c].dtype, np.number)]

    # copies for drifted versions
    X_train_drifted = X_train.copy()
    y_train_drifted = y_train.copy()
    X_test_drifted  = X_test.copy()
    y_test_drifted  = y_test.copy()

    # ---- numeric drift only ----
    if num_cols:
        stds = X_train[num_cols].astype(float).std(ddof=0)
        # handle zero/NaN std gracefully
        min_pos = stds[stds > 0].min()
        stds = stds.replace(0, min_pos if pd.notna(min_pos) else 1.0).fillna(1.0)

        # mean shift + noise (make it obvious)
        shift = 0.5 * stds.values                # +0.5*std shift
        noise = rng.normal(0.0, 0.3 * stds.values,
                           size=(len(X_test_drifted), len(num_cols)))  # 0.3*std noise

        X_test_drifted.loc[:, num_cols] = (
            X_test_drifted.loc[:, num_cols].astype(float).values + shift + noise
        )

        # amplify drift on the top-variance 5 numeric features
        topk = X_train[num_cols].var().sort_values(ascending=False).index[:5]
        X_test_drifted.loc[:, topk] = X_test_drifted.loc[:, topk].astype(float) * 1.5

    # ---- save outputs ----
    save_dir.mkdir(parents=True, exist_ok=True)
    (X_train.assign(**{target_col: y_train})).to_csv(save_dir / "train.csv", index=False)
    (X_test.assign(**{target_col: y_test})).to_csv(save_dir / "test.csv", index=False)
    (X_train_drifted.assign(**{target_col: y_train_drifted})).to_csv(save_dir / "drifted_train.csv", index=False)
    (X_test_drifted.assign(**{target_col: y_test_drifted})).to_csv(save_dir / "drifted_test.csv", index=False)

    return (
        X_train, X_test, y_train, y_test,
        X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted
    )
