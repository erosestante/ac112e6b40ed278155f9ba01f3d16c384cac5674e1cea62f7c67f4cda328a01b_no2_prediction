import os
import re
import glob
import pandas as pd
from pathlib import Path

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

from typing import List, Optional

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
