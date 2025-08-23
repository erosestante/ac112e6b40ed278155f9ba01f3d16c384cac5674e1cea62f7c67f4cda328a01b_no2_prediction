##feature_engineering.py

import pandas as pd
from pathlib import Path

def add_lag_feature(
    df: pd.DataFrame,
    group_col: str = "region",
    target_col: str = "NO2_umol_m2",
    lag: int = 1,
    new_col: str = "NO2_prev_month"
) -> pd.DataFrame:
    df = df.copy()
    df[new_col] = df.groupby(group_col)[target_col].shift(lag)
    return df

def add_time_features(
    df: pd.DataFrame,
    date_col: str = "date"
) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[date_col])
    df["month_num"] = ts.dt.month
    return df

def load_huc_list(
    path: Path,
    region_col: str = "adm3_en"
) -> list[str]:
    huc_df = pd.read_csv(path)
    return (
        huc_df[region_col]
        .dropna()
        .str.lower()
        .str.strip()
        .unique()
        .tolist()
    )

def add_huc_indicator(
    df: pd.DataFrame,
    huc_list: list[str],
    region_col: str = "region",
    new_col: str = "huc"
) -> pd.DataFrame:
    df = df.copy()
    df[new_col] = df[region_col].str.lower().isin(huc_list).astype(int)
    return df

def engineer_features(
    df: pd.DataFrame,
    huc_csv_path: Path
) -> pd.DataFrame:
    """
    Sort by region+date, add lag, time, and HUC indicator features,
    then drop rows where lag is NaN.
    """
    df = df.sort_values(["region", "date"]).reset_index(drop=True)
    df = add_lag_feature(df)
    df = add_time_features(df)
    hucs = load_huc_list(huc_csv_path)
    df = add_huc_indicator(df, hucs)
    return df.dropna(subset=["NO2_prev_month"]).reset_index(drop=True)
