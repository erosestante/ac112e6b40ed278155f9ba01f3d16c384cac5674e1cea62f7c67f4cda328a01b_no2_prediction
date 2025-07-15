import pandas as pd
from pathlib import Path

from data_preprocessing import load_and_prepare_all
from feature_engineering import engineer_features
from model_training import train_naive_baseline, train_xgb_with_optuna
from evaluation import save_metrics


def time_based_split(df: pd.DataFrame, date_col: str = "date", frac_train: float = 0.8):
    dates = pd.to_datetime(df[date_col]).sort_values().unique()
    split = int(len(dates) * frac_train)
    train_dates, val_dates = dates[:split], dates[split:]
    train = df[df[date_col].isin(train_dates)].copy()
    val   = df[df[date_col].isin(val_dates)].copy()
    return train, val


def main():
    print("🚀 Starting NO₂ prediction pipeline\n")

    # Paths
    CITY_PATH = Path("data/raw/PHL_HUCs_Monthly_NO2_2018_2025.csv")
    PROV_PATH = Path("data/raw/PHL_Provinces_Monthly_NO2_2018_2025.csv")
    PARQ_DIR  = Path("data/raw/Archive")
    HUC_CSV   = CITY_PATH  # HUC list is in the city file

    # 1) Load & preprocess
    print("1) Loading & preprocessing data...")
    df = load_and_prepare_all(CITY_PATH, PROV_PATH, PARQ_DIR)
    print(f"   ➤ Combined dataframe shape: {df.shape}")

    # 2) Feature engineering
    print("2) Engineering lag, time, and HUC features...")
    df = engineer_features(df, HUC_CSV)
    print(f"   ➤ After feature engineering: {df.shape}\n")

    # Clean up
    df = df.dropna(subset=["NO2_prev_month"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # 3) Define periods
    print("3) Splitting into pandemic periods...")
    periods = {
        "pre-pandemic":    df[df.date <  "2020-03-01"],
        "during-pandemic": df[(df.date >= "2020-03-01") & (df.date < "2022-01-01")],
        "post-pandemic":   df[df.date >= "2022-01-01"],
    }

    results = {}
    EXCLUDE = ["region", "date", "NO2_umol_m2", "NO2_prev_month"]

    for label, subset in periods.items():
        print(f"\n🔹 Period: {label}")
        train_df, val_df = time_based_split(subset, date_col="date")
        print(f"   • Train shape: {train_df.shape}, Val shape: {val_df.shape}")

        if val_df.empty:
            print("   ⚠️ No validation data; skipping.")
            continue

        # Naive baseline
        print("   • Naive baseline...")
        naive_metrics = train_naive_baseline(val_df) or {}
        print(f"      ↳ Naive metrics: {naive_metrics}")
        save_metrics(naive_metrics, Path(f"reports/{label}/naive_metrics.txt"))

        # XGBoost + Optuna + SHAP
        print("   • Training XGBoost with Optuna...")
        model, xgb_metrics, top10 = train_xgb_with_optuna(
            train_df, val_df, EXCLUDE,
            model_path=f"models/xgb_{label.replace(' ', '_')}.pkl",
            figures_dir=f"reports/{label}"
        )
        print(f"      ↳ XGBoost metrics: {xgb_metrics}")
        save_metrics(xgb_metrics, Path(f"reports/{label}/xgb_metrics.txt"))

        print("      ↳ Top 10 SHAP features:")
        for feat, imp in top10:
            print(f"         • {feat}: {imp:.4f}")

        results[label] = {"naive": naive_metrics, "xgb": xgb_metrics}

    # Summary
    print("\n✅ All periods processed. Summary:")
    summary = {
        lbl: {"naive_RMSE": v["naive"].get("RMSE"), "xgb_RMSE": v["xgb"]["RMSE"]}
        for lbl, v in results.items()
    }
    summary_df = pd.DataFrame(summary).T
    print(summary_df.to_string())

    print("\n🎉 Pipeline complete!")


if __name__ == "__main__":
    main()
