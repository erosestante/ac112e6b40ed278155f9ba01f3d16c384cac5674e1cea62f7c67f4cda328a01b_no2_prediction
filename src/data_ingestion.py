# src/data_ingestion.py
from pathlib import Path
#import shutil

def ensure_datasets(
    raw_dir: Path = Path("data/raw"),
    city_csv: Path = Path("data/raw/PHL_HUCs_Monthly_NO2_2018_2025.csv"),
    prov_csv: Path = Path("data/raw/PHL_Provinces_Monthly_NO2_2018_2025.csv"),
    parquet_dir: Path = Path("data/raw/Archive")
) -> tuple[Path, Path, Path]:
    """
    Ensure the dataset files exist in data/raw and return their Paths.

    Returns:
        (CITY_PATH, PROV_PATH, PARQ_DIR)
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not city_csv.exists():
        raise FileNotFoundError(f"City CSV not found at {city_csv}")
    if not prov_csv.exists():
        raise FileNotFoundError(f"Province CSV not found at {prov_csv}")
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet dir not found at {parquet_dir}")

    return city_csv, prov_csv, parquet_dir
