# NO₂ Prediction with Urban Amenities and Satellite Data

## 🧠 Project Overview
This project focuses on the Philippines, aiming to develop amenity-aware, interpretable machine-learning models that integrate OpenStreetMap amenity counts and satellite-derived air quality observations to forecast regional NO₂ concentrations, across all Philippine provinces and highly urbanized cities.

By fusing spatial amenity information with time-series air quality data, we will generate robust, period-specific forecasts and apply SHAP (SHapley Additive exPlanations) to decompose model outputs and track how the importance of amenity features shifts across three distinct phases: pre-pandemic, pandemic (lockdown), and post-pandemic (reopening).

Our deliverables include:

- Predictive Air Quality Models: Time-aware regression frameworks capable of capturing non-linear relationships between amenities and pollutant levels.

- Interpretability Reports: SHAP-driven analyses revealing evolving drivers of air pollution over time.

This work will equip policymakers and stakeholders with actionable, hyper-local air quality forecasts and transparent insights into shifting pollution drivers, enabling targeted interventions across the Philippines.

---
## 📥 How to Get the Data

### 1. Satellite NO₂ Data (Sentinel‑5P + GEE)
- **Source:** COPERNICUS/S5P/OFFL/L3_NO2 via Google Earth Engine
- **Temporal Coverage:** Jan 2018 – May 2025 (monthly composites)
- **Spatial Aggregation:** Reduced by GADM Level‑2 boundaries for 37 HUCs and 82 provinces
- **Output:** CSV files in `data/raw/`:
  - `PHL_HUCs_Monthly_NO2_2018_2025.csv`
  - `PHL_Provinces_Monthly_NO2_2018_2025.csv`

### 2. Amenity Counts Data (OpenStreetMap)
- **Snapshots:** Quarterly (Jan/Apr/Jul/Oct) from 2018 Q1 to 2025 Q1
- **Tags Extracted:** `building`, `amenity`, `leisure`, `public_transport`, `office`, `shop`, `tourism`
- **Processing Pipeline:**
  1. **Geocoding**: PSGC reference → OSMnx geocode → province GeoJSON → cached in S3.
  2. **Feature Extraction**: Overpass API via OSMnx → raw GeoJSON per date/region/province.
  3. **Spark Transformation**:
     - **`process_province()`** takes a GeoDataFrame and a mapping dict to:
       - Select relevant columns, explode multi‐valued tags
       - Map raw OSM values (case‐insensitive) into ~50 clean categories (unknown → `uncat__OTHER`)
       - Pivot to wide format (one column per category count)
       - Add `gadm` (province), `date` (quarter) metadata
     - Writes one Parquet file per quarter: `data/raw/Archive/features_quarter=YYYY-MM-DD/features.parquet`

---
## ⚙️ Setup Instructions
We manage environments with [UV](https://docs.astral.sh/uv/):
```bash
# 1. Create & activate
uv venv .venv
source .venv/bin/activate        # (.venv/Scripts/activate on Windows)
# 2. Install & lock
uv pip install -r pyproject.toml
uv pip compile pyproject.toml     # generates uv.lock
```
*Optional:* snapshot with `uv pip freeze > requirements.txt`.

---
## 🚀 Running the Pipeline
```bash
python src/run_pipeline.py
```
- **Outputs:**
  - Models in `models/`
  - Metrics & SHAP plots in `reports/pre-pandemic`, `reports/during-pandemic`, `reports/post-pandemic`

---
## 📊 Running Visualizations
`evaluation.py` generates:
- SHAP summary plots
- Feature‐importance bar charts

---
## 🗂 Folder Structure
```text
.
├── data/
│   └── raw/
│       ├── PHL_HUCs_Monthly_NO2_2018_2025.csv
│       ├── PHL_Provinces_Monthly_NO2_2018_2025.csv
│       └── Archive/
│           └── features_quarter=YYYY-MM-DD/  # Contains `_SUCCESS`, `_SUCCESS.crc`, and `part-*.snappy.parquet` files
├── models/                  # Trained XGBoost .pkl
├── reports/                 # Period-specific metrics & plots
├── src/                     # ML scripts: preprocessing → engineering → training → evaluation → orchestrator
├── .pre-commit-config.yaml  # Pre-commit hooks
├── pyproject.toml           # Project metadata & dependencies
├── uv.lock                  # Locked dependencies
├── requirements.txt         # Optional pip snapshot
└── README.md                # This documentation
```

---
## 🔧 Pre-commit Configuration
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.5
    hooks:
      - id: ruff
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
```
Installed via:
```bash
uv pip install pre-commit
pre-commit install
```
All code passes linting and formatting checks.

---
## 💡 Reflection
This is my first time creating a GitHub repository with a fully functioning ML Ops pipeline, so setting up the UV environment and modularizing the code (data preprocessing, feature engineering, model training, evaluation) was a significant challenge. I learned the importance of validating each module independently—ensuring data preprocessing works, then feature engineering, and finally model training—before running the full pipeline. Additionally, I encountered compatibility issues when running the pipeline in a OneDrive-synced directory and resolved them by moving the project to a local, unsynced Documents folder.

---
*This README satisfies course requirements for a production-driven ML project.*
