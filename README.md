# Store Sales Forecasting - Corporación Favorita (Ecuador)

An end-to-end production data science pipeline and machine learning project for forecasting store sales across **Corporación Favorita**, one of Ecuador's largest supermarket chains.

---

## 📌 Project Overview

This repository provides a complete time-series forecasting solution for predicting daily unit sales across **54 store locations** and **33 product families** in Ecuador. 

The project incorporates macro-economic factors (20-point bisect interpolated Ecuadorian crude oil prices), store-level foot traffic (customer transactions via 54 store linear regression models), public holiday events across three geographical tiers (National, Regional, and Local), calendar seasonality, promotional dynamics, and hierarchical target encodings.

### 🏆 Modeling & Machine Learning Approach
- **Evaluation Metric:** Root Mean Squared Logarithmic Error (RMSLE).
- **Target Transformation:** $y = \ln(\text{sales} + 1)$ (`np.log1p`) to align standard RMSE loss with Kaggle's official RMSLE metric.
- **Model Architecture:** Baseline evaluation compares LightGBM (`LGBMRegressor`) against XGBoost (`XGBRegressor` with `tree_method='hist'` and native categorical support).
- **Validation Benchmark:** Evaluated on a 16-day holdout validation split (August 1 to August 15, 2017). Baseline XGBoost achieved a top-tier validation score of **`0.35823` RMSLE**.
- **Multi-Step Test Forecasting:** Implements a **16-Step Direct-Recursive (DirRec)** forecasting strategy. For each horizon date $h \in [1 \dots 16]$ (August 16–31, 2017), a dedicated model is trained for horizon $h$, predictions are generated for all 1,782 store-family pairs, and short-term lags (`sales_lag_1`, `7`, `14`, `trans_lag_1`, `7`) and rolling statistics are updated recursively day-by-day to populate `submission.csv`.

---

## 📊 Datasets (Ecuadorian Retail Data)

The raw data is stored inside `data/raw/` and consists of 7 key CSV files:
1. `train.csv`: Historical daily sales records from **2013-01-02 to 2017-08-15** (`id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion`).
2. `test.csv`: Target evaluation period from **2017-08-16 to 2017-08-31** (16 days).
3. `stores.csv`: Metadata for 54 stores (`city`, `state`, `type`, `cluster`).
4. `transactions.csv`: Daily customer transaction counts per store.
5. `oil.csv`: Daily Ecuadorian crude oil prices ($dcoilwtico$).
6. `holidays_events.csv`: Ecuadorian public holidays, events, and bridge days across National, Regional, and Local scopes.
7. `sample_submission.csv`: Kaggle competition submission format.

---

## 🚀 How to Run the Repository

### 1. Environment Setup
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/roheth286/Store-Sales-Forecasting-.git
cd Store-Sales-Forecasting-

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

### 2. Running the Codebase

You can explore the project through two approaches:

#### **Option A: Production Pipeline Approach (Recommended)**
Run the modular pipeline notebook located inside `notebooks/`:
```bash
notebooks/pipeline.ipynb
```
* Executing `notebooks/pipeline.ipynb` runs the modular code from `src/` end-to-end:
  1. Loads and merges raw datasets from `data/raw/`.
  2. Imputes missing oil prices (20-point bisect algorithm) and transaction records.
  3. Filters pre-opening zero-sales days across late-opening stores.
  4. Manufactures all **42 feature columns** across `df` and `test_df` (`full_df` concatenation) and saves `data/processed/train_processed.parquet`.
  5. Generates and saves all **9 high-resolution EDA plots** directly into `plots/`.
  6. Fits 54 store linear regression models for test transaction estimation.
  7. Trains LightGBM & XGBoost baseline models on the 16-day validation split.
  8. Executes 16-step DirRec forecasting to predict `test.csv` sales and outputs `data/processed/submission.csv`.

#### **Option B: Original Exploration Notebook**
If you wish to view the original cell-by-cell exploratory data analysis and initial experimentation:
```bash
main.ipynb
```

---

## 📁 Directory Structure

```text
Store_Sales_Forecasting/
├── .venv/                         # Virtual environment
├── data/
│   ├── raw/                       # 7 Raw CSV datasets
│   └── processed/                 # Prepared datasets & submission.csv
├── notebooks/
│   └── pipeline.ipynb             # Modular end-to-end pipeline notebook
├── plots/                         # 9 Generated EDA visualization plots (PNG format)
├── docs/
│   └── feature_engineering.md     # Detailed documentation of all 42 features & EDA rationale
├── src/                           # Production source code
│   ├── preprocessing/             # Loaders, holiday logic, 20-pt bisect oil & transaction imputation, cleaning
│   ├── features/                  # Calendar, sales lags, rolling windows, promo & hierarchical features
│   ├── visualization/             # Master plotting module for all 9 EDA charts
│   ├── models/                    # Model training (LGBM/XGB), 16-step DirRec predictor & evaluation
│   └── utils/                     # Submission CSV exporter
├── main.ipynb                     # Original exploration notebook
├── README.md                      # Project documentation
└── requirements.txt               # Project Python package dependencies
```

---

## 📑 Feature Documentation
For an in-depth breakdown of how all 42 feature columns were manufactured, the data cleaning steps, and the statistical clues discovered from our EDA plots, please see [docs/feature_engineering.md](docs/feature_engineering.md).
