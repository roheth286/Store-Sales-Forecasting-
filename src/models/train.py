"""
Model training module matching Cell 91 (LightGBM) and Cell 92 (XGBoost) of main.ipynb.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error


def train_lgbm(df, features=None, val_end_date="2017-08-15"):
    """
    Trains baseline LightGBM regressor on 16-day holdout validation split.
    Matches Cell 91 of main.ipynb exactly.
    """
    df = df.copy()

    # 1. Convert string categorical columns to 'category' dtype for LightGBM
    cat_cols = ["family", "city", "state", "type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 2. Target Log Transformation: y = ln(sales + 1)
    if "sales_log" not in df.columns:
        df["sales_log"] = np.log1p(df["sales"])

    # 3. Define feature columns if not explicitly passed
    if features is None:
        features = [col for col in df.columns if col not in ["id", "date", "sales", "sales_log"]]

    # 4. Perform 16-Day Time-Based Holdout Split
    train_mask = df["date"] <= "2017-07-31"
    val_mask = (df["date"] >= "2017-08-01") & (df["date"] <= val_end_date)

    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, "sales_log"]
    X_val, y_val = df.loc[val_mask, features], df.loc[val_mask, "sales_log"]

    print(f"X_train shape: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"X_val shape:   {X_val.shape[0]:,} rows × {X_val.shape[1]} features")

    # 5. Initialize & Train Baseline LightGBM Regressor
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True)]
    )

    # 6. Evaluate Baseline Validation RMSLE Score
    val_preds_log = model.predict(X_val)
    val_preds_log = np.clip(val_preds_log, 0, None)
    val_rmsle = root_mean_squared_error(y_val, val_preds_log)

    print("\n" + "=" * 50)
    print(f"Baseline LightGBM Validation RMSLE Score: {val_rmsle:.5f}")
    print("=" * 50)

    return model, val_rmsle


def train_xgb(df, features=None, val_end_date="2017-08-15"):
    """
    Trains baseline XGBoost regressor on 16-day holdout validation split.
    Matches Cell 92 of main.ipynb exactly.
    """
    df = df.copy()

    # 1. Convert string categorical columns to 'category' dtype for XGBoost
    cat_cols = ["family", "city", "state", "type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 2. Target Log Transformation: y = ln(sales + 1)
    if "sales_log" not in df.columns:
        df["sales_log"] = np.log1p(df["sales"])

    # 3. Define feature columns if not explicitly passed
    if features is None:
        features = [col for col in df.columns if col not in ["id", "date", "sales", "sales_log"]]

    # 4. Perform 16-Day Time-Based Holdout Split
    train_mask = df["date"] <= "2017-07-31"
    val_mask = (df["date"] >= "2017-08-01") & (df["date"] <= val_end_date)

    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, "sales_log"]
    X_val, y_val = df.loc[val_mask, features], df.loc[val_mask, "sales_log"]

    print(f"X_train shape: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"X_val shape:   {X_val.shape[0]:,} rows × {X_val.shape[1]} features")

    # 5. Initialize & Train XGBoost Regressor
    model_xgb = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=8,
        tree_method="hist",          # Fast histogram-based tree building
        enable_categorical=True,     # Native categorical support
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # 6. Evaluate Validation RMSLE Score
    val_preds_xgb = model_xgb.predict(X_val)
    val_preds_xgb = np.clip(val_preds_xgb, 0, None)
    xgb_rmsle = root_mean_squared_error(y_val, val_preds_xgb)

    print("\n" + "=" * 50)
    print(f"XGBoost Validation RMSLE Score: {xgb_rmsle:.5f}")
    print("=" * 50)

    return model_xgb, xgb_rmsle
