"""
Test forecasting module using Direct-Recursive (DirRec) strategy matching Cell 93 of main.ipynb.
"""

import numpy as np
import pandas as pd
import xgboost as xgb


def dirrec_predict(df, test_df, store_models, features=None):
    df = df.copy().sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)
    test_df = test_df.copy().sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)

    cat_cols = ["family", "city", "state", "type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
        if col in test_df.columns:
            test_df[col] = test_df[col].astype("category")

    if "sales_log" not in df.columns:
        df["sales_log"] = np.log1p(df["sales"])

    if features is None:
        features = [col for col in df.columns if col not in ["id", "date", "sales", "sales_log"]]

    test_dates = sorted(test_df["date"].unique())

    # 2. Build per-(store, family) sales history from last 30 days of training
    train_end = df["date"].max()
    hist_start = train_end - pd.Timedelta(days=29)

    sales_hist = {}
    for (snbr, fam), grp in df[df["date"] >= hist_start].groupby(["store_nbr", "family"]):
        sales_hist[(snbr, str(fam))] = grp["sales"].tolist()

    # Build per-store transaction history
    daily_tr = df[df["date"] >= hist_start].groupby(["date", "store_nbr"])["transactions"].first().reset_index()
    daily_tr = daily_tr.sort_values(by=["store_nbr", "date"])
    trans_hist = {}
    for snbr, grp in daily_tr.groupby("store_nbr"):
        trans_hist[snbr] = grp["transactions"].tolist()

    # 3. DirRec 16-Step Loop
    for h, target_date in enumerate(test_dates, start=1):
        print(f"Horizon {h}/16: {target_date.strftime('%Y-%m-%d')}", end=" → ")

        # 3a. Create shifted target for horizon h
        y_shifted = df.groupby(["store_nbr", "family"])["sales_log"].shift(-h)
        valid = y_shifted.notnull()

        # 3b. Train XGBoost model for this horizon
        model_h = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=8,
            tree_method="hist", enable_categorical=True,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        model_h.fit(df.loc[valid, features], y_shifted[valid], verbose=False)

        # 3c. Predict this day
        day_mask = test_df["date"] == target_date
        preds_log = model_h.predict(test_df.loc[day_mask, features])
        preds_sales = np.clip(np.expm1(np.clip(preds_log, 0, None)), 0, None)
        test_df.loc[day_mask, "sales"] = preds_sales

        # 3d. Estimate transactions using 54 store linear regression models
        day_store_sales = test_df.loc[day_mask].groupby("store_nbr")["sales"].sum()
        pred_trans_map = {}
        for snbr in day_store_sales.index:
            m, c = store_models[snbr]
            est = int(round(max(0, m * day_store_sales[snbr] + c)))
            test_df.loc[day_mask & (test_df["store_nbr"] == snbr), "transactions"] = est
            pred_trans_map[snbr] = est

        # 3e. Update history buffers with today's predictions
        pred_sales_map = dict(zip(
            zip(test_df.loc[day_mask, "store_nbr"].values,
                test_df.loc[day_mask, "family"].astype(str).values),
            test_df.loc[day_mask, "sales"].values
        ))

        for (snbr, fam), sale in pred_sales_map.items():
            sales_hist.setdefault((snbr, fam), []).append(sale)

        for snbr, trans in pred_trans_map.items():
            trans_hist.setdefault(snbr, []).append(trans)

        # 3f. Update SALES LAG features for future test dates
        for lag_days, lag_col in [(1, "sales_lag_1"), (7, "sales_lag_7"), (14, "sales_lag_14")]:
            future_date = target_date + pd.Timedelta(days=lag_days)
            future_mask = test_df["date"] == future_date
            if future_mask.any():
                keys = list(zip(
                    test_df.loc[future_mask, "store_nbr"].values,
                    test_df.loc[future_mask, "family"].astype(str).values
                ))
                test_df.loc[future_mask, lag_col] = [pred_sales_map.get(k, np.nan) for k in keys]

        # 3g. Update TRANSACTION LAG features for future test dates
        for lag_days, lag_col in [(1, "trans_lag_1"), (7, "trans_lag_7")]:
            future_date = target_date + pd.Timedelta(days=lag_days)
            future_mask = test_df["date"] == future_date
            if future_mask.any():
                test_df.loc[future_mask, lag_col] = test_df.loc[future_mask, "store_nbr"].map(pred_trans_map)

        # 3h. Update ROLLING features for next day
        next_date = target_date + pd.Timedelta(days=1)
        next_mask = test_df["date"] == next_date
        if next_mask.any():
            # Sales rolling stats from history buffer
            roll_mean_30 = {k: np.mean(v[-30:]) for k, v in sales_hist.items()}
            roll_std_7 = {}
            for k, v in sales_hist.items():
                roll_std_7[k] = np.std(v[-7:], ddof=1) if len(v) >= 2 else 0

            next_keys = list(zip(
                test_df.loc[next_mask, "store_nbr"].values,
                test_df.loc[next_mask, "family"].astype(str).values
            ))
            test_df.loc[next_mask, "sales_roll_mean_30"] = [roll_mean_30.get(k, 0) for k in next_keys]
            test_df.loc[next_mask, "sales_roll_std_7"] = [roll_std_7.get(k, 0) for k in next_keys]

            # Transaction rolling mean from history buffer
            trans_roll_7 = {s: np.mean(v[-7:]) for s, v in trans_hist.items()}
            test_df.loc[next_mask, "trans_roll_mean_7"] = test_df.loc[next_mask, "store_nbr"].map(trans_roll_7)

        print(f"Done! Mean predicted sales: {preds_sales.mean():.2f}")

    print("\n" + "=" * 50)
    print("DirRec 16-Day Prediction Complete!")
    print(f"test_df shape: {test_df.shape}")
    print(f"Remaining NaN in sales: {test_df['sales'].isnull().sum()}")
    print(f"Remaining NaN in transactions: {test_df['transactions'].isnull().sum()}")
    print("=" * 50)

    return test_df
