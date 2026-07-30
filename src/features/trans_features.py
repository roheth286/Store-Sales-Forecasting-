import pandas as pd

def add_transaction_features(target_df):
    daily_store = target_df.groupby(["date", "store_nbr"]).agg(
        total_sales=("sales", "sum"),
        transactions=("transactions", "first")
    ).reset_index()

    daily_store = daily_store.sort_values(by=["store_nbr", "date"]).reset_index(drop=True)

    daily_store["trans_lag_1"] = daily_store.groupby("store_nbr")["transactions"].shift(1).fillna(0)
    daily_store["trans_lag_7"] = daily_store.groupby("store_nbr")["transactions"].shift(7).fillna(0)

    daily_store["trans_roll_mean_7"] = daily_store.groupby("store_nbr")["transactions"].transform(
        lambda x: x.rolling(window=7, center=False, min_periods=1).mean()
    )
    daily_store["trans_roll_mean_30"] = daily_store.groupby("store_nbr")["transactions"].transform(
        lambda x: x.rolling(window=30, center=False, min_periods=1).mean()
    )

    trans_feature_cols = ["trans_lag_1", "trans_lag_7", "trans_roll_mean_7", "trans_roll_mean_30"]
    for col in trans_feature_cols:
        if col in target_df.columns:
            target_df.drop(columns=[col], inplace=True)

    target_df = target_df.merge(
        daily_store[["date", "store_nbr"] + trans_feature_cols],
        on=["date", "store_nbr"],
        how="left"
    )
    return target_df
