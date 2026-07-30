import pandas as pd

def add_sales_features(df):
    df = df.sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)

    target_lags = [1, 7, 14, 21, 28]
    for lag in target_lags:
        df[f"sales_lag_{lag}"] = df.groupby(["store_nbr", "family"])["sales"].shift(lag)

    df["sales_roll_mean_30"] = df.groupby(["store_nbr", "family"])["sales"].transform(
        lambda x: x.rolling(window=30, center=False, min_periods=1).mean()
    )

    df["sales_roll_std_7"] = df.groupby(["store_nbr", "family"])["sales"].transform(
        lambda x: x.rolling(window=7, center=False, min_periods=1).std()
    ).fillna(0)

    return df
