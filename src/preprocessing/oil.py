import pandas as pd

def process_oil(target_df, oil_df, start_date="2013-01-01", end_date="2017-08-31"):
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    full_oil_df = pd.DataFrame({"date": all_dates})

    full_oil_df = full_oil_df.merge(oil_df, on="date", how="left")
    full_oil_df["oil_price"] = full_oil_df["dcoilwtico"].interpolate(method="linear").bfill().ffill()

    full_oil_df["oil_roll_mean_30"] = full_oil_df["oil_price"].rolling(window=30, center=True, min_periods=1).mean()
    full_oil_df["oil_diff_7"] = full_oil_df["oil_price"].diff(7).fillna(0)

    target_df["oil_price"] = target_df["date"].map(dict(zip(full_oil_df["date"], full_oil_df["oil_price"])))
    target_df["oil_roll_mean_30"] = target_df["date"].map(dict(zip(full_oil_df["date"], full_oil_df["oil_roll_mean_30"])))
    target_df["oil_diff_7"] = target_df["date"].map(dict(zip(full_oil_df["date"], full_oil_df["oil_diff_7"])))

    return target_df, full_oil_df
