import pandas as pd
import numpy as np
import bisect

def process_oil(target_df, oil_df, start_date="2013-01-01", end_date="2017-08-31"):
    known_oil = oil_df.dropna(subset=["dcoilwtico"]).sort_values("date").reset_index(drop=True)
    known_dates = known_oil["date"].tolist()
    known_prices = known_oil["dcoilwtico"].tolist()
    known_dict = dict(zip(known_dates, known_prices))

    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    date_to_price = {}

    for d in all_dates:
        if d in known_dict:
            date_to_price[d] = known_dict[d]
        else:
            idx = bisect.bisect_right(known_dates, d)
            slice_before = known_prices[max(0, idx - 10) : idx]
            slice_after = known_prices[idx : min(len(known_prices), idx + 10)]
            date_to_price[d] = np.mean(slice_before + slice_after)

    full_oil_df = pd.DataFrame({"date": all_dates})
    full_oil_df["oil_price"] = full_oil_df["date"].map(date_to_price)

    full_oil_df["oil_roll_mean_30"] = full_oil_df["oil_price"].rolling(window=30, center=True, min_periods=1).mean()
    full_oil_df["oil_diff_7"] = full_oil_df["oil_price"].diff(7).fillna(0)

    target_df["oil_price"] = target_df["date"].map(dict(zip(full_oil_df["date"], full_oil_df["oil_price"])))
    target_df["oil_roll_mean_30"] = target_df["date"].map(dict(zip(full_oil_df["date"], full_oil_df["oil_roll_mean_30"])))
    target_df["oil_diff_7"] = target_df["date"].map(dict(zip(full_oil_df["date"], full_oil_df["oil_diff_7"])))

    return target_df, full_oil_df
