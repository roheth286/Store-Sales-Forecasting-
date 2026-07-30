import pandas as pd

def add_calendar_features(target_df):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = target_df["date"].dt.day_name()

    for d in days:
        target_df[f"day_{d}"] = (day_name == d).astype(int)

    day_of_month = target_df["date"].dt.day
    is_month_end = target_df["date"].dt.is_month_end

    target_df["is_payday_window"] = (
        day_of_month.isin([1, 2, 15, 16, 17]) | is_month_end | (day_of_month >= 30)
    ).astype(int)

    return target_df
