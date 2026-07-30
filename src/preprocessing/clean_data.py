import pandas as pd

def filter_pre_opening_days(df):
    daily_sales = df.groupby(["date", "store_nbr"])["sales"].sum().reset_index()
    first_active = daily_sales[daily_sales["sales"] > 0].groupby("store_nbr")["date"].min().reset_index()
    first_active.rename(columns={"date": "first_active_date"}, inplace=True)

    df = df.merge(first_active, on="store_nbr", how="left")
    df = df[df["date"] >= df["first_active_date"]].copy()
    df.drop(columns=["first_active_date"], inplace=True)
    return df
