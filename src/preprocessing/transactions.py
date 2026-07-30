import pandas as pd
import numpy as np

def process_transactions(df, transactions_df):
    if "transactions" in df.columns:
        df.drop(columns=["transactions"], inplace=True)

    daily_sales = df.groupby(["date", "store_nbr"])["sales"].sum().reset_index()
    daily_sales.rename(columns={"sales": "total_sales"}, inplace=True)

    daily_sales = daily_sales.merge(transactions_df, on=["date", "store_nbr"], how="left")

    store_models = {}
    for store_nbr in daily_sales["store_nbr"].unique():
        store_data = daily_sales[
            (daily_sales["store_nbr"] == store_nbr) & 
            (daily_sales["transactions"].notnull()) & 
            (daily_sales["total_sales"] > 0)
        ]
        x = store_data["total_sales"].values
        y = store_data["transactions"].values
        m, c = np.polyfit(x, y, 1)
        store_models[store_nbr] = (m, c)

    daily_sales.loc[daily_sales["total_sales"] == 0, "transactions"] = 0

    nan_rows = daily_sales["transactions"].isnull()
    predicted_vals = []

    for idx, row in daily_sales[nan_rows].iterrows():
        store_nbr = row["store_nbr"]
        total_sales = row["total_sales"]
        m, c = store_models[store_nbr]
        pred_val = int(round(max(0, m * total_sales + c)))
        predicted_vals.append(pred_val)

    daily_sales.loc[nan_rows, "transactions"] = predicted_vals
    daily_sales["transactions"] = daily_sales["transactions"].astype(int)

    df = df.merge(daily_sales[["date", "store_nbr", "transactions"]], on=["date", "store_nbr"], how="left")
    return df, daily_sales
