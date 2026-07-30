import pandas as pd

def add_hierarchical_features(df):
    mean_family = df.groupby("family")["sales"].transform("mean")
    mean_store = df.groupby("store_nbr")["sales"].transform("mean")
    mean_cluster = df.groupby("cluster")["sales"].transform("mean")
    mean_type = df.groupby("type")["sales"].transform("mean")

    df["mean_sales_by_family"] = mean_family
    df["mean_sales_by_store"] = mean_store
    df["mean_sales_by_cluster"] = mean_cluster
    df["mean_sales_by_type"] = mean_type

    return df
