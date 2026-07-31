import pandas as pd

def add_hierarchical_features(df, test_df=None):
    mean_family = df.groupby("family")["sales"].mean().to_dict()
    mean_store = df.groupby("store_nbr")["sales"].mean().to_dict()
    mean_cluster = df.groupby("cluster")["sales"].mean().to_dict()
    mean_type = df.groupby("type")["sales"].mean().to_dict()

    df["mean_sales_by_family"] = df["family"].map(mean_family)
    df["mean_sales_by_store"] = df["store_nbr"].map(mean_store)
    df["mean_sales_by_cluster"] = df["cluster"].map(mean_cluster)
    df["mean_sales_by_type"] = df["type"].map(mean_type)

    if test_df is not None:
        test_df["mean_sales_by_family"] = test_df["family"].map(mean_family)
        test_df["mean_sales_by_store"] = test_df["store_nbr"].map(mean_store)
        test_df["mean_sales_by_cluster"] = test_df["cluster"].map(mean_cluster)
        test_df["mean_sales_by_type"] = test_df["type"].map(mean_type)
        return df, test_df

    return df
