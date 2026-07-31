import numpy as np
import pandas as pd
from src.features.calendar_features import add_calendar_features
from src.features.sales_features import add_sales_features
from src.features.trans_features import add_transaction_features
from src.features.promo_features import add_promo_features
from src.features.hierarchical_features import add_hierarchical_features


def build_all_features(df, test_df=None):
    df = add_calendar_features(df)

    if test_df is None:
        df = add_sales_features(df)
        df = add_transaction_features(df)
        df = add_promo_features(df)
        df = add_hierarchical_features(df)
        return df

    # Dual processing for df and test_df matching main.ipynb Cells 61-87
    test_df = add_calendar_features(test_df)
    df, test_df = add_hierarchical_features(df, test_df)
    df, test_df = add_promo_features(df, test_df)

    # Set target sales and missing transactions on test_df to NaN & align columns 100% with df
    test_df["sales"] = np.nan
    test_df["transactions"] = np.nan
    test_df = test_df[df.columns]

    # Concatenate into full_df to calculate lags seamlessly across boundary
    full_df = pd.concat([df, test_df], ignore_index=True)
    full_df = full_df.sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)

    full_df = add_sales_features(full_df)
    full_df = add_transaction_features(full_df)

    # Extract updated df and test_df
    test_df = full_df[full_df["sales"].isnull()].copy().reset_index(drop=True)
    df = full_df[full_df["sales"].notnull()].copy().reset_index(drop=True)

    # Drop initial 28 days of 2013 where sales_lag_28 is NaN
    df = df.dropna(subset=["sales_lag_28"]).reset_index(drop=True)

    return df, test_df
