import pandas as pd
import numpy as np

def add_promo_features(df, test_df=None):
    df = df.sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)

    df["onpromotion_lead_1"] = df.groupby(["store_nbr", "family"])["onpromotion"].shift(-1).fillna(0)

    family_stats = df.groupby(["family", df["onpromotion"] > 0])["sales"].mean().unstack(fill_value=0)
    family_stats.columns = ["No_Promo_Sales", "Promo_Sales"]

    family_stats["family_dollar_diff"] = family_stats["Promo_Sales"] - family_stats["No_Promo_Sales"]
    family_stats["family_promo_ratio"] = np.where(
        family_stats["No_Promo_Sales"] > 0,
        family_stats["Promo_Sales"] / family_stats["No_Promo_Sales"],
        1.0
    )

    df["family_dollar_diff"] = df["family"].map(family_stats["family_dollar_diff"])
    df["family_promo_ratio"] = df["family"].map(family_stats["family_promo_ratio"])
    df["expected_promo_boost"] = df["onpromotion"] * df["family_promo_ratio"]
    df.drop(columns=["family_promo_ratio"], inplace=True)

    if test_df is not None:
        test_df = test_df.sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)
        test_df["onpromotion_lead_1"] = test_df.groupby(["store_nbr", "family"])["onpromotion"].shift(-1).fillna(0)
        test_df["family_dollar_diff"] = test_df["family"].map(family_stats["family_dollar_diff"])
        test_df["family_promo_ratio"] = test_df["family"].map(family_stats["family_promo_ratio"])
        test_df["expected_promo_boost"] = test_df["onpromotion"] * test_df["family_promo_ratio"]
        test_df.drop(columns=["family_promo_ratio"], inplace=True)
        return df, test_df

    return df
