from src.features.calendar_features import add_calendar_features
from src.features.sales_features import add_sales_features
from src.features.trans_features import add_transaction_features
from src.features.promo_features import add_promo_features
from src.features.hierarchical_features import add_hierarchical_features

def build_all_features(df):
    df = add_calendar_features(df)
    df = add_sales_features(df)
    df = add_transaction_features(df)
    df = add_promo_features(df)
    df = add_hierarchical_features(df)
    return df
