import pandas as pd

def load_raw_data(data_dir="data/raw"):
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    stores_df = pd.read_csv(f"{data_dir}/stores.csv")
    transactions_df = pd.read_csv(f"{data_dir}/transactions.csv")
    oil_df = pd.read_csv(f"{data_dir}/oil.csv")
    holidays_df = pd.read_csv(f"{data_dir}/holidays_events.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    train_df['date'] = pd.to_datetime(train_df['date'], format='%Y-%m-%d')
    transactions_df['date'] = pd.to_datetime(transactions_df['date'], format='%Y-%m-%d')
    oil_df['date'] = pd.to_datetime(oil_df['date'], format='%Y-%m-%d')
    holidays_df['date'] = pd.to_datetime(holidays_df['date'], format='%Y-%m-%d')
    test_df['date'] = pd.to_datetime(test_df['date'], format='%Y-%m-%d')

    df = train_df.merge(stores_df, on="store_nbr", how="left")
    test_df = test_df.merge(stores_df, on="store_nbr", how="left")

    return df, stores_df, transactions_df, oil_df, holidays_df, test_df
