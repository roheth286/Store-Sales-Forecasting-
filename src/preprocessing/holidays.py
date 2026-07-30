import pandas as pd

def process_holidays(target_df, holidays_df):
    holidays = holidays_df[(~holidays_df["transferred"]) & (holidays_df["type"] != "Work Day")].copy()
    holidays["is_Holiday"] = holidays["type"].isin(["Holiday", "Additional", "Bridge"]).astype(int)
    holidays["is_Event"] = (holidays["type"] == "Event").astype(int)

    national = holidays[holidays["locale"] == "National"].groupby("date")[["is_Holiday", "is_Event"]].max().reset_index()

    regional = holidays[holidays["locale"] == "Regional"].rename(columns={"locale_name": "state"})
    regional = regional.groupby(["date", "state"])[["is_Holiday", "is_Event"]].max().reset_index()

    local = holidays[holidays["locale"] == "Local"].rename(columns={"locale_name": "city"})
    local = local.groupby(["date", "city"])[["is_Holiday", "is_Event"]].max().reset_index()

    temp = target_df[["date", "state", "city"]].copy()

    temp = temp.merge(national, on="date", how="left")
    temp.rename(columns={"is_Holiday": "nat_h", "is_Event": "nat_e"}, inplace=True)

    temp = temp.merge(regional, on=["date", "state"], how="left")
    temp.rename(columns={"is_Holiday": "reg_h", "is_Event": "reg_e"}, inplace=True)

    temp = temp.merge(local, on=["date", "city"], how="left")
    temp.rename(columns={"is_Holiday": "loc_h", "is_Event": "loc_e"}, inplace=True)

    for col in ["nat_h", "nat_e", "reg_h", "reg_e", "loc_h", "loc_e"]:
        temp[col] = temp[col].fillna(0).astype(int)

    target_df["is_Holiday"] = temp[["nat_h", "reg_h", "loc_h"]].max(axis=1)
    target_df["is_Event"] = temp[["nat_e", "reg_e", "loc_e"]].max(axis=1)
    return target_df
