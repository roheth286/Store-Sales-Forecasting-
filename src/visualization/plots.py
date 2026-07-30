import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import periodogram
from statsmodels.tsa.stattools import pacf

def plot_holiday_vs_sales(df, output_dir="plots"):
    plt.close('all')
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='is_Holiday', y='sales', errorbar=('ci', 95), color="#467297")
    plt.title('Average Store Sales per Day', fontsize=12, fontweight='bold')
    plt.xlabel('Is Holiday (0 = No, 1 = Yes)')
    plt.ylabel('Average Sales')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/holiday_vs_sales.png", dpi=300, bbox_inches="tight")
    plt.close('all')

def plot_sales_vs_oil(df, full_oil_df, output_dir="plots"):
    plt.close('all')
    os.makedirs(output_dir, exist_ok=True)
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    daily_sales['sales_roll_30'] = daily_sales['sales'].rolling(window=30, center=True, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    color = '#316896'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales (30-day Centered MA)', color=color)
    ax1.plot(daily_sales['date'], daily_sales['sales_roll_30'], color=color, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#b03a2e'
    ax2.set_ylabel('Oil Price (30-day Centered MA)', color=color)
    ax2.plot(full_oil_df['date'], full_oil_df['oil_roll_mean_30'], color=color, linestyle='--', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Relationship: Sales vs. Oil Price (30-Day Centered Moving Averages)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f"{output_dir}/sales_vs_oil.png", dpi=300, bbox_inches="tight")
    plt.close('all')

def plot_periodogram(df, output_dir="plots"):
    plt.close('all')
    os.makedirs(output_dir, exist_ok=True)
    daily_sales = df.groupby("date")["sales"].sum().asfreq("D").interpolate()

    freqs, spectrum = periodogram(
        daily_sales,
        fs=365,
        detrend="linear",
        scaling="spectrum"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.step(freqs, spectrum, color="purple", linewidth=1.5)
    ax.set_xscale("log")

    ax.set_xticks(
        [1, 2, 4, 6, 12, 24, 52, 104],
        labels=[
            "Annual (1/yr)",
            "Semiannual (2/yr)",
            "Quarterly (4/yr)",
            "Bimonthly (6/yr)",
            "Monthly (12/yr)",
            "Payday (24/yr)",
            "Weekly (52/yr)",
            "Semiweekly (104/yr)"
        ]
    )

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_ylabel("Variance / Power (Sales Variance)")
    ax.set_xlabel("Frequency (Cycles per Year)")
    ax.set_title("Periodogram of Total Daily Sales (Identifying Dominant Frequencies)", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/periodogram.png", dpi=300, bbox_inches="tight")
    plt.close('all')

def plot_weekly_seasonality(df, output_dir="plots"):
    plt.close('all')
    os.makedirs(output_dir, exist_ok=True)
    seasonal_df = df.groupby("date")["sales"].sum().reset_index()
    seasonal_df["day"] = seasonal_df["date"].dt.day_name()
    seasonal_df["week"] = seasonal_df["date"].dt.to_period("W").astype(str)

    weekday_order = ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"]

    avg_series = seasonal_df.groupby("day")["sales"].mean()
    avg_sales_ordered = [avg_series[day] for day in weekday_order]

    plt.figure(figsize=(15, 7))

    for _, week_data in seasonal_df.groupby("week"):
        week_data = week_data.copy()
        week_data["day"] = pd.Categorical(week_data["day"], categories=weekday_order, ordered=True)
        week_data = week_data.sort_values("day")
        plt.plot(week_data["day"], week_data["sales"], color="gray", alpha=0.15, linewidth=1)

    plt.plot(
        weekday_order,
        avg_sales_ordered,
        color="red",
        linewidth=4,
        marker="o",
        label="Average Weekly Pattern"
    )

    plt.title("Weekly Seasonal Plot: Sales Movement Over 7 Days Across All Weeks", fontsize=14, fontweight="bold")
    plt.xlabel("Day of the Week", fontweight="bold")
    plt.ylabel("Total Daily Sales", fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weekly_seasonality.png", dpi=300, bbox_inches="tight")
    plt.close('all')

def plot_monthly_seasonality(df, output_dir="plots"):
    plt.close('all')
    os.makedirs(output_dir, exist_ok=True)
    daily_df = df.groupby("date")["sales"].sum().reset_index()
    daily_df["month"] = daily_df["date"].dt.month_name()
    daily_df["year"] = daily_df["date"].dt.year

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    monthly_avg = daily_df.groupby(["year", "month"], observed=False)["sales"].mean().reset_index()
    monthly_avg["month"] = pd.Categorical(monthly_avg["month"], categories=month_order, ordered=True)
    monthly_avg = monthly_avg.sort_values(["year", "month"])

    overall_monthly_avg = (
        monthly_avg.groupby("month", observed=False)["sales"]
        .mean()
        .reindex(month_order)
    )
    overall_monthly_ordered = [overall_monthly_avg[m] for m in month_order]

    plt.figure(figsize=(15, 7))

    years = sorted(monthly_avg["year"].unique())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, year in enumerate(years):
        year_data = monthly_avg[monthly_avg["year"] == year].copy()
        year_data = year_data.sort_values("month")
        plt.plot(
            year_data["month"],
            year_data["sales"],
            marker="o",
            linewidth=2,
            alpha=0.7,
            color=colors[idx % len(colors)],
            label=f"Year {year}"
        )

    plt.plot(
        month_order,
        overall_monthly_ordered,
        color="black",
        linewidth=4,
        linestyle="--",
        marker="s",
        label="Overall Average Pattern"
    )

    plt.title("Monthly Seasonal Plot: Average Daily Sales by Month (2013 - 2017)", fontsize=14, fontweight="bold")
    plt.xlabel("Month of the Year", fontweight="bold")
    plt.ylabel("Average Daily Sales", fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_seasonality.png", dpi=300, bbox_inches="tight")
    plt.close('all')

def plot_autocorrelation(df, output_dir="plots"):
    plt.close('all')
    os.makedirs(output_dir, exist_ok=True)
    daily_sales = df.groupby(["date", "store_nbr", "family"])["sales"].sum().reset_index()
    daily_sales = daily_sales.sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)

    acf_results = {}
    for lag in range(1, 32):
        lagged = daily_sales.groupby(["store_nbr", "family"])["sales"].shift(lag)
        valid_idx = lagged.notnull()
        corr = daily_sales.loc[valid_idx, "sales"].corr(lagged[valid_idx])
        acf_results[lag] = corr

    lags = list(acf_results.keys())
    corrs = list(acf_results.values())
    colors = ["#d9534f" if lag in [7, 14, 21, 28] else "#9bc2e6" for lag in lags]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(lags, corrs, color=colors, edgecolor="navy", width=0.7)

    ax.set_xticks(lags)
    ax.set_ylim(0.80, 0.95)
    ax.set_xlabel("Lag Number (Days)", fontweight="bold")
    ax.set_ylabel("Pearson Correlation Coefficient", fontweight="bold")
    ax.set_title("Autocorrelation of Sales by Lag (Lags 1 to 31)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/autocorrelation.png", dpi=300, bbox_inches="tight")
    plt.close('all')

def plot_pacf(df, output_dir="plots"):
    plt.close('all')
    os.makedirs(output_dir, exist_ok=True)
    daily_sales = df.groupby("date")["sales"].sum()
    pacf_values = pacf(daily_sales, nlags=31, method="ywm")

    lags = np.arange(1, 32)
    pacf_vals_1_31 = pacf_values[1:32]
    n = len(daily_sales)
    conf_limit = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(14, 6))

    markerline, stemlines, baseline = ax.stem(
        lags, 
        pacf_vals_1_31, 
        linefmt="b-", 
        markerfmt="bo", 
        basefmt="r-"
    )
    plt.setp(markerline, markersize=6)
    plt.setp(stemlines, linewidth=1.5)

    ax.axhline(y=conf_limit, color="gray", linestyle="--", linewidth=1.5, label="95% Confidence Threshold")
    ax.axhline(y=-conf_limit, color="gray", linestyle="--", linewidth=1.5)

    ax.set_xticks(lags)
    ax.set_xlabel("Lag Number (Days)", fontweight="bold")
    ax.set_ylabel("Partial Autocorrelation Coefficient", fontweight="bold")
    ax.set_title("Partial Autocorrelation Function (PACF) - Lags 1 to 31", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pacf.png", dpi=300, bbox_inches="tight")
    plt.close('all')

def plot_store_hierarchies(df, output_dir="plots"):
    plt.close('all')
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    family_sales = df.groupby("family")["sales"].mean().sort_values(ascending=True)
    axes[0, 0].barh(family_sales.index, family_sales.values, color="#9bc2e6", edgecolor="navy", height=0.6)
    axes[0, 0].set_title("1. All 33 Product Families (Average Daily Sales per Store)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Mean Sales Units")

    cluster_sales = df.groupby("cluster")["sales"].mean().sort_values(ascending=True)
    axes[0, 1].barh([f"Cluster {c}" for c in cluster_sales.index], cluster_sales.values, color="#e07a5f", edgecolor="maroon", height=0.6)
    axes[0, 1].set_title("2. All 17 Store Clusters (Average Daily Sales per Store)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Mean Sales Units")

    type_sales = df.groupby("type")["sales"].mean().sort_values(ascending=False)
    axes[1, 0].bar(type_sales.index, type_sales.values, color="#6baf73", edgecolor="darkgreen", width=0.6)
    axes[1, 0].set_title("3. All 5 Store Types (Average Daily Sales Volume)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Store Format / Class")
    axes[1, 0].set_ylabel("Mean Sales Units")

    type_shares = df.groupby("type")["sales"].sum()
    explode = (0.05, 0.05, 0, 0, 0)
    axes[1, 1].pie(type_shares, labels=[f"Type {t}" for t in type_shares.index], autopct='%1.1f%%', startangle=140, explode=explode)
    axes[1, 1].set_title("4. Share of Total National Sales by Store Format (Pie Chart)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/store_hierarchies.png", dpi=300, bbox_inches="tight")
    plt.close('all')

def plot_promo_impact_all_families(df, output_dir="plots"):
    plt.close('all')
    os.makedirs(output_dir, exist_ok=True)
    promo_sales_all = (
        df.groupby(["family", df["onpromotion"] > 0])["sales"]
        .mean()
        .unstack(fill_value=0)
    )
    promo_sales_all.columns = ["No_Promo_Sales", "Promo_Sales"]

    promo_sales_all["Dollar_Diff"] = promo_sales_all["Promo_Sales"] - promo_sales_all["No_Promo_Sales"]

    promo_sales_all["Uplift_Pct"] = np.where(
        promo_sales_all["No_Promo_Sales"] > 0,
        (promo_sales_all["Dollar_Diff"] / promo_sales_all["No_Promo_Sales"]) * 100,
        0
    )

    all_33_sorted = promo_sales_all.sort_values(by="Dollar_Diff", ascending=True)

    fig, ax = plt.subplots(figsize=(16, 20))

    y_positions = np.arange(len(all_33_sorted))
    bar_width = 0.40

    ax.barh(
        y_positions - bar_width/2, 
        all_33_sorted["No_Promo_Sales"], 
        bar_width, 
        label="Non-Promoted Days", 
        color="lightgray", 
        edgecolor="gray"
    )

    ax.barh(
        y_positions + bar_width/2, 
        all_33_sorted["Promo_Sales"], 
        bar_width, 
        label="Promoted Days", 
        color="#2ca02c", 
        edgecolor="darkgreen"
    )

    for idx, (no_p, p, diff, uplift) in enumerate(zip(all_33_sorted["No_Promo_Sales"], all_33_sorted["Promo_Sales"], all_33_sorted["Dollar_Diff"], all_33_sorted["Uplift_Pct"])):
        if p > 0 and no_p > 0:
            label_text = f"+${diff:,.0f}/day (+{uplift:.1f}%)"
            ax.text(p + max(all_33_sorted["Promo_Sales"])*0.01, y_positions[idx] + bar_width/2, label_text, va="center", fontweight="bold", color="darkgreen", fontsize=9)
        elif p == 0:
            ax.text(no_p + max(all_33_sorted["Promo_Sales"])*0.01, y_positions[idx], "No Promo Data", va="center", fontstyle="italic", color="gray", fontsize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(all_33_sorted.index, fontweight="bold", fontsize=10)
    ax.set_xlabel("Average Daily Sales ($)", fontweight="bold", fontsize=12)
    ax.set_title("Promotion Impact: Sorted by Absolute Dollar Sales Gain per Day ($)", fontsize=15, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/promo_impact_all_families.png", dpi=300, bbox_inches="tight")
    plt.close('all')

def plot_all_visualizations(df, full_oil_df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating and saving all 9 plots to '{output_dir}/'...")
    plot_holiday_vs_sales(df, output_dir=output_dir)
    plot_sales_vs_oil(df, full_oil_df, output_dir=output_dir)
    plot_periodogram(df, output_dir=output_dir)
    plot_weekly_seasonality(df, output_dir=output_dir)
    plot_monthly_seasonality(df, output_dir=output_dir)
    plot_autocorrelation(df, output_dir=output_dir)
    plot_pacf(df, output_dir=output_dir)
    plot_store_hierarchies(df, output_dir=output_dir)
    plot_promo_impact_all_families(df, output_dir=output_dir)
    print(f"All 9 plots saved successfully to '{output_dir}/'!")
