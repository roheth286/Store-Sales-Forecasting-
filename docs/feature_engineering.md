# Feature Engineering & Preprocessing Documentation

This document provides a comprehensive technical guide to the data preprocessing pipeline and the **42 features** constructed for the Corporación Favorita Store Sales Forecasting model.

---

## 1. Data Preprocessing & Merging Pipeline

The raw dataset spans multiple separate CSV files containing store metadata, economic indicators, foot traffic, and public holidays in Ecuador. To create a clean, unified dataset without data leakage or row duplication, the following 5-step preprocessing workflow was implemented:

### Step 1.1: Base Dataset Merge
* `train.csv` (3,000,888 rows) was merged with `stores.csv` on `store_nbr` using a left join to attach store metadata (`city`, `state`, `type`, `cluster`).
* `date` columns across all dataframes were converted to standardized `datetime64[ns]` objects.

### Step 1.2: 4-Step Holiday Logic & Scope Aggregation
Directly merging `holidays_events.csv` causes severe row duplication because multiple holidays (or national + local events) occur on the same calendar date. To prevent duplication:
1. **Filtering:** Filtered out transferred holidays (`transferred == True`) and standard work days (`type == 'Work Day'`).
2. **Category Binary Mapping:** Categorized remaining holiday events into `is_Holiday` (1 for `Holiday`, `Additional`, `Bridge`) and `is_Event` (1 for `Event`, such as Earthquake emergency relief).
3. **Location Scope Tables:** Created 3 separate aggregated lookup tables based on location scope:
   * **National Scope (`locale == 'National'`):** Applies to all 54 stores in Ecuador on that date.
   * **Regional Scope (`locale == 'Regional'`):** Applies only to stores located in matching `state`.
   * **Local Scope (`locale == 'Local'`):** Applies only to stores located in matching `city`.
4. **Aggregation:** Grouped each scope table by `[date, location]` using `.max()` so multiple holidays on one date collapse into a single binary flag (`1`).

### Step 1.3: Continuous Daily Oil Price Interpolation (20-Point Bisect Neighborhood Average)
* `oil.csv` contains daily crude oil prices ($dcoilwtico$), but has missing data on weekends, holidays, and market closure gaps.
* Created a continuous daily date table spanning **Jan 1, 2013 to Aug 31, 2017**.
* For every missing oil price date, used a **20-point bisect neighborhood average algorithm** (`bisect.bisect_right`):
  - Located the **10 nearest known oil prices before** the date.
  - Located the **10 nearest known oil prices after** the date.
  - Calculated their 20-point local mean to continuously interpolate missing prices across weekend gaps and holiday closures without introducing artificial boundary jumps.

### Step 1.4: Store Transaction & Foot Traffic Imputation
* `transactions.csv` was missing records for 118 active trading days across various stores.
* **Closed Days:** For days where store daily sales totaled exactly zero (`total_sales == 0`), transactions were explicitly set to `0`.
* **Active Glitch Days:** For active days with missing transactions, store-by-store linear regression models ($y = mx + c$) were fitted on known sales vs. transaction data ($R^2 > 0.85$). Missing transaction values were imputed using each store's custom linear fit:
  $$\text{imputed\_transactions} = \max(0, \text{round}(m \cdot \text{total\_sales} + c))$$

### Step 1.5: Filtering Pre-Opening Zero-Sales Rows
* Analysis revealed that 8 late-opening stores (Store 52, 22, 21, 42, 29, 20, 53, 36) had artificial zero-sales records logged prior to their actual grand opening dates (e.g., Store 52 opened on April 20, 2017, but had 1,566 zero-sales rows starting from 2013).
* Identified the first active sales date ($\text{sales} > 0$) for each store and filtered out all pre-opening rows.
* **Impact:** Reduced dataset size from `3,000,888` to `2,778,831` clean active rows, eliminating `222,057` uninformative zero targets.

---

## 2. Detailed Breakdown of All 42 Features

Below is the complete technical documentation for every feature column in the final dataset, explaining **How** it was computed and **Why** it was created based on clues discovered from EDA plots.

---

### Category 1: Identifiers & Target (5 Columns)

#### 1. `id`
* **How:** Original unique row ID from `train.csv`.
* **Why:** Required for row indexing and Kaggle submission formatting.

#### 2. `date`
* **How:** Chronological date timestamp (`YYYY-MM-DD`).
* **Why:** Primary time index for temporal grouping, lag creation, and sorting.

#### 3. `store_nbr`
* **How:** Categorical integer (1 to 54) representing individual store locations.
* **Why:** Allows the model to distinguish store-level sales capacity and geographic differences.

#### 4. `family`
* **How:** Categorical string (33 unique product categories, e.g., `GROCERY I`, `BEVERAGES`, `PRODUCE`).
* **Why:** Primary product level grouping; sales volume and promotional responsiveness vary dramatically by family.

#### 5. `sales`
* **How:** Numerical float representing daily unit sales per store-family pair.
* **Why:** **Target Variable** to be predicted.

---

### Category 2: Store Metadata & Promotions (5 Columns)

#### 6. `city`
* **How:** Categorical store city from `stores.csv` (e.g., `Quito`, `Guayaquil`, `Cuenca`).
* **Why:** Captures city-level consumer purchasing power and local holiday effects.

#### 7. `state`
* **How:** Categorical store state/province from `stores.csv` (e.g., `Pichincha`, `Guayas`).
* **Why:** Enables matching regional state holidays and province-level economic trends.

#### 8. `type`
* **How:** Categorical store format class (`Type A`, `B`, `C`, `D`, `E`).
* **Why:** EDA revealed Type A stores generate over 2x the sales volume of Type C/E stores.

#### 9. `cluster`
* **How:** Integer (1 to 17) representing clusters of similar store formats.
* **Why:** Groups stores with similar product assortments and purchasing behavior.

#### 10. `onpromotion`
* **How:** Integer count of items within a product family actively on promotion on that date.
* **Why:** Direct measure of promotional activity; primary driver of sales spikes.

---

### Category 3: Preprocessed Holiday & Event Flags (2 Columns)

#### 11. `is_Holiday`
* **How:** Binary flag (`1` or `0`), equal to 1 if a National, Regional, or Local holiday was active.
* **Why (EDA Clue):** Bar plot analysis (`plot_holiday_vs_sales`) proved average daily store sales increase from ~355 units on regular days to >420 units on holiday days (+18.3% lift).

#### 12. `is_Event`
* **How:** Binary flag (`1` or `0`), equal to 1 for major events (e.g., April 2016 Earthquake relief).
* **Why (EDA Clue):** Major national events cause sudden emergency purchasing surges across staple categories.

---

### Category 4: Macro-Economic Oil Signals (3 Columns)

#### 13. `oil_price`
* **How:** Daily crude oil price ($dcoilwtico$), continuously imputed across weekend and holiday gaps using a 20-point bisect neighborhood average algorithm (10 known oil prices before + 10 known oil prices after).
* **Why:** Ecuador is an oil-dependent economy; crude oil export revenues directly influence national consumer purchasing power.

#### 14. `oil_roll_mean_30`
* **How:** 30-day centered moving average of crude oil prices (`window=30, center=True, min_periods=1`).
* **Why (EDA Clue):** The dual Y-axis plot (`plot_sales_vs_oil`) proved a striking **inverse relationship**: as crude oil prices plummeted from over $100/barrel in 2014 down to ~$30/barrel in late 2015, total store sales steadily rose across Ecuador, proving that sales follow macroeconomic oil trends over smooth 30-day windows rather than daily price noise.

#### 15. `oil_diff_7`
* **How:** 7-day price difference ($\text{oil}_t - \text{oil}_{t-7}$).
* **Why:** Measures short-term oil price shocks and rapid price movements.

---

### Category 5: Store Foot Traffic & Transaction Features (5 Columns)

#### 16. `transactions`
* **How:** Store daily customer transactions (imputed for closed/glitch days).
* **Why:** Foot traffic is the strongest physical correlate of store sales volume.

#### 17. `trans_lag_1`
* **How:** 1-day lagged store transaction count (`groupby('store_nbr')['transactions'].shift(1)`).
* **Why:** Captures immediate yesterday customer traffic.

#### 18. `trans_lag_7`
* **How:** 7-day lagged store transaction count (`shift(7)`).
* **Why:** Captures same-day-last-week customer traffic.

#### 19. `trans_roll_mean_7`
* **How:** 7-day simple trailing moving average of store transactions.
* **Why:** Measures recent 1-week foot traffic momentum.

#### 20. `trans_roll_mean_30`
* **How:** 30-day simple trailing moving average of store transactions.
* **Why:** Measures monthly baseline store popularity and customer traffic.

---

### Category 6: Calendar & Payday Window Signals (8 Columns)

#### 21 – 27. `day_Monday` through `day_Sunday`
* **How:** 7 One-Hot binary indicator columns for each day of the week.
* **Why (EDA Clue):** The Weekly Seasonal Plot (`plot_weekly_seasonality`) proved extreme day-of-week seasonality: Thursday is the lowest sales day (~283k total units), while Sunday is the peak sales day (~463k total units).

#### 28. `is_payday_window`
* **How:** Binary flag equal to 1 for days 1, 2, 15, 16, 17, 30, 31, and month-ends.
* **Why (EDA Clue):** The Periodogram (`plot_periodogram`) identified a prominent frequency spike at **24 cycles/year** (semi-monthly payday frequency, matching Ecuador's 15th and 30th salary payments).

---

### Category 7: Sales Lags & Trailing Rolling Windows (7 Columns)

#### 29. `sales_lag_1`
* **How:** 1-day lagged sales grouped by `(store_nbr, family)` (`shift(1)`).
* **Why (EDA Clue):** PACF plot (`plot_pacf`) showed the strongest direct correlation spike at Lag 1 ($r = 0.767$).

#### 30. `sales_lag_7`
* **How:** 7-day lagged sales grouped by `(store_nbr, family)` (`shift(7)`).
* **Why (EDA Clue):** ACF plot (`plot_autocorrelation`) and Periodogram proved a massive weekly cyclical correlation ($r = 0.936$ at Lag 7).

#### 31. `sales_lag_14`
* **How:** 14-day lagged sales grouped by `(store_nbr, family)` (`shift(14)`).
* **Why (EDA Clue):** ACF plot proved 2-week cyclical correlation ($r = 0.928$ at Lag 14).

#### 32. `sales_lag_21`
* **How:** 21-day lagged sales grouped by `(store_nbr, family)` (`shift(21)`).
* **Why (EDA Clue):** ACF plot proved 3-week cyclical correlation ($r = 0.916$ at Lag 21).

#### 33. `sales_lag_28`
* **How:** 28-day lagged sales grouped by `(store_nbr, family)` (`shift(28)`).
* **Why (EDA Clue):** ACF plot proved 4-week cyclical correlation ($r = 0.912$ at Lag 28).

#### 34. `sales_roll_mean_30`
* **How:** 30-day simple trailing moving average sales grouped by `(store_nbr, family)` (`center=False, min_periods=1`).
* **Why (EDA Clue):** Monthly Seasonal Plot (`plot_monthly_seasonality`) showed long-term trend shifts across May (Mother's Day), September (Back-to-School), and December (Christmas).

#### 35. `sales_roll_std_7`
* **How:** 7-day simple trailing standard deviation of sales grouped by `(store_nbr, family)`.
* **Why:** Captures short-term sales volatility and uncertainty.

---

### Category 8: Manufactured Promotion Dynamics (3 Columns)

#### 36. `onpromotion_lead_1`
* **How:** 1-day lead promotion count (`shift(-1)` looking 1 day ahead into tomorrow).
* **Why:** Promotional campaigns are announced in advance; customers adjust today's purchases if a discount starts tomorrow.

#### 37. `family_dollar_diff`
* **How:** Category-level historical average dollar sales gain on promotion ($\text{Promo\_Sales} - \text{No\_Promo\_Sales}$).
* **Why (EDA Clue):** Promotion Impact Plot (`plot_promo_impact_all_families`) proved that `BEVERAGES`, `GROCERY I`, and `PRODUCE` generate **+$1,600 to +$1,900 extra dollars per day** on promotion, whereas `BOOKS` generates $0.

#### 38. `expected_promo_boost`
* **How:** Product of daily `onpromotion` count and category promo sales ratio ($\text{onpromotion} \times \frac{\text{Promo\_Sales}}{\text{No\_Promo\_Sales}}$).
* **Why (EDA Clue):** `PRODUCE` sales jump +207.8% on promotion ($3.07\times$ multiplier), whereas `AUTOMOTIVE` jumps very little. This feature scales `onpromotion` by category responsiveness.

---

### Category 9: Hierarchical Level Target Encodings (4 Columns)

#### 39. `mean_sales_by_family`
* **How:** Overall historical mean sales grouped by product `family`.
* **Why (EDA Clue):** Diagnostic Grid Plot (`plot_store_hierarchies`, Panel 1) showed `GROCERY I` averages >4,000 units while `BOOKS` averages <1 unit.

#### 40. `mean_sales_by_store`
* **How:** Overall historical mean sales grouped by `store_nbr`.
* **Why:** Captures overall store size and sales capacity.

#### 41. `mean_sales_by_cluster`
* **How:** Overall historical mean sales grouped by store `cluster`.
* **Why (EDA Clue):** Diagnostic Grid Plot (Panel 2) proved Cluster 5 stores average >1,100 units while Cluster 3 stores average ~200 units.

#### 42. `mean_sales_by_type`
* **How:** Overall historical mean sales grouped by store format `type`.
* **Why (EDA Clue):** Diagnostic Grid Plot (Panel 3 & 4) proved Type A stores account for 32.9% of total national sales volume.
