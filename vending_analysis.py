# 1). Data Cleaning and Preparation: Handle missing values, duplicate entries, 
# and data type conversions. Ensure the datasets are clean and ready for analysis.

# Step1: Load and Inspect Data
# Purpose: Confirm data structure, types, and initial cleanliness.

# Load data
import pandas as pd
import numpy as np

import sweetviz as sv
from ydata_profiling import ProfileReport

import webbrowser

inventory_df = pd.read_csv("Inventory_Turnover.csv")
restock_df = pd.read_csv("Restock_data.csv")

# Inspect data
print(inventory_df.info())
# print(inventory_df.head())

print(restock_df.info())
print(restock_df.head())

# Step 2: Handle Missing Values

# Check for Missing Data
print(inventory_df.isnull().sum())
print(restock_df.isnull().sum())

# If any columns contain missing data, we decide:
# 1. Drop rows (if small count and non-critical).
# 2. Impute values (if meaningful defaults or calculations can be applied).
# Based on the output, both dataframes have no nulls now, but it's always good to validate.

# Step 3: Remove Duplicates

# Check and drop duplicates:

# Check for duplicate rows
print(inventory_df.duplicated().sum())
print(restock_df.duplicated().sum())

# Drop duplicates if any
inventory_df = inventory_df.drop_duplicates()
restock_df = restock_df.drop_duplicates()

# Step 4: Convert Date Columns to datetime

# dispense_date and restock_date must be converted:

# inventory_df has date format like "22-02-2024" to be converted to the YYYY-MM-DD... format
inventory_df['dispense_date'] = pd.to_datetime(inventory_df['dispense_date'], format="%d-%m-%Y")

# restock_df has full timestamp format
restock_df['restock_date'] = pd.to_datetime(restock_df['restock_date'])

# Step 5: Verify and Convert Numeric Columns

# Ensure correct types (e.g., total, package_qty, qty_dispensed are numeric):

# Confirm numeric types
print(inventory_df[['package_qty', 'qty_dispensed']].dtypes)
print(restock_df[['total']].dtypes)

# Check for non-numeric values (if from external sources)
inventory_df['package_qty'] = pd.to_numeric(inventory_df['package_qty'], errors='coerce')
inventory_df['qty_dispensed'] = pd.to_numeric(inventory_df['qty_dispensed'], errors='coerce')
restock_df['total'] = pd.to_numeric(restock_df['total'], errors='coerce')

# If any NaN values are introduced due to bad entries, 
# you'd detect them now and handle appropriately (e.g., dropping or imputing).

# Step 6: Trim Whitespace and Standardize Strings

# Useful for joining/merging on device_id or sku.

# Strip whitespace in string columns
inventory_df['sku'] = inventory_df['sku'].str.strip()
inventory_df['device_id'] = inventory_df['device_id'].str.strip()

restock_df['device_id'] = restock_df['device_id'].str.strip()
restock_df['currency_code'] = restock_df['currency_code'].str.strip()

# Step 7: Create Additional Useful Columns

# Add month, week, or day for time series/grouped analysis:

# Add month columns
inventory_df['month'] = inventory_df['dispense_date'].dt.month_name()
restock_df['month'] = restock_df['restock_date'].dt.month_name()
# inventory_df['dispense_month'] = inventory_df['dispense_date'].dt.to_period('M')
# restock_df['restock_month'] = restock_df['restock_date'].dt.to_period('M')

#Add ISO week number (1–53) and day columns
inventory_df['week'] = inventory_df['dispense_date'].dt.isocalendar().week
restock_df['week'] = restock_df['restock_date'].dt.isocalendar().week

inventory_df['day'] = inventory_df['dispense_date'].dt.day_name()
restock_df['day'] = restock_df['restock_date'].dt.day_name()

# Preview the changes:
print(inventory_df[['dispense_date', 'month','week', 'day']].head())
print(restock_df[['restock_date', 'month', 'week', 'day']].head())

# Step 8: Final Check: Confirm Cleaned Data
print(inventory_df.info())
print(restock_df.info())
print(inventory_df.head())
print(restock_df.head())

# Summary
# We have:
# Cleaned, de-duplicated data
# Standard datetime types
# Trimmed strings for accurate merges
# Ready for aggregation, visualization, or joining across device_id

# ////////////////////

# 2). Exploratory Data Analysis (EDA): Conduct an initial analysis 
# to understand the datasets characteristics, 
# including demand patterns, seasonality, and inventory levels.

#### EDA Using Sweetviz and YData-Profiling

# === Generate Sweetviz report for inventory_df ===
# print("Generating Sweetviz report for inventory_df...")
# inventory_report = sv.analyze(inventory_df)
# inventory_report.show_html("inventory_sweetviz_report.html")

# === Generate YData-Profiling report for inventory_df ===
# Transform the DataFrame into a Profile Report
# inventory_profile_report = ProfileReport(df=inventory_df, explorative=True, title='Inventory Analytics')
# inventory_profile_report.to_file('inventory_profile_report.html')
# webbrowser.open("inventory_profile_report.html")

# # === Generate Sweetviz report for restock_df ===
# print("Generating Sweetviz report for restock_df...")
# restock_report = sv.analyze(restock_df)
# restock_report.show_html("restock_sweetviz_report.html")

# # === Generate YData-Profiling report for restock_df ===
# # Transform the DataFrame into a Profile Report
# restock_profile_report = ProfileReport(df=restock_df, explorative=True, title='Restock Analytics')
# restock_profile_report.to_file('restock_profile_report.html')
# webbrowser.open("restock_profile_report.html")

# -------------
# -------------

# Step-by-Step EDA
# Let's walk through a detailed EDA) of the datasets, focusing on:
# A. Demand patterns (how much is dispensed over time)
# B. Seasonality (trends across months/days)
# C. Inventory levels (what's available vs. restocked)

# Step1: Import additional libraries

import matplotlib.pyplot as plt
import seaborn as sns

# Step2: Summary statistics
# Look for: Date ranges, Outliers in qty_dispense or package_qty
print(inventory_df.describe())
# min package_qty = 1, max package_qty = 2, mean = 1, std = 0.022
# min dispense_qty = 1, max dispense_qty = 162, mean = 8.15, std = 9.055
print(restock_df.describe())
# min total = 5, max total = 612, mean = 272.5, std = 101.077

# Step 3: Generate plots for demand patterns, seasonality, and inventory levels.

# Create output directory if it doesn't exist
import os

# Set up plot saving
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# 1. DAILY DEMAND PATTERN
daily_demand = inventory_df.groupby('dispense_date')['qty_dispensed'].sum()

plt.figure(figsize=(14, 5))
daily_demand.plot()
plt.title("Daily Total Quantity Dispensed")
plt.ylabel("Quantity")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "daily_quantity_dispensed.png"))
plt.savefig(os.path.join(output_dir, "daily_quantity_dispensed.pdf"))
plt.close()

# 2. DEMAND BY DAY OF WEEK

dow_demand = inventory_df.groupby('day')['qty_dispensed'].sum().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

plt.figure(figsize=(10, 6))
# sns.barplot(x=dow_demand.index, y=dow_demand.values, palette="viridis")
sns.barplot(x=dow_demand.index, y=dow_demand.values, color="skyblue")
plt.title("Total Quantity Dispensed by Day of Week")
plt.ylabel("Quantity")
plt.xlabel("Day of Week")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "demand_by_day_of_week.png"))
plt.savefig(os.path.join(output_dir, "demand_by_day_of_week.pdf"))
plt.close()

# 3. DEMAND BY WEEK OF YEAR
woy_demand = inventory_df.groupby('week')['qty_dispensed'].sum()
woy_demand.plot()
plt.title("Total Quantity Dispensed by Week of Year")
plt.ylabel("Quantity")
plt.xlabel("Week of Year")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "weekly_quantity_dispensed.png"))
plt.savefig(os.path.join(output_dir, "weekly_quantity_dispensed.pdf"))
plt.close()

# 4. DEMAND BY MONTH OF YEAR
monthly_demand = inventory_df.groupby('month')['qty_dispensed'].sum()
monthly_demand.plot()
plt.title("Total Quantity Dispensed Monthly")
plt.ylabel("Quantity")
plt.xlabel("Month of Year")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "monthly_quantity_dispensed.png"))
plt.savefig(os.path.join(output_dir, "monthly_quantity_dispensed.pdf"))
plt.close()

# 5. DISPENSES OVER TIME PER DEVICE
device_demand = inventory_df.groupby(['dispense_date', 'device_id'])['qty_dispensed'].sum().unstack().fillna(0)

plt.figure(figsize=(14, 6))
device_demand.plot(legend=False)
plt.title("Daily Dispense Volume per Device (Trend)")
plt.ylabel("Quantity")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "device_dispense_trend.png"))
plt.savefig(os.path.join(output_dir, "device_dispense_trend.pdf"))
plt.close()

# 6. RESTOCKS OVER TIME
restock_counts = restock_df.groupby('restock_date').size()

plt.figure(figsize=(14, 5))
restock_counts.plot(kind='bar')
plt.title("Restock Frequency Over Time")
plt.ylabel("Restock Events")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "restock_frequency.png"))
plt.savefig(os.path.join(output_dir, "restock_frequency.pdf"))
plt.close()

# 7. PACKAGE VS DISPENSED DISTRIBUTION
plt.figure(figsize=(10, 6))
sns.histplot(inventory_df['package_qty'], label='Package Qty', color='skyblue', kde=True)
sns.histplot(inventory_df['qty_dispensed'], label='Qty Dispensed', color='orange', kde=True)
plt.title("Distribution: Package Quantity vs. Dispensed Quantity")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "package_vs_dispensed_distribution.png"))
plt.savefig(os.path.join(output_dir, "package_vs_dispensed_distribution.pdf"))
plt.close()

print(f"EDA visuals saved in: {os.path.abspath(output_dir)}")

#///////////////////////
#///////////////////////

# 3)•Feature Engineering: 
# Create new features that could be helpful for the model, 
# such as time-based attributes (day of the week, month, etc.) or lag features.

# #### Why? Because these features help capture seasonality, demand cycles, and weekday effects.

# A. Additional Time-Based Features (Inventory Dataset)

inventory_df['year'] = inventory_df['dispense_date'].dt.year
inventory_df['month_num'] = inventory_df['dispense_date'].dt.month
inventory_df['day_of_week'] = inventory_df['dispense_date'].dt.dayofweek  # 0=Monday
inventory_df['is_weekend'] = inventory_df['day_of_week'].isin([5, 6])  # Saturday or Sunday
inventory_df['day_num'] = inventory_df['dispense_date'].dt.day # Monday = 0

# # Additional Time-Based Features (Restock Dataset)
# restock_df['year'] = restock_df['restock_date'].dt.year
# restock_df['month_num'] = restock_df['restock_date'].dt.month
# restock_df['day_of_week'] = restock_df['restock_date'].dt.dayofweek  # 0=Monday
# restock_df['is_weekend'] = restock_df['day_of_week'].isin([5, 6])  # Saturday or Sunday
# restock_df['day_num'] = restock_df['restock_date'].dt.day # Monday = 0

# B. Lag Features (Qty Dispensed on Previous Days)

# Sort by date and device
# Ensures that the data is chronologically ordered per device, 
# which is critical when creating lag and rolling features.
inventory_df = inventory_df.sort_values(by=['device_id', 'dispense_date'])

# Lag features per device
# Why? These features help your model understand short-term demand memory and weekly cycles.
inventory_df['lag_1'] = inventory_df.groupby('device_id')['qty_dispensed'].shift(1)
inventory_df['lag_7'] = inventory_df.groupby('device_id')['qty_dispensed'].shift(7)  # same weekday last week
inventory_df['rolling_7_mean'] = inventory_df.groupby('device_id')['qty_dispensed'].transform(lambda x: x.shift(1).rolling(7).mean())
inventory_df['rolling_7_std'] = inventory_df.groupby('device_id')['qty_dispensed'].transform(lambda x: x.shift(1).rolling(7).std())
# What the last 2 lines do:
# Groups by device_id; For each device:
# Shifts values by 1 (so today's value isn't included)
# Calculates the rolling average and std over the previous 7 days

# Summary Table:
# Feature	               Purpose	                                 Method
# lag_1	            Value from 1 day before	                    shift(1)
# lag_7	            Value from 7 days before	                shift(7)
# rolling_7_mean	Average of last 7 days (excluding today)	shift(1).rolling(7).mean()
# rolling_7_std     Std of last 7 days (excluding today)	    shift(1).rolling(7).std()

# B1. Cumulative Quantity Dispensed
# Why? Helps identify overall usage and wear-out rates of machines.
inventory_df['cumulative_dispense'] = inventory_df.groupby('device_id')['qty_dispensed'].cumsum()

# B2. Days Since Last Dispense (Per Device)

# Why? Indicates frequency of usage, irregularities, or gaps in demand.
inventory_df['days_since_last'] = inventory_df.groupby('device_id')['dispense_date'].diff().dt.days

# ******* C. Merge with Restock Data to Create Supply-Demand Features *****

# Why? This reveals how long it’s been since a machine was 
# last stocked — useful for identifying restock efficiency.

# Merge nearest restock before dispense
inventory_with_last = pd.merge_asof(
    inventory_df.sort_values('dispense_date'),
    restock_df[['device_id', 'restock_date']].sort_values('restock_date'),
    by='device_id',
    left_on='dispense_date',
    right_on='restock_date',
    direction='backward'
)

# Purpose: Matches each dispense event with the most recent restock event that happened on or before that dispense date.

# Here's what's happening:
# merge_asof() is like a "fuzzy join" that merges on nearest keys, not exact matches.
# It:
# Matches on device_id
# Joins each dispense_date with the last restock_date before it (direction='backward')
# Both DataFrames must be sorted by their respective datetime columns!

# Example:

# dispense_date	restock_date (matched)
# 2024-01-05	2024-01-03
# 2024-01-09	2024-01-08
# 2024-01-15	2024-01-08

# This way, we know which restock was still in effect before each dispense.

# Days since last restock
inventory_with_last['days_since_restock'] = (inventory_with_last['dispense_date'] - inventory_with_last['restock_date']).dt.days

# Purpose: Measures how many days passed between the matched restock and the dispense date.

# The .dt.days extracts just the integer number of days.
# This new feature can help model inventory pressure, time-to-depletion, or restock efficiency.
# Example:
# dispense_date	restock_date	days_since_restock
# 2024-01-05	2024-01-03	         2
# 2024-01-09	2024-01-08	         1

# Why this is valuable:
# We can now analyze demand trends after a restock.
# This feature (days_since_restock) can help detect:
# How fast inventory depletes
# How long restocks last
# Seasonality in restocking needs

# Days until next restock
# We already calculated days_since_restock using a backward join. 
# Now let’s calculate days_until_next_restock using a forward join with merge_asof.

# Forward join: find the next restock after each dispense
# Make sure both DataFrames are sorted
inventory_df_sorted = inventory_with_last.sort_values('dispense_date')
restock_df_sorted = restock_df[['device_id', 'restock_date']].sort_values('restock_date')

# Perform merge_asof
inventory_with_next = pd.merge_asof(
    inventory_df_sorted,
    restock_df_sorted,
    by='device_id',
    left_on='dispense_date',
    right_on='restock_date',
    direction='forward'
)

# # Debug step: Show columns to verify merge
# print("🧾 Columns after forward merge:", inventory_with_next.columns.tolist())

# Rename if restock_date was successfully added
if 'restock_date_y' in inventory_with_next.columns:
    inventory_with_next.rename(columns={'restock_date_x': 'last_restock_date', 'restock_date_y': 'next_restock_date'}, inplace=True)
else:
    raise KeyError("❌ 'restock_date_y' column not found after forward merge. Cannot rename to 'next_restock_date'.")

# Calculate days until next restock
inventory_with_next['days_until_next_restock'] = (
    inventory_with_next['next_restock_date'] - inventory_with_next['dispense_date']
).dt.days

# ******* D. Visualize Dispense Trends vs. Restock *******
# This can help us spot bottlenecks, predict depletion, and optimize scheduling.
# Example Plot:

# Choose a device to plot (or loop through a few)
device = 'device_af645ebf4c96eb6e430529a2a9913686'  # Replace with actual ID from your dataset

# Filter data
device_data = inventory_with_next[inventory_with_next['device_id'] == device]

# # Plot qty_dispensed over time
# plt.figure(figsize=(12, 6))
# plt.plot(device_data['dispense_date'], device_data['qty_dispensed'], label='Qty Dispensed')

# # Mark restock events
# plt.scatter(device_data['last_restock_date'], [0]*len(device_data), color='green', label='Last Restock', marker='^')
# plt.scatter(device_data['next_restock_date'], [0]*len(device_data), color='red', label='Next Restock', marker='v')

# plt.title(f'Device: {device} — Dispense & Restock Timeline')
# plt.xlabel('Date')
# plt.ylabel('Qty Dispensed')
# plt.legend()
# plt.tight_layout()

max_y = device_data['qty_dispensed'].max()

plt.figure(figsize=(12, 6))
plt.plot(device_data['dispense_date'], device_data['qty_dispensed'], label='Qty Dispensed', color='blue')

plt.scatter(device_data['last_restock_date'], [max_y * 1.05] * len(device_data), color='green', label='Last Restock', marker='^')
plt.scatter(device_data['next_restock_date'], [max_y * 1.10] * len(device_data), color='red', label='Next Restock', marker='v')

plt.title(f'Device: {device} — Dispense & Restock Timeline')
plt.xlabel('Date')
plt.ylabel('Qty Dispensed')
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "device_13686_Dispense_and_Restock_Timeline.png"))
plt.savefig(os.path.join(output_dir, "device_13686_Dispense_and_Restock_Timeline.pdf"))

plt.show()

