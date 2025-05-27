
# STEP1: Data Cleaning & Preparation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import getcwd

# Path to the CSV file
data1_path = getcwd() + "/data/Inventory_Turnover.csv"
data2_path = getcwd() + "/data/Restock_data.csv"

# Load your datasets
inv_df = pd.read_csv(filepath_or_buffer=data1_path)
res_df = pd.read_csv(filepath_or_buffer=data2_path)

# Clean any hidden whitespace immediately after loading
inv_df.columns = inv_df.columns.str.strip()
res_df.columns = res_df.columns.str.strip()

# ----------------------
# INVENTORY DATA CLEANING
# ----------------------
inv_df['dispense_date'] = pd.to_datetime(inv_df['dispense_date'], errors='coerce')
inv_df.drop_duplicates(inplace=True)

# Check for nulls
print("Inventory missing values:")
print(inv_df.isnull().sum())

# ----------------------
# RESTOCK DATA CLEANING
# ----------------------
res_df['restock_date'] = pd.to_datetime(res_df['restock_date'], errors='coerce')
res_df.drop_duplicates(inplace=True)

# Check for nulls
print("\nRestock missing values:")
print(res_df.isnull().sum())

# Remove any zero or negative restocks
res_df = res_df[res_df['total'] > 0]

# ----------------------

# STEP2: Initial EDA

# ------------------------
# High-level stats
# ------------------------
print("\nInventory Summary:")
print(inv_df.describe(include='all'))

print("\nRestock Summary:")
print(res_df.describe(include='all'))

# ------------------------
# Time Series: Overall demand
# ------------------------

# Set up plot saving
output_dir = "new_plots"
os.makedirs(output_dir, exist_ok=True)

# daily_demand = inv_df.groupby('dispense_date')['qty_dispensed'].sum()

# plt.figure(figsize=(12, 4))
# daily_demand.plot()
# plt.title("Total Quantity Dispensed Over Time")
# plt.xlabel("Date")
# plt.ylabel("Qty Dispensed")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "total_quantity_dispensed_over_time.pdf"))
# plt.show()

monthly_demand = inv_df.set_index('dispense_date').resample('ME')['qty_dispensed'].sum()

plt.figure(figsize=(12, 4))
monthly_demand.plot(marker='o')
plt.title("Monthly Dispensed Quantity (Resampled)")
plt.ylabel("Qty Dispensed")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "monthly_dispensed_quantity.pdf"))
# plt.show()

# ------------------------
# Device usage ranking
# ------------------------

device_usage = inv_df.groupby('device_id')['qty_dispensed'].sum().sort_values(ascending=False)

# # Highlight top SKU and device before plotting
# # Top device
# top_device = device_usage.idxmax()
# top_device_qty = device_usage.max()
# print(f"\nTop dispensing device: {top_device} ({top_device_qty:,} units dispensed)")

#plot
# plt.figure(figsize=(10, 5))
# sns.barplot(x=device_usage.head(10).index, y=device_usage.head(10).values)
# plt.xticks(rotation=45)
# plt.title("Top 10 Devices by Dispensed Quantity")
# plt.ylabel("Qty Dispensed")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "top_10_devices_by_dispense_quantity.pdf"))
# plt.show()

# ----------------------------

# STEP3. Step-by-Step Feature Engineering

# Engineer time-based and lag-based features to help a forecasting 
# model understand when and how much demand changes.

# Make sure the date column is datetime
inv_df['dispense_date'] = pd.to_datetime(inv_df['dispense_date'])

# Time-based features
inv_df['day_of_week'] = inv_df['dispense_date'].dt.dayofweek  # 0=Mon, 6=Sun
inv_df['is_weekend'] = inv_df['day_of_week'].isin([5, 6]).astype(int)
inv_df['month'] = inv_df['dispense_date'].dt.month
inv_df['weekofyear'] = inv_df['dispense_date'].dt.isocalendar().week
inv_df['day_of_month'] = inv_df['dispense_date'].dt.day
inv_df['year'] = inv_df['dispense_date'].dt.year

# Lag Features
# Use past data to inform future behavior — critical for forecasting.
# Daily SKU demand (SKU = Stock Keeping Unit)
sku_daily = inv_df.groupby(['sku', 'dispense_date'])['qty_dispensed'].sum().reset_index()

# Sort before lagging
sku_daily.sort_values(by=['sku', 'dispense_date'], inplace=True)

# Lag features: previous 1, 3, and 7 days' demand
# This gives us both short-term memory (lag_1, lag_3 rolling_3) and weekly patterns (lag_7, rolling_7).
sku_daily['lag_1'] = sku_daily.groupby('sku')['qty_dispensed'].shift(1)
sku_daily['lag_3'] = sku_daily.groupby('sku')['qty_dispensed'].shift(3)
sku_daily['lag_7'] = sku_daily.groupby('sku')['qty_dispensed'].shift(7)

# Rolling mean
sku_daily['rolling_3'] = sku_daily.groupby('sku')['qty_dispensed'].transform(lambda x: x.shift(1).rolling(3).mean())
sku_daily['rolling_7'] = sku_daily.groupby('sku')['qty_dispensed'].transform(lambda x: x.shift(1).rolling(7).mean())

# Encode Demand Volatility (to flag unstable demand)
# Use rolling standard deviation to understand SKU demand instability: 
sku_daily['rolling_std_7'] = sku_daily.groupby('sku')['qty_dispensed'].transform(lambda x: x.shift(1).rolling(7).std())

# Combine Back with Inventory Metadata, (if needed later)
# Merge back with inv_df
inv_df = inv_df.merge(sku_daily, on=['sku', 'dispense_date'], how='left')

# print("Inventory columns:", inv_df.columns.tolist())
# print("Restock columns:", res_df.columns.tolist())

# Inventory columns: ['sku', 'dispense_date', 'device_id', 'package_qty', 
# 'qty_dispensed_x', 'day_of_week', 'is_weekend', 'month', 'weekofyear', 
# 'day_of_month', 'year', 'qty_dispensed_y', 'lag_1', 'lag_3', 'lag_7', 
# 'rolling_3', 'rolling_7', 'rolling_std_7']

# Fix duplicated qty_dispensed columns
inv_df['qty_dispensed'] = inv_df['qty_dispensed_x'].combine_first(inv_df['qty_dispensed_y'])
inv_df.drop(columns=['qty_dispensed_x', 'qty_dispensed_y'], inplace=True)

print(inv_df[['package_qty', 'qty_dispensed']].describe())

# -----------------------

# STEP4: Descriptive Statistics for both datasets

# a. Summary for Numerical Columns
print("\nInventory - Numerical Summary:")
print(inv_df[['package_qty', 'qty_dispensed']].describe())

print("\nRestock - Numerical Summary:")
print(res_df[['total']].describe())

# b. Frequency Count for Categorical Columns
print("\nInventory - Top SKUs:")
print(inv_df['sku'].value_counts().head(5))

print("\nInventory - Top Devices:")
print(inv_df['device_id'].value_counts().head(5))

print("\nRestock - Currency Codes:")
print(res_df['currency_code'].value_counts())

# c. Visual Summary: Boxplots to Detect Outliers

# Inventory Dispense Quantity
plt.figure(figsize=(8, 4))
sns.boxplot(x=inv_df['qty_dispensed'])
plt.title("Boxplot: Quantity Dispensed")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_qty_dispensed.pdf"))
# plt.show()

# Restock Total
plt.figure(figsize=(8, 4))
sns.boxplot(x=res_df['total'])
plt.title("Boxplot: Restock Total Value")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_restock_total.pdf"))
# plt.show()

# d. Quantile Summary
#Quantiles help you decide thresholds (e.g., for defining outliers):

print("\nInventory - Dispense Quantiles:")
print(inv_df['qty_dispensed'].quantile([0.01, 0.25, 0.5, 0.75, 0.99]))

print("\nRestock - Total Quantiles:")
print(res_df['total'].quantile([0.01, 0.25, 0.5, 0.75, 0.99]))

# ----------------------------

# STEP5: Monthly Trends – Dispense & Restock

# a. Monthly Dispensed Quantity
monthly_dispense = inv_df.set_index('dispense_date').resample('ME')['qty_dispensed'].sum()

# Monthly Restock Cost (or Quantity if you have that instead)
monthly_restock = res_df.set_index('restock_date').resample('ME')['total'].sum()

# b. Plot the Trends Together
plt.figure(figsize=(14, 6))
monthly_dispense.plot(label='Total Dispensed', marker='o', color='skyblue')
monthly_restock.plot(label='Total Restocked (Cost)', marker='s', color='salmon')

plt.title("Monthly Trends: Dispense vs. Restock")
plt.xlabel("Month")
plt.ylabel("Volume / Cost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "monthly_trends_dispense_vs_restock.pdf"))
# plt.show()

# c. Seasonality by Month or Weekday

# 1. Average Quantity Dispensed by Month
month_avg = inv_df.groupby('month')['qty_dispensed'].mean()

plt.figure(figsize=(10, 4))
month_avg.plot(kind='bar', color='skyblue')
plt.title("Average Dispensed Quantity by Month")
plt.xlabel("Month")
plt.ylabel("Avg. Qty Dispensed")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "seasonality_monthly_avg_dispensed.pdf"))
# plt.show()

# 2. Average Quantity Dispensed by Day of Week

weekday_avg = inv_df.groupby('day_of_week')['qty_dispensed'].mean()
weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.figure(figsize=(8, 4))
sns.barplot(x=weekday_labels, y=weekday_avg.values, hue=weekday_labels, palette='coolwarm', legend=False)
plt.title("Average Dispensed Quantity by Day of Week")
plt.ylabel("Avg. Qty Dispensed")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "seasonality_weekday_avg_dispensed.pdf"))
# plt.show()

# 3. Highlight Anomalies or Peaks
# We could also use a rolling window (already in rolling_7) 
# to plot smoothed trends and spikes:

# Smoothed rolling view for one SKU
sku_id = inv_df['sku'].value_counts().idxmax()  # most common SKU
sku_view = inv_df[inv_df['sku'] == sku_id].set_index('dispense_date')

plt.figure(figsize=(12, 5))
sku_view['qty_dispensed'].plot(label='Actual')
sku_view['rolling_7'].plot(label='7-Day Rolling Avg', linestyle='--')
plt.title(f"Dispense Trend for SKU {sku_id}")
plt.ylabel("Qty Dispensed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"sku_{sku_id}_trend.pdf"))
# plt.show()

# d. Summary Table Export
# export monthly summary data:

# Combine monthly dispense and restock
monthly_summary = pd.concat([monthly_dispense, monthly_restock], axis=1).reset_index()
monthly_summary.columns = ['Month', 'Total_Dispensed', 'Total_Restock']

# Save to Excel inside output_dir

# Change filename if necessary
filename = "monthly_trend_summary.xlsx"
filepath = os.path.join(output_dir, filename)

# Check if file exists, and remove it first
if os.path.exists(filepath):
    os.remove(filepath)

# Now export
monthly_summary.to_excel(filepath, index=False)

print(f"Monthly summary exported to: {filepath}")


# -----------------

# STEP6: Product Analysis: 
# Determine which SKUs have the highest turnover and 
# identify any seasonality in their sales.

# To perform Product Analysis
# - identifying high-turnover SKUs and detecting seasonality in their sales -
# we can follow these steps:

# a. Identify High Turnover SKUs
# Definition:
# Turnover = Quantity Dispensed / Quantity Restocked (simplified, per SKU)

# Create day-only fields to join on
inv_df['dispense_day'] = inv_df['dispense_date'].dt.date
res_df['restock_day'] = res_df['restock_date'].dt.date

# Merge on device_id + Day
# This assumes restocking happens before dispensing (same day).

merged_df = pd.merge(
    inv_df,
    res_df[['device_id', 'restock_day', 'total']],
    left_on=['device_id', 'dispense_day'],
    right_on=['device_id', 'restock_day'],
    how='left'
)

# Group by SKU and aggregate total dispensed and restocked quantities
sku_summary = merged_df.groupby('sku').agg(
    total_dispensed=('qty_dispensed', 'sum'),
    total_restocked=('total', 'sum')
).reset_index()

# Calculate turnover
# Avoid division by zero by using np.where or filling zero restock with NaN
import numpy as np

sku_summary['turnover'] = np.where(
    sku_summary['total_restocked'] > 0,
    sku_summary['total_dispensed'] / sku_summary['total_restocked'],
    np.nan  # Or 0 if you prefer
)
# Sort by turnover descending
sku_summary_sorted = sku_summary.sort_values(by='turnover', ascending=False)

sku_summary_sorted.to_excel(os.path.join(output_dir, 'sku_turnover_summary.xlsx'), index=False)

# b. Top 10 SKUs by Highest Turnover

top_turnover = sku_summary.sort_values('turnover', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_turnover,
    x='turnover',
    y='sku',
    hue='sku',  # Assign hue for color mapping
    palette='crest',
    dodge=False,
    legend=False  # Hide duplicate legend since y-axis already labels SKU
)
plt.title('Top 10 SKUs by Highest Turnover')
plt.xlabel('Turnover (Dispensed / Restocked)')
plt.ylabel('SKU')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top10_skus_turnover.png'))
plt.close()

# Save data
top_turnover.to_csv(os.path.join(output_dir, 'top10_skus_turnover.csv'), index=False)

# Top 10 SKUs by Highest Estimated Restocks

top_restocked = sku_summary.sort_values('total_restocked', ascending=False).head(10)

# Save plot
plt.figure(figsize=(10, 6))
sns.barplot(x='total_restocked', y='sku', hue='sku', data=top_restocked, palette='flare')
plt.title('Top 10 SKUs by Total Restocked Quantity')
plt.xlabel('Total Quantity Restocked')
plt.ylabel('SKU')
plt.tight_layout()
restock_plot_path = os.path.join(output_dir, 'top10_skus_restocked.png')
plt.savefig(restock_plot_path)
plt.close()

# Save data
top_restocked.to_csv(os.path.join(output_dir, 'top10_skus_restocked.csv'), index=False)

# Monthly Dispensing Trends for Top 5 SKUs (Seasonality)
top_5_skus = top_turnover['sku'].head(5).tolist()

# Prepare monthly trends data
monthly_trends = (
    merged_df[merged_df['sku'].isin(top_5_skus)]
    .groupby(['sku', 'month'])['qty_dispensed']
    .sum()
    .reset_index()
)

# Save plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_trends, x='month', y='qty_dispensed', hue='sku', marker='o')
plt.title('Monthly Dispensing Trend (Top 5 SKUs by Turnover)')
plt.xlabel('Month')
plt.ylabel('Quantity Dispensed')
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
seasonality_plot_path = os.path.join(output_dir, 'monthly_trends_top5_skus.png')
plt.savefig(seasonality_plot_path)
plt.close()

# Save data
monthly_trends.to_csv(os.path.join(output_dir, 'top5_sku_monthly_trends.csv'), index=False)

# combined plot
# Set style
sns.set(style='whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('SKU-Level Vending Report Summary', fontsize=18, fontweight='bold')

# -------------------------------
# Plot 1: Top 10 SKUs by Turnover
# -------------------------------
sns.barplot(
    data=top_turnover,
    x='turnover',
    y='sku',
    hue='sku',
    palette='crest',
    dodge=False,
    legend=False,
    ax=axes[0, 0]
)
axes[0, 0].set_title('Top 10 SKUs by Turnover')
axes[0, 0].set_xlabel('Turnover (Dispensed / Restocked)')
axes[0, 0].set_ylabel('SKU')

# --------------------------------------
# Plot 2: Top 10 SKUs by Total Restocked
# --------------------------------------
sns.barplot(
    data=top_restocked,
    x='total_restocked',
    y='sku',
    hue='sku',
    palette='flare',
    dodge=False,
    legend=False,
    ax=axes[0, 1]
)
axes[0, 1].set_title('Top 10 SKUs by Total Restocked')
axes[0, 1].set_xlabel('Total Restocked')
axes[0, 1].set_ylabel('SKU')

# --------------------------------------
# Plot 3: Monthly Trend for Top 5 SKUs
# --------------------------------------
sns.lineplot(
    data=monthly_trends,
    x='month',
    y='qty_dispensed',
    hue='sku',
    marker='o',
    palette='tab10',
    ax=axes[1, 0]
)
axes[1, 0].set_title('Monthly Dispensing Trend (Top 5 SKUs)')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Quantity Dispensed')
axes[1, 0].set_xticks(range(1, 13))

# Hide unused subplot (axes[1, 1])
axes[1, 1].axis('off')

# Layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
combined_path = os.path.join(output_dir, 'sku_report_summary.png')
plt.savefig(combined_path, dpi=300)
plt.close()

#------------------------------

# STEP7: Correlation Analysis: Explore correlations 
# between the quantity dispensed and restock frequency or cost.

# a. Aggregate Total Dispensed and Total Restock Cost per SKU
# Was already done above

#/*
# Group by SKU and aggregate total dispensed and restocked quantities
# sku_summary = merged_df.groupby('sku').agg(
#     total_dispensed=('qty_dispensed', 'sum'),
#     total_restocked=('total', 'sum')
# ).reset_index()
# */


# b. Aggregate Restock Frequency per SKU 

# Count how often each SKU-device was restocked (proxy for frequency)
restock_freq = merged_df.groupby('sku')['restock_day'].nunique().reset_index(name='restock_frequency')

# Merge frequency
merged_summary_freq = sku_summary.merge(restock_freq, on='sku', how='left')

# c. Calculate Correlation Matrix
# Compute correlation matrix
correlation_matrix = merged_summary_freq[['total_dispensed', 'total_restocked', 'restock_frequency']].corr()
print(correlation_matrix)
# Define file path
corr_excel_path = os.path.join(output_dir, 'correlation_matrix.xlsx')
# Save to Excel
correlation_matrix.round(3).to_excel(corr_excel_path)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title('Correlation Between Dispensed Quantity, Restock Frequency, and Total', fontsize=14)
plt.tight_layout()

# Save to output_dir
corr_plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
plt.savefig(corr_plot_path, dpi=300)
plt.close()

# Interpretation Ideas:
# High positive correlation between total_dispensed and restock_frequency? 
# → consistent demand triggers more restocking.

# High correlation between dispensed and restock_cost? 
# → cost likely scales with quantity.

#  d. Pairplot or Regression lines for visual inspection per SKU group

# Option 1: sns.pairplot() — For visualizing multiple numeric relationships per

# print("merged_df columns:", merged_df.columns.tolist())

# 1.1 Add total_restocked and turnover to merged_df

# Total restocked per SKU/device/date from the merged 'res_df' (joined already)
merged_df['total_restocked'] = merged_df['total'] 

# Avoid divide-by-zero: Replace 0 restocked with NaN for clean turnover
merged_df['turnover'] = merged_df['qty_dispensed'] / merged_df['total_restocked'].replace(0, np.nan)

# 1.2 Filter top SKUs by volume or turnover and create pairplot

# Top N SKUs by quantity dispensed
N = 5
top_skus = (
    merged_df.groupby('sku')['qty_dispensed']
    .sum()
    .nlargest(N)
    .index
)

# Filter dataset
pairplot_df = merged_df[merged_df['sku'].isin(top_skus)].copy()

# Ensure needed columns are present
pairplot_cols = ['sku', 'qty_dispensed', 'total_restocked', 'turnover']
pairplot_df = pairplot_df[pairplot_cols].dropna()

# Generate pairplot with SKU as hue
sns.pairplot(pairplot_df, hue='sku', corner=True, plot_kws={'alpha': 0.6})

# Save plot to output_dir
pairplot_path = os.path.join(output_dir, 'top_skus_pairplot.png')
plt.savefig(pairplot_path, bbox_inches='tight')
plt.close()

# Option 2: sns.lmplot() — For scatterplots with regression lines by SKU

# Example: Regression of qty_dispensed vs. total_restocked
top_skus = merged_df.groupby('sku')['turnover'].mean().nlargest(5).index.tolist()
sku_subset = merged_df[merged_df['sku'].isin(top_skus)]

sns.lmplot(
    data=sku_subset,
    x='total_restocked',
    y='qty_dispensed',
    hue='sku',
    col='sku',
    col_wrap=3,
    height=4,
    scatter_kws={'alpha': 0.6},
    line_kws={'color': 'red'}
)

# Save regression plots
regplot_path = os.path.join(output_dir, 'sku_lmplot.png')
plt.savefig(regplot_path, bbox_inches='tight')
plt.close()

# Updated Plot Code with log transformation applied manually:

# Step 1: Get top SKUs by turnover
sku_turnover = merged_df.groupby('sku')['turnover'].mean()
top_skus = sku_turnover.nlargest(5).index.tolist()

# Step 2: Filter and prepare data
plot_df = merged_df[merged_df['sku'].isin(top_skus)].copy()
plot_df = plot_df[['sku', 'qty_dispensed', 'total_restocked']].dropna()

# Step 3: Filter out zero/negative values and log-transform
plot_df = plot_df[(plot_df['qty_dispensed'] > 0) & (plot_df['total_restocked'] > 0)]
plot_df['log_qty_dispensed'] = np.log10(plot_df['qty_dispensed'])
plot_df['log_total_restocked'] = np.log10(plot_df['total_restocked'])

# Step 4: Plot regression in log space
sns.set(style="whitegrid")
g = sns.lmplot(
    data=plot_df,
    x='log_total_restocked', y='log_qty_dispensed',
    col='sku',
    hue='sku',
    height=4, aspect=1.2,
    scatter_kws={'alpha': 0.5}
)
g.fig.suptitle("Log-Log Regression: Dispensed vs Restocked (Top SKUs)", y=1.05)
plt.tight_layout()

# Step 5: Save plot
plot_path = os.path.join(output_dir, "sku_regression_log_transformed.png")
plt.savefig(plot_path)
plt.close()
print(f"Log-transformed regression plot saved to: {plot_path}")


# ------------------

# STEP8: Seasonality and Patterns: Investigate any seasonal patterns 
# or cyclic behaviour in dispensing and restocking activities.

# The merged_df is well-equipped for seasonality and cyclic 
# pattern analysis across dispensing and restocking activities

# GOAL:
# Analyze patterns over:
# Months (seasonality)
# Week of Year (cyclical behavior)
# Compare dispensed vs restocked quantities

# a.  Aggregate by Month & Week of Year

# Monthly summary: Dispensed and Restocked
monthly_summary = merged_df.groupby(['year', 'month'])[['qty_dispensed', 'total_restocked']].sum().reset_index()
monthly_summary['date'] = pd.to_datetime(monthly_summary[['year', 'month']].assign(day=1))

# Weekly pattern: Week of Year
weekly_summary = merged_df.groupby(['year', 'weekofyear'])[['qty_dispensed', 'total_restocked']].sum().reset_index()
weekly_summary['week'] = weekly_summary['weekofyear'].astype(int)
weekly_summary['year_week'] = weekly_summary['year'].astype(str) + '-W' + weekly_summary['week'].astype(str)

# b. Plot Monthly Seasonality

sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))

# Melt for Seaborn
monthly_melt = monthly_summary.melt(id_vars='date', value_vars=['qty_dispensed', 'total_restocked'], var_name='type', value_name='quantity')
monthly_melt['type'] = monthly_melt['type'].map({'qty_dispensed': 'Dispensed', 'total_restocked': 'Restocked'})

sns.lineplot(data=monthly_melt, x='date', y='quantity', hue='type', palette='Set2')
plt.title("Monthly Seasonality: Dispensed vs Restocked")
plt.xlabel("Month")
plt.ylabel("Quantity")
plt.xticks(rotation=45)
plt.tight_layout()

# Save
month_plot_path = os.path.join(output_dir, "monthly_seasonality.png")
plt.savefig(month_plot_path)
plt.close()

# c. Plot Weekly Cyclical Patterns

plt.figure(figsize=(12, 6))
weekly_melt = weekly_summary.melt(id_vars='weekofyear', value_vars=['qty_dispensed', 'total_restocked'], var_name='type', value_name='quantity')
weekly_melt['type'] = weekly_melt['type'].map({'qty_dispensed': 'Dispensed', 'total_restocked': 'Restocked'})

sns.lineplot(data=weekly_melt, x='weekofyear', y='quantity', hue='type', palette='coolwarm')
plt.title("Cyclic Patterns by Week of Year")
plt.xlabel("Week of Year")
plt.ylabel("Quantity")
plt.tight_layout()

# Save
week_plot_path = os.path.join(output_dir, "weekly_cyclic_patterns.png")
plt.savefig(week_plot_path)
plt.close()

# d. Day of Week Analysis
# Understand if vending and restocking are more common on certain days.

# Aggregate by day of week
dow_trend = merged_df.groupby('day_of_week')[['qty_dispensed', 'total_restocked']].mean().reset_index()
dow_trend_melted = dow_trend.melt(id_vars='day_of_week', var_name='type', value_name='avg_qty')

# d.1 Plot
plt.figure(figsize=(10, 4))
sns.barplot(data=dow_trend_melted, x='day_of_week', y='avg_qty', hue='type', palette='Set2')
plt.title('Day-of-Week Patterns: Dispense & Restock')
plt.xlabel('Day of Week (0=Mon, 6=Sun)')
plt.tight_layout()

dow_path = os.path.join(output_dir, 'seasonality_dow_dispense_restock.png')
plt.savefig(dow_path)
plt.close()

# Prepare data
dow_dispensed = merged_df.groupby('day_of_week')['qty_dispensed'].mean()
dow_restocked = merged_df.groupby('day_of_week')['total_restocked'].mean()

# d.2 Plot with twin y-axes: This keeps both datasets on their natural scale but separates them visually.
fig, ax1 = plt.subplots(figsize=(10, 5))

# First axis (dispensed)
ax1.set_title('Day-of-Week Patterns: Dispensed vs Restocked')
ax1.set_xlabel('Day of Week (0=Mon, 6=Sun)')
ax1.set_ylabel('Avg Quantity Dispensed', color='tab:blue')
ax1.bar(dow_dispensed.index - 0.2, dow_dispensed.values, width=0.4, color='tab:blue', label='Dispensed')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Second axis (restocked)
ax2 = ax1.twinx()
ax2.set_ylabel('Avg Quantity Restocked', color='tab:orange')
ax2.bar(dow_restocked.index + 0.2, dow_restocked.values, width=0.4, color='tab:orange', label='Restocked')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Save
plt.tight_layout()
dual_axis_path = os.path.join(output_dir, 'seasonality_dow_dispense_restock_dual_axis.png')
plt.savefig(dual_axis_path)
plt.close()

# d.3 Normalize or Scale the Data (e.g., MinMax or Z-Score): 
# If the trend is more important than the actual magnitude, 
# you can rescale both series to make them visually comparable.

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Combine data
dow = merged_df.groupby('day_of_week')[['qty_dispensed', 'total_restocked']].mean().reset_index()

# Min-max scaling
scaler = MinMaxScaler()
dow_scaled = pd.DataFrame(scaler.fit_transform(dow[['qty_dispensed', 'total_restocked']]),
                          columns=['qty_dispensed_scaled', 'total_restocked_scaled'])
dow_scaled['day_of_week'] = dow['day_of_week']

# Melt and plot
dow_melted = dow_scaled.melt(id_vars='day_of_week', var_name='type', value_name='scaled_qty')

plt.figure(figsize=(10, 5))
sns.barplot(data=dow_melted, x='day_of_week', y='scaled_qty', hue='type', palette='coolwarm')
plt.title('Day-of-Week Patterns (Scaled): Dispensed vs Restocked')
plt.ylabel('Scaled Quantity (0–1)')
plt.xlabel('Day of Week (0=Mon, 6=Sun)')
plt.tight_layout()

# Save
scaled_path = os.path.join(output_dir, 'seasonality_dow_dispense_restock_scaled.png')
plt.savefig(scaled_path)
plt.close()

# e. Heatmap of Monthly Dispense by SKU (if >1 year of data)

# Create a pivot: sku x month-year
merged_df['month_str'] = merged_df['year'].astype(str) + '-' + merged_df['month'].astype(str).str.zfill(2)
sku_monthly = merged_df.groupby(['sku', 'month_str'])['qty_dispensed'].sum().unstack(fill_value=0)

plt.figure(figsize=(14, 8))
sns.heatmap(sku_monthly, cmap="YlGnBu", linewidths=0.5)
plt.title("Heatmap: Monthly Dispensed Volume per SKU")
plt.xlabel("Month")
plt.ylabel("SKU")
plt.tight_layout()

# Save
heatmap_path = os.path.join(output_dir, "sku_monthly_dispense_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

#  f. Apply Rolling Averages to Smooth Dispense Trends
# We already have:
# lag_1, lag_3, lag_7: Lag features — past values for 1, 3, or 7 days
# rolling_3, rolling_7: Smoothed averages over 3 and 7-day windows
# We can visualize these to smooth noisy daily patterns.

# Example: Plot with Rolling Average

sku_to_plot = 'be61be55295db1941eaf232ee6288fa7'  # Choose a sample SKU from top10_skus_turnover.csv
sku_df = merged_df[merged_df['sku'] == sku_to_plot]

plt.figure(figsize=(14, 5))
sns.lineplot(data=sku_df, x='dispense_date', y='qty_dispensed', label='Daily Dispense')
sns.lineplot(data=sku_df, x='dispense_date', y='rolling_7', label='7-Day Rolling Avg', linestyle='--')
plt.title(f"Dispensing Trend for sku_88fa7 with Rolling Average")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"rolling_avg_sku_88fa7.png"))
plt.close()

# g. Annotate Holidays or Events
# We can define holidays/events manually and annotate them on trend plots.

from datetime import datetime

holidays = [
    ("New Year", "2023-01-01"),
    ("Easter", "2023-03-31"),
    ("Christmas", "2023-12-25"),
]

sku_df = merged_df[merged_df['sku'] == sku_to_plot]

plt.figure(figsize=(14, 5))
sns.lineplot(data=sku_df, x='dispense_date', y='qty_dispensed', label='Daily Dispense')

for name, date_str in holidays:
    date_obj = pd.to_datetime(date_str)
    plt.axvline(date_obj, color='red', linestyle='--')
    plt.text(date_obj, plt.ylim()[1]*0.95, name, rotation=90, color='red')

plt.title(f"sku_88fa7 Dispense with Holiday Annotations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"holiday_annotations_sku_88fa7.png"))
plt.close()

# --------------------------

# STEP9: Demand Variability by SKU: Investigate the variability in demand for each SKU over time 
# to identify high-variance items that may require special attention in inventory planning.

# To investigate demand variability by SKU, we can analyze the variance or standard deviation of 
# qty_dispensed over time. High variability often signals SKUs that may require buffer stock, 
# forecasting adjustments, or manual review for restocking strategies.

# 1. Calculate Variability Metrics per SKU

# We’ll compute:
# Mean quantity dispensed
# Standard deviation (std)
# Coefficient of variation (CV = std / mean) — this normalizes the variability

# Calculate SKU-level demand variability
demand_variability = merged_df.groupby('sku')['qty_dispensed'].agg(
    mean_dispensed='mean',
    std_dispensed='std'
).reset_index()

# Add coefficient of variation
demand_variability['cv_dispensed'] = demand_variability['std_dispensed'] / demand_variability['mean_dispensed']
demand_variability = demand_variability.sort_values(by='cv_dispensed', ascending=False)

# Save to file
demand_variability.to_csv(os.path.join(output_dir, "sku_demand_variability.csv"), index=False)

# 2. Visualize Top N High-Variance SKUs

top_var_skus = demand_variability.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_var_skus,
    x='sku',
    y='cv_dispensed',
    hue='sku',
    palette='viridis'
)
plt.title("Top 10 SKUs by Demand Variability (Coefficient of Variation)")
plt.ylabel("CV of Quantity Dispensed")
plt.xlabel("SKU")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top10_demand_variability_skus.png"))

# 3. Plot Time Series for Top Variable SKUs

top_skus = top_var_skus['sku'].tolist()

for sku in top_skus:
    sku_data = merged_df[merged_df['sku'] == sku].sort_values('dispense_date')

    plt.figure(figsize=(14, 4))
    sns.lineplot(x='dispense_date', y='qty_dispensed', data=sku_data, label='Dispensed Qty')
    sns.lineplot(x='dispense_date', y='rolling_7', data=sku_data, label='7-Day Rolling Avg', linestyle='--')

    plt.title(f"Demand Trend for High-Variance SKU: {sku}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"timeseries_high_var_{sku}.png"))
    plt.close()

# Interpretation
# Metric	        Meaning
# std_dispensed	    Raw variability in how much is dispensed
# cv_dispensed	    Normalized variability (independent of average level)
# High CV (>1.0)	Indicates very unstable demand; review for special inventory rules

# -----------------------

# STEP10: Seasonality in SKU Demand: Perform a seasonality analysis for individual SKUs 
# to discover any cyclical demand patterns, which can inform restocking strategies.

# To uncover seasonality in SKU demand, we'll analyze how the quantity dispensed (qty_dispensed) 
# varies across time periods like month, week of year, or day of week, per SKU.

# 1. Aggregate Demand by Month for Each SKU

# Group by SKU and Month to find average monthly demand
sku_monthly = merged_df.groupby(['sku', 'month'])['qty_dispensed'].mean().reset_index()
sku_monthly.rename(columns={'qty_dispensed': 'avg_monthly_demand'}, inplace=True)

# 2. Plot Monthly Seasonality for Top N SKUs

# Choose top N SKUs by volume or turnover, then generate seasonality plots:

# Get top SKUs by average volume
top_skus_by_volume = (
    merged_df.groupby('sku')['qty_dispensed']
    .sum()
    .sort_values(ascending=False)
    .head(5)  # choose N
    .index.tolist()
)

# Filter for top SKUs
top_sku_monthly = sku_monthly[sku_monthly['sku'].isin(top_skus_by_volume)]

# Seasonal Line Plot by Month

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=top_sku_monthly,
    x='month',
    y='avg_monthly_demand',
    hue='sku',
    palette='tab10',
    marker='o'
)
plt.title("Monthly Seasonality of SKU Demand")
plt.xlabel("Month")
plt.ylabel("Avg Quantity Dispensed")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sku_seasonality_by_month.png"))

# Insights & Next Steps
# Seasonal Pattern	         Strategy
# Peaks in certain months	 Pre-stock or accelerate restocks in peak periods
# Dips in low months	     Avoid overstock, reduce restocking frequency
# Highly volatile SKUs	     Combine seasonality with variability analysis for alerts

# ----------------------

# STEP11: Device Utilization: Analyze the utilization rate of each
# vending machine device to identify underused or overused machines, 
# which could indicate issues with machine placement or inventory selection.

# To analyze device utilization in your vending machine data, we’ll assess 
# how actively each device_id dispenses items over time. This helps identify 
# underused or overused machines, which can guide decisions on:
# Placement optimization
# Inventory tailoring
# Maintenance or replacement scheduling

# Key Metric: Device Utilization Rate
# We can define utilization rate in a few ways, for example:

# Option 1: Total quantity dispensed per device

device_usage = merged_df.groupby('device_id')['qty_dispensed'].sum().reset_index()
device_usage.rename(columns={'qty_dispensed': 'total_dispensed'}, inplace=True)

# Option 2: Average quantity dispensed per active day

# Count unique dispensing days per device
active_days = merged_df.groupby('device_id')['dispense_date'].nunique().reset_index()
active_days.rename(columns={'dispense_date': 'active_days'}, inplace=True)

# Combine with total quantity dispensed
device_usage = device_usage.merge(active_days, on='device_id')
device_usage['avg_dispensed_per_day'] = device_usage['total_dispensed'] / device_usage['active_days']

# Option 3: Utilization as a percentage of busiest machine

max_dispensed = device_usage['total_dispensed'].max()
device_usage['utilization_pct'] = (device_usage['total_dispensed'] / max_dispensed) * 100

# Plotting Device Utilization

# 1. Barplot of Total Dispensed per Device
plt.figure(figsize=(14, 6))
sns.barplot(
    data=device_usage.sort_values('total_dispensed', ascending=False),
    x='device_id',
    y='total_dispensed',
    hue='device_id',
    palette='viridis'
)
plt.title("Device Utilization: Total Items Dispensed")
plt.xticks(rotation=45)
plt.ylabel("Total Quantity Dispensed")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "device_utilization_total_dispensed.png"))

# 2. Barplot of Average Dispensed per Active Day
plt.figure(figsize=(14, 6))
sns.barplot(
    data=device_usage.sort_values('avg_dispensed_per_day', ascending=False),
    x='device_id',
    y='avg_dispensed_per_day',
    hue='device_id',
    palette='coolwarm'
)
plt.title("Device Utilization: Avg Dispensed per Active Day")
plt.xticks(rotation=45)
plt.ylabel("Avg Qty per Day")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "device_utilization_avg_per_day.png"))

# Insights You Can Draw:
# Utilization Pattern	      Interpretation
# High total + high avg	      High-traffic device; consider scaling inventory
# Low total + low avg	      Possibly poor placement or irrelevant SKUs
# Low total but high avg	  Rarely used but intense demand on certain days
# High total but low avg	  Frequently used, but lightly each day