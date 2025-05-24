# 1). Data Cleaning and Preparation: Handle missing values, duplicate entries, 
# and data type conversions. Ensure the datasets are clean and ready for analysis.

# Step1: Load and Inspect Data
# Purpose: Confirm data structure, types, and initial cleanliness.

# Load data
import pandas as pd
import numpy as np

import sweetviz as sv
from ydata_profiling import ProfileReport

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

# Step-by-Step EDA

# Step1: Basic Dataset Overview

# Summary statistics
print(inventory_df.describe())
# min package_qty = 1, max package_qty = 2, mean = 1, std = 0.022
# min dispense_qty = 1, max dispense_qty = 162, mean = 8.15, std = 9.055
print(restock_df.describe())
# min total = 5, max total = 612, mean = 272.5, std = 101.077


# === Generate Sweetviz report for inventory_df ===
# print("Generating Sweetviz report for inventory_df...")
# inventory_report = sv.analyze(inventory_df)
# inventory_report.show_html("inventory_sweetviz_report.html")

# === Generate YData-Profiling report for inventory_df ===
# Transform the DataFrame into a Profile Report
# inventory_profile_report = ProfileReport(df=inventory_df, explorative=True, title='Inventory Analytics')
# inventory_profile_report.to_file('inventory_profile_report.html')

# import webbrowser
# webbrowser.open("inventory_profile_report.html")

# # === Generate Sweetviz report for restock_df ===
# print("Generating Sweetviz report for restock_df...")
# restock_report = sv.analyze(restock_df)
# restock_report.show_html("restock_sweetviz_report.html")

# === Generate YData-Profiling report for restock_df ===
# Transform the DataFrame into a Profile Report
restock_profile_report = ProfileReport(df=restock_df, explorative=True, title='Restock Analytics')
restock_profile_report.to_file('restock_profile_report.html')

import webbrowser
webbrowser.open("restock_profile_report.html")

