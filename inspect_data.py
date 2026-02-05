"""
Temporary script to inspect parquet data structure.
This helps us understand the data format before creating production config.

Usage:
    python inspect_data.py
"""

import pandas as pd
from pathlib import Path

# Load calibration data (smallest file)
data_path = Path("data/calibration/us30_calibration.parquet")

print("="*70)
print("DATA INSPECTION")
print("="*70)
print(f"\nFile: {data_path}")
print(f"Size: {data_path.stat().st_size / 1024 / 1024:.2f} MB")

# Read data
df = pd.read_parquet(data_path)

print(f"\n{'='*70}")
print("BASIC INFO")
print("="*70)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn names ({len(df.columns)} total):")
print("-"*70)
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    null_count = df[col].isnull().sum()
    print(f"{i:3d}. {col:30s} {str(dtype):15s} (nulls: {null_count})")

print(f"\n{'='*70}")
print("DATA TYPES SUMMARY")
print("="*70)
print(df.dtypes.value_counts())

print(f"\n{'='*70}")
print("SAMPLE ROWS (first 5)")
print("="*70)
print(df.head())

print(f"\n{'='*70}")
print("KEY STATISTICS")
print("="*70)

# Check for date column
date_cols = [col for col in df.columns if 'date' in col.lower()]
if date_cols:
    date_col = date_cols[0]
    print(f"\nDate column: {date_col}")
    df[date_col] = pd.to_datetime(df[date_col])
    print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"  Unique dates: {df[date_col].nunique()}")

# Check for group column
group_cols = [col for col in df.columns if 'group' in col.lower()]
if group_cols:
    group_col = group_cols[0]
    print(f"\nGroup column: {group_col}")
    print(f"  Unique groups: {df[group_col].nunique()}")
    print(f"  Rows per group: min={df.groupby(group_col).size().min()}, "
          f"max={df.groupby(group_col).size().max()}, "
          f"mean={df.groupby(group_col).size().mean():.1f}")

# Check for label column
label_cols = [col for col in df.columns if any(x in col.lower() for x in ['pnl', 'label', 'target', 'reward'])]
if label_cols:
    label_col = label_cols[0]
    print(f"\nLabel column: {label_col}")
    print(f"  Mean: {df[label_col].mean():.4f}")
    print(f"  Std:  {df[label_col].std():.4f}")
    print(f"  Min:  {df[label_col].min():.4f}")
    print(f"  Max:  {df[label_col].max():.4f}")