"""Data loading utilities."""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_data(path: str, date_col: str = "date", parse_dates: bool = True) -> pd.DataFrame:
    """
    Load preprocessed data from parquet or CSV file.
    
    Args:
        path: Path to data file
        date_col: Name of date column
        parse_dates: Whether to parse dates
        
    Returns:
        DataFrame with loaded data
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    # Load based on file extension
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix in ['.csv', '.txt']:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Parse dates
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    
    return df


def validate_data(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    label_col: str,
    feature_cols: list
) -> None:
    """
    Validate that data has required columns and structure.
    
    Args:
        df: DataFrame to validate
        date_col: Date column name
        group_col: Group column name
        label_col: Label column name
        feature_cols: List of feature column names
        
    Raises:
        ValueError: If validation fails
    """
    required_cols = [date_col, group_col, label_col] + feature_cols
    missing = set(required_cols) - set(df.columns)
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for nulls in critical columns
    critical_cols = [date_col, group_col, label_col]
    for col in critical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(f"Column {col} has {null_count} null values")
    
    # Check date ordering
    if not df[date_col].is_monotonic_increasing:
        print("Warning: Data is not sorted by date. Sorting now.")
        df.sort_values(date_col, inplace=True)
    
    print("Data validation passed")


def get_date_range(df: pd.DataFrame, date_col: str) -> tuple:
    """
    Get date range of data.
    
    Args:
        df: DataFrame
        date_col: Date column name
        
    Returns:
        Tuple of (start_date, end_date)
    """
    return df[date_col].min(), df[date_col].max()


def filter_by_date_range(
    df: pd.DataFrame,
    date_col: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter data by date range.
    
    Args:
        df: DataFrame to filter
        date_col: Date column name
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        
    Returns:
        Filtered DataFrame
    """
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    
    return df
