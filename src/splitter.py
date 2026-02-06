"""Walk-forward time series splitting."""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple
from datetime import timedelta


class WalkForwardSplitter:
    """
    Generate walk-forward train/test splits for time series data.
    
    Supports both rolling and expanding windows.
    """
    
    def __init__(
        self,
        initial_train_days: int = 365,
        train_window_days: int = 365,
        test_window_days: int = 30,
        retrain_frequency_days: int = 7,
        expanding_window: bool = False,
        max_training_window_days: int = 504
    ):
        """
        Initialize walk-forward splitter.
        
        Args:
            initial_train_days: Initial training period (in TRADING days, not calendar days)
            train_window_days: Training window size in trading days (if rolling)
            test_window_days: Test period length in TRADING days
            retrain_frequency_days: How often to create new splits (TRADING days)
            expanding_window: If True, use expanding window with max cap
            max_training_window_days: Maximum training window in TRADING days (cap for expanding)
        
        Note: All day counts refer to TRADING days (actual dates in data), 
              not calendar days. This handles weekends/holidays automatically.
        """
        self.initial_train_days = initial_train_days
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.retrain_frequency_days = retrain_frequency_days
        self.expanding_window = expanding_window
        self.max_training_window_days = max_training_window_days
    
    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "date"
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward splits based on actual trading days in data.
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Yields:
            Tuples of (train_df, test_df)
        """
        df = df.sort_values(date_col)
        unique_dates = pd.Series(df[date_col].unique()).sort_values().reset_index(drop=True)
        n_dates = len(unique_dates)
        
        # Find index where test period starts (after initial training period)
        if self.initial_train_days >= n_dates:
            return  # Not enough data
        
        test_start_idx = self.initial_train_days
        
        while test_start_idx < n_dates:
            # Define test window (exactly test_window_days trading days)
            test_end_idx = min(test_start_idx + self.test_window_days, n_dates)
            
            # Stop if we can't get a full test window
            if test_end_idx - test_start_idx < self.test_window_days:
                break
            
            # Define train window based on strategy
            if self.expanding_window:
                # Expanding window: grow from start, but cap at max_training_window_days
                train_start_idx = max(0, test_start_idx - self.max_training_window_days)
            else:
                # Rolling window: fixed size
                train_start_idx = max(0, test_start_idx - self.train_window_days)
            
            train_end_idx = test_start_idx
            
            # Get actual date boundaries
            train_start_date = unique_dates.iloc[train_start_idx]
            train_end_date = unique_dates.iloc[train_end_idx - 1]  # Inclusive
            test_start_date = unique_dates.iloc[test_start_idx]
            test_end_date = unique_dates.iloc[test_end_idx - 1]  # Inclusive
            
            # Filter dataframe by date ranges (inclusive on both ends)
            train_df = df[
                (df[date_col] >= train_start_date) & 
                (df[date_col] <= train_end_date)
            ]
            
            test_df = df[
                (df[date_col] >= test_start_date) & 
                (df[date_col] <= test_end_date)
            ]
            
            # Skip if either split is empty
            if len(train_df) == 0 or len(test_df) == 0:
                test_start_idx += self.retrain_frequency_days
                continue
            
            yield train_df, test_df
            
            # Move to next test window (by retrain_frequency trading days)
            test_start_idx += self.retrain_frequency_days
    
    def get_split_info(
        self,
        df: pd.DataFrame,
        date_col: str = "date"
    ) -> list:
        """
        Get information about all splits without loading data.
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            List of dicts with split information
        """
        # Count actual unique trading days
        train_trading_days = train_df[date_col].nunique()
        test_trading_days = test_df[date_col].nunique()
        
        info = {
            'split_id': i,
            'train_start': train_df[date_col].min(),
            'train_end': train_df[date_col].max(),
            'test_start': test_df[date_col].min(),
            'test_end': test_df[date_col].max(),
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_trading_days': train_trading_days,
            'test_trading_days': test_trading_days,
            'test_size': len(test_df),
            'train_days': (train_df[date_col].max() - train_df[date_col].min()).days,
            'test_days': (test_df[date_col].max() - test_df[date_col].min()).days
        }
        split_info.append(info)
        
        return split_info


def create_splitter(config) -> WalkForwardSplitter:
    """
    Create walk-forward splitter from config.
    
    Args:
        config: Configuration object
        
    Returns:
        WalkForwardSplitter instance
    """
    return WalkForwardSplitter(
        initial_train_days=config.walkforward.initial_train_days,
        train_window_days=config.walkforward.train_window_days,
        test_window_days=config.walkforward.test_window_days,
        retrain_frequency_days=config.walkforward.retrain_frequency_days,
        expanding_window=config.walkforward.expanding_window,
        max_training_window_days=config.walkforward.max_training_window_days
    )
