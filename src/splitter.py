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
        expanding_window: bool = False
    ):
        """
        Initialize walk-forward splitter.
        
        Args:
            initial_train_days: Initial training period length
            train_window_days: Training window size (if rolling)
            test_window_days: Test period length
            retrain_frequency_days: How often to create new splits
            expanding_window: If True, use all data up to test start
        """
        self.initial_train_days = initial_train_days
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.retrain_frequency_days = retrain_frequency_days
        self.expanding_window = expanding_window
    
    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "date"
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward splits.
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Yields:
            Tuples of (train_df, test_df)
        """
        df = df.sort_values(date_col)
        dates = df[date_col].unique()
        dates = np.sort(dates)
        
        # Minimum date for first test split
        min_date = dates[0]
        first_test_start = min_date + pd.Timedelta(days=self.initial_train_days)
        
        # Start from first test date
        current_test_start = first_test_start
        
        while current_test_start <= dates[-1]:
            test_end = current_test_start + pd.Timedelta(days=self.test_window_days)
            
            # Stop if test window goes beyond data
            if test_end > dates[-1]:
                break
            
            # Define train window
            if self.expanding_window:
                train_start = min_date
            else:
                train_start = current_test_start - pd.Timedelta(days=self.train_window_days)
            
            train_end = current_test_start
            
            # Get train and test data
            train_df = df[
                (df[date_col] >= train_start) & 
                (df[date_col] < train_end)
            ]
            
            test_df = df[
                (df[date_col] >= current_test_start) & 
                (df[date_col] < test_end)
            ]
            
            # Skip if either split is empty
            if len(train_df) == 0 or len(test_df) == 0:
                current_test_start += pd.Timedelta(days=self.retrain_frequency_days)
                continue
            
            yield train_df, test_df
            
            # Move to next test window
            current_test_start += pd.Timedelta(days=self.retrain_frequency_days)
    
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
        split_info = []
        
        for i, (train_df, test_df) in enumerate(self.split(df, date_col)):
            info = {
                'split_id': i,
                'train_start': train_df[date_col].min(),
                'train_end': train_df[date_col].max(),
                'test_start': test_df[date_col].min(),
                'test_end': test_df[date_col].max(),
                'train_size': len(train_df),
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
        expanding_window=config.walkforward.expanding_window
    )
