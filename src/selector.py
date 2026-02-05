"""Selection policies for choosing top-K configs."""

import pandas as pd
import numpy as np
from typing import List


def select_top_k(
    df: pd.DataFrame,
    score_col: str = 'predicted_score',
    k: int = 3
) -> pd.DataFrame:
    """
    Select top-K configs by predicted score.
    
    Args:
        df: DataFrame with predictions
        score_col: Column name for scores
        k: Number of configs to select
        
    Returns:
        DataFrame with selected configs
    """
    return df.nlargest(k, score_col)


def select_top_k_per_group(
    df: pd.DataFrame,
    group_col: str,
    score_col: str = 'predicted_score',
    k: int = 3
) -> pd.DataFrame:
    """
    Select top-K configs per group (e.g., per day).
    
    Args:
        df: DataFrame with predictions
        group_col: Column name for grouping
        score_col: Column name for scores
        k: Number of configs to select per group
        
    Returns:
        DataFrame with selected configs
    """
    selected = df.groupby(group_col, group_keys=False).apply(
        lambda x: x.nlargest(k, score_col)
    )
    return selected.reset_index(drop=True)


def add_rankings(
    df: pd.DataFrame,
    group_col: str,
    score_col: str = 'predicted_score',
    label_col: str = 'pnl'
) -> pd.DataFrame:
    """
    Add predicted and actual rankings to dataframe.
    
    Args:
        df: DataFrame with scores and labels
        group_col: Column name for grouping
        score_col: Column name for predicted scores
        label_col: Column name for actual labels
        
    Returns:
        DataFrame with ranking columns added
    """
    df = df.copy()
    
    # Predicted rank (from scores)
    df['predicted_rank'] = df.groupby(group_col)[score_col].rank(
        ascending=False,
        method='first'
    )
    
    # Actual rank (from realized PnL)
    df['actual_rank'] = df.groupby(group_col)[label_col].rank(
        ascending=False,
        method='first'
    )
    
    return df
