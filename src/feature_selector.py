"""Feature selection utilities with parallel flagging."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
from src.ranker import train_ranker


def flag_low_variance_features(
    df: pd.DataFrame,
    features: List[str],
    threshold: float = 1e-6
) -> Set[str]:
    """
    Flag features with very low variance.
    
    Args:
        df: DataFrame containing features
        features: List of feature names
        threshold: Variance threshold
        
    Returns:
        Set of features flagged for low variance
    """
    variances = df[features].var()
    low_var = variances[variances <= threshold].index.tolist()    
    return set(low_var)


def flag_correlated_features(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    threshold: float = 0.95
) -> Set[str]:
    """
    Flag highly correlated features.
    
    When two features are highly correlated, flags the one less
    correlated with the target variable.
    
    Args:
        df: DataFrame containing features
        features: List of feature names
        label_col: Target variable column name
        threshold: Correlation threshold
        
    Returns:
        Set of features flagged for high correlation
    """
    # Compute correlation matrices
    feature_corr = df[features].corr().abs()
    target_corr = df[features + [label_col]].corr()[label_col].abs()
    
    flagged = set()
    
    for i, feat1 in enumerate(features):
        if feat1 in flagged:
            continue
        
        for feat2 in features[i+1:]:
            if feat2 in flagged:
                continue
            
            # Check if highly correlated
            if feature_corr.loc[feat1, feat2] > threshold:
                # Flag the one less correlated with target
                if target_corr[feat1] >= target_corr[feat2]:
                    flagged.add(feat2)
                else:
                    flagged.add(feat1)
    
    return flagged


def flag_low_importance_features(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    group_col: str,
    params: dict,
    min_percentile: int = 5
) -> Tuple[Set[str], pd.DataFrame]:
    """
    Flag features with low importance.
    
    Args:
        train_df: Training data
        feature_cols: List of feature names
        label_col: Label column name
        group_col: Group column name
        params: Model parameters
        min_percentile: Flag features below this importance percentile
        
    Returns:
        Tuple of (flagged_features, importance_df)
    """
    # Train model on all features
    model = train_ranker(
        train_df,
        feature_cols,
        label_col,
        group_col,
        params
    )
    
    # Get importance
    importance_df = model.get_feature_importance()
    
    # Compute threshold
    threshold = np.percentile(
        importance_df['importance'],
        min_percentile
    )
    
    # Flag features below threshold
    flagged = set(
        importance_df[
            importance_df['importance'] <= threshold
        ]['feature'].tolist()
    )
    
    return flagged, importance_df


def select_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    group_col: str,
    params: dict,
    correlation_threshold: float = 0.95,
    min_importance_percentile: int = 5,
    variance_threshold: float = 1e-6,
    removal_strategy: str = "conservative"
) -> Tuple[List[str], dict]:
    """
    Feature selection with parallel flagging.
    
    Strategies:
    - aggressive: Remove if flagged by ANY criterion
    - moderate: Remove if flagged by 2+ criteria
    - conservative: Remove if flagged by ALL criteria
    
    Args:
        df: Full dataset
        feature_cols: Initial list of feature names
        label_col: Label column name
        group_col: Group column name
        params: Model parameters
        correlation_threshold: Correlation threshold
        min_importance_percentile: Importance percentile threshold
        variance_threshold: Variance threshold
        removal_strategy: Strategy for removal (aggressive/moderate/conservative)
        
    Returns:
        Tuple of (final_features, selection_info_dict)
    """
    variance_threshold = float(variance_threshold)
    correlation_threshold = float(correlation_threshold)
    min_importance_percentile = int(min_importance_percentile)
        
    # Flag features based on each criterion (parallel)
    variance_flags = flag_low_variance_features(df, feature_cols, variance_threshold)
    correlation_flags = flag_correlated_features(df, feature_cols, label_col, correlation_threshold)
    
    # Use subset of data for importance (speed)
    train_subset = df.head(min(len(df), 500000))
    importance_flags, importance_df = flag_low_importance_features(
        train_subset,
        feature_cols,
        label_col,
        group_col,
        params,
        min_importance_percentile
    )
    
    # Create flag count per feature
    flag_counts = {}
    for feat in feature_cols:
        count = 0
        if feat in variance_flags:
            count += 1
        if feat in correlation_flags:
            count += 1
        if feat in importance_flags:
            count += 1
        flag_counts[feat] = count
    
    # Apply removal strategy
    if removal_strategy == "aggressive":
        # Remove if flagged by ANY criterion (1+ flags)
        to_remove = {f for f, count in flag_counts.items() if count >= 1}
    elif removal_strategy == "moderate":
        # Remove if flagged by 2+ criteria
        to_remove = {f for f, count in flag_counts.items() if count >= 2}
    else:
        # Remove if flagged by ALL criteria
        to_remove = {f for f, count in flag_counts.items() if count == 3}
    
    final_features = [f for f in feature_cols if f not in to_remove]
    
    selection_info = {
        'initial_count': len(feature_cols),
        'final_count': len(final_features),
        'removal_strategy': removal_strategy,
        'flagged_variance': list(variance_flags),
        'flagged_correlation': list(correlation_flags),
        'flagged_importance': list(importance_flags),
        'flag_counts': flag_counts,
        'removed_features': list(to_remove),
        'final_features': final_features,
        'importance': importance_df.to_dict('records')
    }
    
    return final_features, selection_info
