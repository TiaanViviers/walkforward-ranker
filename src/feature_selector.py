"""Feature selection utilities."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from src.ranker import train_ranker


def remove_low_variance_features(
    df: pd.DataFrame,
    features: List[str],
    threshold: float = 1e-6
) -> List[str]:
    """
    Remove features with very low variance.
    
    Args:
        df: DataFrame containing features
        features: List of feature names
        threshold: Variance threshold
        
    Returns:
        List of features with sufficient variance
    """
    variances = df[features].var()
    keep = variances[variances > threshold].index.tolist()
    removed = set(features) - set(keep)
    
    if removed:
        print(f"Removed {len(removed)} low-variance features")
    
    return keep


def remove_correlated_features(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    threshold: float = 0.95
) -> Tuple[List[str], List[str]]:
    """
    Remove highly correlated features.
    
    When two features are highly correlated, keeps the one more
    correlated with the target variable.
    
    Args:
        df: DataFrame containing features
        features: List of feature names
        label_col: Target variable column name
        threshold: Correlation threshold
        
    Returns:
        Tuple of (kept_features, removed_features)
    """
    # Compute correlation matrices
    feature_corr = df[features].corr().abs()
    target_corr = df[features + [label_col]].corr()[label_col].abs()
    
    to_remove = set()
    
    for i, feat1 in enumerate(features):
        if feat1 in to_remove:
            continue
        
        for feat2 in features[i+1:]:
            if feat2 in to_remove:
                continue
            
            # Check if highly correlated
            if feature_corr.loc[feat1, feat2] > threshold:
                # Keep the one more correlated with target
                if target_corr[feat1] >= target_corr[feat2]:
                    to_remove.add(feat2)
                else:
                    to_remove.add(feat1)
    
    kept = [f for f in features if f not in to_remove]
    removed = list(to_remove)
    
    print(f"Removed {len(removed)} correlated features (threshold={threshold})")
    
    return kept, removed


def select_by_importance(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    group_col: str,
    params: dict,
    min_percentile: int = 5
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on model importance.
    
    Args:
        train_df: Training data
        feature_cols: List of feature names
        label_col: Label column name
        group_col: Group column name
        params: Model parameters
        min_percentile: Keep features above this importance percentile
        
    Returns:
        Tuple of (selected_features, importance_df)
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
    
    # Select features above threshold
    selected = importance_df[
        importance_df['importance'] > threshold
    ]['feature'].tolist()
    
    print(f"Selected {len(selected)} / {len(feature_cols)} features "
          f"(>{min_percentile}th percentile)")
    
    return selected, importance_df


def select_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    group_col: str,
    params: dict,
    correlation_threshold: float = 0.95,
    min_importance_percentile: int = 5,
    variance_threshold: float = 1e-6
) -> Tuple[List[str], dict]:
    """
    Full feature selection pipeline.
    
    Steps:
    1. Remove low-variance features
    2. Remove correlated features
    3. Select by importance
    
    Args:
        df: Full dataset
        feature_cols: Initial list of feature names
        label_col: Label column name
        group_col: Group column name
        params: Model parameters
        correlation_threshold: Correlation threshold
        min_importance_percentile: Importance percentile threshold
        variance_threshold: Variance threshold
        
    Returns:
        Tuple of (final_features, selection_info_dict)
    """
    variance_threshold = float(variance_threshold)
    correlation_threshold = float(correlation_threshold)
    min_importance_percentile = int(min_importance_percentile)
    
    # Remove low variance
    features_v1 = remove_low_variance_features(
        df, feature_cols, variance_threshold
    )
    print(f"After variance filter: {len(features_v1)}")
    
    # Remove correlated
    features_v2, removed_corr = remove_correlated_features(
        df, features_v1, label_col, correlation_threshold
    )
    print(f"After correlation filter: {len(features_v2)}")
    
    # Select by importance
    # Use subset of data for speed (first year)
    train_subset = df.head(min(len(df), 500000))
    
    features_v3, importance_df = select_by_importance(
        train_subset,
        features_v2,
        label_col,
        group_col,
        params,
        min_importance_percentile
    )
    print(f"After importance filter: {len(features_v3)}")
    
    # Collect selection info
    selection_info = {
        'initial_count': len(feature_cols),
        'after_variance': len(features_v1),
        'after_correlation': len(features_v2),
        'final_count': len(features_v3),
        'removed_variance': list(set(feature_cols) - set(features_v1)),
        'removed_correlation': removed_corr,
        'final_features': features_v3,
        'importance': importance_df.to_dict('records')
    }
    
    return features_v3, selection_info
