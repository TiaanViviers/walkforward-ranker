"""Evaluation metrics for ranking models."""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import ndcg_score


def hit_rate_at_k(
    df: pd.DataFrame,
    k: int = 3,
    group_col: str = 'group_id',
    predicted_rank_col: str = 'predicted_rank',
    actual_rank_col: str = 'actual_rank'
) -> float:
    """
    Calculate hit rate at K.
    
    Hit rate: proportion of groups where at least one top-K prediction
    was in the actual top-K.
    
    Args:
        df: DataFrame with rankings
        k: K value
        group_col: Group column name
        predicted_rank_col: Predicted rank column
        actual_rank_col: Actual rank column
        
    Returns:
        Hit rate (0 to 1)
    """
    hits = []
    
    for _, group in df.groupby(group_col):
        predicted_top_k = set(group[group[predicted_rank_col] <= k].index)
        actual_top_k = set(group[group[actual_rank_col] <= k].index)
        
        # Check if any overlap
        hit = len(predicted_top_k & actual_top_k) > 0
        hits.append(hit)
    
    return np.mean(hits)


def precision_at_k(
    df: pd.DataFrame,
    k: int = 3,
    group_col: str = 'group_id',
    predicted_rank_col: str = 'predicted_rank',
    actual_rank_col: str = 'actual_rank'
) -> float:
    """
    Calculate precision at K.
    
    Precision: proportion of predicted top-K that are in actual top-K.
    
    Args:
        df: DataFrame with rankings
        k: K value
        group_col: Group column name
        predicted_rank_col: Predicted rank column
        actual_rank_col: Actual rank column
        
    Returns:
        Precision (0 to 1)
    """
    precisions = []
    
    for _, group in df.groupby(group_col):
        predicted_top_k = set(group[group[predicted_rank_col] <= k].index)
        actual_top_k = set(group[group[actual_rank_col] <= k].index)
        
        if len(predicted_top_k) == 0:
            continue
        
        precision = len(predicted_top_k & actual_top_k) / len(predicted_top_k)
        precisions.append(precision)
    
    return np.mean(precisions)


def ndcg_at_k(
    df: pd.DataFrame,
    k: int = 3,
    group_col: str = 'group_id',
    label_col: str = 'pnl',
    score_col: str = 'predicted_score'
) -> float:
    """
    Calculate NDCG at K.
    
    Args:
        df: DataFrame with scores and labels
        k: K value
        group_col: Group column name
        label_col: Label column (true relevance)
        score_col: Score column (predicted relevance)
        
    Returns:
        NDCG score (0 to 1)
    """
    ndcg_scores = []
    
    for _, group in df.groupby(group_col):
        if len(group) < k:
            continue
        
        y_true = group[label_col].values.reshape(1, -1)
        y_score = group[score_col].values.reshape(1, -1)
        
        try:
            score = ndcg_score(y_true, y_score, k=k)
            ndcg_scores.append(score)
        except:
            continue
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def regret(
    df: pd.DataFrame,
    k: int = 3,
    group_col: str = 'group_id',
    predicted_rank_col: str = 'predicted_rank',
    label_col: str = 'pnl'
) -> float:
    """
    Calculate regret: missed PnL vs. optimal selection.
    
    Regret = (Best possible PnL) - (Realized PnL from selection)
    
    Args:
        df: DataFrame with rankings and labels
        k: K value
        group_col: Group column name
        predicted_rank_col: Predicted rank column
        label_col: Label column
        
    Returns:
        Average regret
    """
    regrets = []
    
    for _, group in df.groupby(group_col):
        # Best possible: top-K by actual PnL
        best_pnl = group.nlargest(k, label_col)[label_col].sum()
        
        # Realized: top-K by prediction
        realized_pnl = group[group[predicted_rank_col] <= k][label_col].sum()
        
        regret_value = best_pnl - realized_pnl
        regrets.append(regret_value)
    
    return np.mean(regrets)


def evaluate_predictions(
    df: pd.DataFrame,
    k: int = 3,
    group_col: str = 'group_id',
    predicted_rank_col: str = 'predicted_rank',
    actual_rank_col: str = 'actual_rank',
    label_col: str = 'relevance_grade',
    score_col: str = 'predicted_score',
    pnl_col: str = 'pnl'
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        df: DataFrame with all necessary columns
        k: K value for top-K metrics
        group_col: Group column name
        predicted_rank_col: Predicted rank column
        actual_rank_col: Actual rank column
        label_col: Label column (relevance grade)
        score_col: Score column
        pnl_col: Actual PnL column for business metrics
        
    Returns:
        Dictionary of metrics
    """
    # Calculate PnL metrics (most important for business)
    predicted_pnl_per_group = df[df[predicted_rank_col] <= k].groupby(group_col)[pnl_col].sum()
    optimal_pnl_per_group = df.groupby(group_col).apply(lambda x: x.nlargest(k, pnl_col)[pnl_col].sum())
    
    mean_predicted_pnl = predicted_pnl_per_group.mean()
    mean_optimal_pnl = optimal_pnl_per_group.mean()
    median_predicted_pnl = predicted_pnl_per_group.median()
    total_predicted_pnl = predicted_pnl_per_group.sum()
    total_optimal_pnl = optimal_pnl_per_group.sum()
    
    # Win rate of predicted configs
    predicted_win_rate = (df[df[predicted_rank_col] <= k][pnl_col] > 0).sum() / (df[predicted_rank_col] <= k).sum()
    optimal_win_rate = (df.groupby(group_col).apply(lambda x: x.nlargest(k, pnl_col)[pnl_col] > 0).sum() / 
                        df.groupby(group_col).apply(lambda x: k).sum())
    
    # PnL efficiency: ratio of realized to optimal
    pnl_efficiency = mean_predicted_pnl / mean_optimal_pnl if mean_optimal_pnl != 0 else 0.0
    
    metrics = {
        # === PNL METRICS (PRIMARY) ===
        'mean_selected_pnl': mean_predicted_pnl,
        'median_selected_pnl': median_predicted_pnl,
        'total_selected_pnl': total_predicted_pnl,
        'mean_optimal_pnl': mean_optimal_pnl,
        'total_optimal_pnl': total_optimal_pnl,
        'pnl_efficiency': pnl_efficiency,  # ratio: selected/optimal
        f'regret_{k}': regret(df, k, group_col, predicted_rank_col, pnl_col),
        'win_rate_selected': predicted_win_rate,
        'win_rate_optimal': optimal_win_rate,
        
        # === RANKING METRICS (SECONDARY) ===
        f'hit_rate_{k}': hit_rate_at_k(
            df, k, group_col, predicted_rank_col, actual_rank_col
        ),
        f'precision_{k}': precision_at_k(
            df, k, group_col, predicted_rank_col, actual_rank_col
        ),
        f'ndcg_{k}': ndcg_at_k(
            df, k, group_col, label_col, score_col
        ),
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Print metrics in formatted way with PnL metrics highlighted."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    # PnL metrics first (most important)
    print("\n📊 PNL METRICS (Primary):")
    print("-" * 60)
    pnl_keys = ['mean_selected_pnl', 'median_selected_pnl', 'total_selected_pnl',
                'mean_optimal_pnl', 'total_optimal_pnl', 'pnl_efficiency',
                'win_rate_selected', 'win_rate_optimal']
    regret_keys = [k for k in metrics.keys() if 'regret' in k]
    
    for key in pnl_keys + regret_keys:
        if key in metrics:
            value = metrics[key]
            if 'efficiency' in key:
                print(f"  {key:30s}: {value:10.2%}")
            elif 'win_rate' in key:
                print(f"  {key:30s}: {value:10.2%}")
            else:
                print(f"  {key:30s}: {value:10.2f}")
    
    # Ranking metrics second
    print("\n🎯 RANKING METRICS (Secondary):")
    print("-" * 60)
    ranking_keys = ['hit_rate', 'precision', 'ndcg']
    for metric, value in metrics.items():
        if any(rk in metric for rk in ranking_keys):
            if 'ndcg' in metric:
                print(f"  {metric:30s}: {value:10.4f}")
            else:
                print(f"  {metric:30s}: {value:10.2%}")
    
    print("=" * 60)
