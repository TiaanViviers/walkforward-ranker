"""
Results Analysis Script

Analyzes walk-forward results and generates diagnostic plots.

Usage:
    python scripts/04_evaluate.py --run-id 20260205_103000
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import evaluate_predictions, print_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate results')
    parser.add_argument('--run-id', required=True, help='Run ID to analyze')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.results_dir) / args.run_id
    
    if not results_path.exists():
        print(f"Error: Results not found at {results_path}")
        return
    
    # Load predictions
    predictions_df = pd.read_parquet(results_path / 'predictions.parquet')
    
    with open(results_path / 'split_metrics.json', 'r') as f:
        split_metrics = json.load(f)
            
    # Load feature importance
    importance_df = pd.read_csv(results_path / 'feature_importance.csv')
    
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE")
    print("="*70)
    
    # Determine k from split metrics
    k = 3
    for key in split_metrics[0].keys():
        if key.startswith('ndcg_'):
            k = int(key.split('_')[1])
            break
    
    overall_metrics = evaluate_predictions(
        predictions_df,
        k=k,
        group_col='group_id',
        label_col='relevance_grade',
        pnl_col='pnl'
    )
    print_metrics(overall_metrics)
    
    # Per-split performance
    print("\n" + "="*70)
    print("PER-SPLIT PERFORMANCE")
    print("="*70)
    
    splits_df = pd.DataFrame(split_metrics)
    
    print(f"\nNDCG@{k} over time:")
    print("-" * 70)
    for _, row in splits_df.iterrows():
        print(f"  Split {row['split']:2d} | {row['test_start']} to {row['test_end']} | NDCG: {row[f'ndcg_{k}']:.4f}")
    print("-" * 70)
    
    print(f"\nMean NDCG@{k}: {splits_df[f'ndcg_{k}'].mean():.4f}")
    print(f"Std NDCG@{k}:  {splits_df[f'ndcg_{k}'].std():.4f}")
    print(f"Min NDCG@{k}:  {splits_df[f'ndcg_{k}'].min():.4f}")
    print(f"Max NDCG@{k}:  {splits_df[f'ndcg_{k}'].max():.4f}")
    
    # Feature importance
    print("\n" + "="*70)
    print("TOP 20 FEATURES BY IMPORTANCE")
    print("="*70)
    
    print(importance_df.head(20).to_string(index=False))
    
    # Time-based analysis
    print("\n" + "="*70)
    print("PERFORMANCE OVER TIME")
    print("="*70)
    
    if 'date' in predictions_df.columns:
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        predictions_df['year'] = predictions_df['date'].dt.year
        
        yearly_metrics = predictions_df.groupby('year').apply(
            lambda x: evaluate_predictions(x, k=k, group_col='group_id', label_col='relevance_grade', pnl_col='pnl')
        )
        
        print("\nPerformance by year:")
        for year in sorted(yearly_metrics.index):
            metrics = yearly_metrics[year]
            print(f"  {year}: NDCG@{k}={metrics[f'ndcg_{k}']:.4f}, Hit Rate@{k}={metrics[f'hit_rate_{k}']:.4f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
