"""
Walk-Forward Training Script

Trains ranking model using walk-forward validation on replay window data.

Usage:
    python scripts/02_train.py --asset us30 --config config/production_config.yaml
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data_loader import load_data
from src.feature_registry import FeatureRegistry
from src.splitter import create_splitter
from src.ranker import train_ranker
from src.selector import add_rankings
from src.evaluator import evaluate_predictions, print_metrics
from src.model_artifacts import ModelArtifact
from src.asset_utils import get_asset_paths


def main():
    parser = argparse.ArgumentParser(description='Walk-forward training')
    parser.add_argument('--asset', required=True, help='Asset identifier (e.g., us30)')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--run-id', help='Optional run ID for saving model')
    parser.add_argument('--verbose', action='store_true', help='Print detailed per-split information')
    args = parser.parse_args()
    
    print("="*70)
    print("WALK-FORWARD TRAINING")
    print("="*70)
    print(f"Asset: {args.asset}")
    
    asset_paths = get_asset_paths(args.asset, args.data_dir)
    config = load_config(args.config)
    registry_path = config.paths.feature_registry.replace('.json', f'_{args.asset}.json')
    registry = FeatureRegistry(registry_path)
    feature_cols = registry.get_feature_order()
    df = load_data(str(asset_paths.replay_window), config.data.date_col)
    
    # Validate features exist and filter to selected features + essential columns
    essential_cols = [config.data.date_col, config.data.group_col, config.data.label_col]
    if hasattr(config.data, 'pnl_col') and config.data.pnl_col:
        essential_cols.append(config.data.pnl_col)
    
    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Keep only essential columns + selected features
    df = df[essential_cols + feature_cols]
    splitter = create_splitter(config)
    
    all_predictions = []
    split_metrics = []
    
    # Count total splits for progress
    total_splits = sum(1 for _ in splitter.split(df, config.data.date_col))
    splitter = create_splitter(config)  # Recreate after counting
    
    if not args.verbose:
        print(f"\nTraining {total_splits} walk-forward splits...", end='', flush=True)
    
    for i, (train_df, test_df) in enumerate(splitter.split(df, config.data.date_col)):
        if args.verbose:
            print(f"\nSplit {i+1}/{total_splits}:")
            print(f"  Train: {train_df[config.data.date_col].min()} to {train_df[config.data.date_col].max()} ({len(train_df)} rows)")
            print(f"  Test:  {test_df[config.data.date_col].min()} to {test_df[config.data.date_col].max()} ({len(test_df)} rows)")
        else:
            # Show progress dots
            if (i + 1) % 10 == 0:
                print(f" {i+1}", end='', flush=True)
            elif (i + 1) % 5 == 0:
                print('.', end='', flush=True)
        
        # Train model
        model = train_ranker(
            train_df,
            feature_cols,
            config.data.label_col,
            config.data.group_col,
            config.get_model_params()
        )
        
        # Predict on test
        test_df = test_df.copy()
        test_df['predicted_score'] = model.predict(test_df[feature_cols])
        
        # Add rankings (actual rank based on PnL, not relevance_grade)
        test_df = add_rankings(
            test_df,
            config.data.group_col,
            'predicted_score',
            config.data.pnl_col  # Use actual PnL for ranking
        )
        
        # Evaluate
        metrics = evaluate_predictions(
            test_df,
            k=config.selection.top_k,
            group_col=config.data.group_col,
            label_col=config.data.label_col,
            pnl_col=config.data.pnl_col
        )
        
        if args.verbose:
            print(f"  Mean PnL (selected): ${metrics['mean_selected_pnl']:.2f}")
            print(f"  PnL Efficiency: {metrics['pnl_efficiency']:.2%}")
            print(f"  NDCG@{config.selection.top_k}: {metrics[f'ndcg_{config.selection.top_k}']:.4f}")
        
        # Store results
        all_predictions.append(test_df)
        split_metrics.append({
            'split': i,
            'train_start': str(train_df[config.data.date_col].min()),
            'train_end': str(train_df[config.data.date_col].max()),
            'test_start': str(test_df[config.data.date_col].min()),
            'test_end': str(test_df[config.data.date_col].max()),
            **metrics
        })
    
    if not args.verbose:
        print(f" {total_splits}/{total_splits}")  # Complete the progress line
    
    print("\n" + "="*70)
    print("WALK-FORWARD COMPLETE")
    print("="*70)
    
    # Combine all predictions
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Overall metrics
    print("\nOverall Performance:")
    overall_metrics = evaluate_predictions(
        all_predictions_df,
        k=config.selection.top_k,
        group_col=config.data.group_col,
        label_col=config.data.label_col,
        pnl_col=config.data.pnl_col
    )
    print_metrics(overall_metrics)
    
    # Train final model on all data
    print("\nTraining final model on full dataset...")
    final_model = train_ranker(
        df,
        feature_cols,
        config.data.label_col,
        config.data.group_col,
        config.get_model_params()
    )
    
    # Save model
    artifact_mgr = ModelArtifact(config.paths.models_dir)
    
    # Include asset in run_id
    if args.run_id:
        full_run_id = f"{args.asset}_{args.run_id}"
    else:
        from datetime import datetime
        full_run_id = f"{args.asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run_id = artifact_mgr.save(
        final_model,
        config.to_dict(),
        feature_cols,
        metrics=overall_metrics,
        run_id=full_run_id,
        additional_metadata={
            'asset': args.asset,
            'n_splits': len(split_metrics),
            'total_train_samples': len(df)
        }
    )
    
    # Save detailed results
    results_dir = Path(config.paths.results_dir) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    all_predictions_df.to_parquet(results_dir / 'predictions.parquet', index=False)
    print(f"\nPredictions saved: {results_dir / 'predictions.parquet'}")
    
    # Save split metrics
    with open(results_dir / 'split_metrics.json', 'w') as f:
        json.dump(split_metrics, f, indent=2)
    print(f"Split metrics saved: {results_dir / 'split_metrics.json'}")
    
    # Save feature importance
    importance_df = final_model.get_feature_importance()
    importance_df.to_csv(results_dir / 'feature_importance.csv', index=False)
    print(f"Feature importance saved: {results_dir / 'feature_importance.csv'}")
    
    print("\nTraining complete!")
    print(f"Run ID: {run_id}")


if __name__ == '__main__':
    main()
