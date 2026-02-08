"""
Test Quarterly Hyperparameter Tuning

Quick test to verify Optuna integration works correctly.

Usage:
    python scripts/test_optuna.py --asset us30 --config config/production_config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data_loader import load_data
from src.feature_registry import FeatureRegistry
from src.asset_utils import get_asset_paths
from src.hyperparameter_tuner import HyperparameterTuner


def main():
    parser = argparse.ArgumentParser(description='Test Optuna hyperparameter tuning')
    parser.add_argument('--asset', required=True, help='Asset identifier (e.g., us30)')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--n-trials', type=int, default=10, help='Number of trials (default: 10 for testing)')
    args = parser.parse_args()
    
    print("="*70)
    print("TESTING OPTUNA HYPERPARAMETER TUNING")
    print("="*70)
    print(f"Asset: {args.asset}")
    print(f"Trials: {args.n_trials}")
    print()
    
    # Load config and data
    asset_paths = get_asset_paths(args.asset, args.data_dir)
    config = load_config(args.config)
    registry_path = config.paths.feature_registry.replace('.json', f'_{args.asset}.json')
    registry = FeatureRegistry(registry_path)
    feature_cols = registry.get_feature_order()
    
    # Load calibration data for testing
    print("Loading calibration data...")
    df = load_data(str(asset_paths.calibration), config.data.date_col)
    print(f"Loaded {len(df):,} rows")
    print()
    
    # Keep essential columns + features
    essential_cols = [config.data.date_col, config.data.group_col, config.data.label_col]
    if hasattr(config.data, 'pnl_col') and config.data.pnl_col:
        essential_cols.append(config.data.pnl_col)
    
    df = df[essential_cols + feature_cols]
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        feature_cols=feature_cols,
        label_col=config.data.label_col,
        group_col=config.data.group_col,
        pnl_col=config.data.pnl_col,
        n_trials=args.n_trials,
        timeout_minutes=5,  # Short timeout for testing
        study_name="test_study",
        verbose=True
    )
    
    # Run tuning
    try:
        best_params, study = tuner.tune(
            df,
            date_col=config.data.date_col,
            validation_days=10  # Small validation set for testing
        )
        
        print("\n" + "="*70)
        print("TEST SUCCESSFUL!")
        print("="*70)
        print(f"Best parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest NDCG@3: {study.best_value:.4f}")
        print(f"Trials completed: {len(study.trials)}")
        print()
        
        # Show convergence
        print("Convergence:")
        for i, trial in enumerate(study.trials[:5]):
            print(f"  Trial {trial.number}: NDCG={trial.value:.4f}")
        if len(study.trials) > 5:
            print(f"  ...")
            for trial in study.trials[-3:]:
                print(f"  Trial {trial.number}: NDCG={trial.value:.4f}")
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
