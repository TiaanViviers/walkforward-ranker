"""
Visualize Walk-Forward Split Strategy

Shows how the training window expands and when retraining occurs.

Usage:
    python scripts/visualize_splits.py --asset us30 --config config/production_config.yaml
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data_loader import load_data
from src.splitter import create_splitter
from src.asset_utils import get_asset_paths


def visualize_splits(asset: str, config_path: str, data_dir: str = 'data', max_splits: int = 20):
    """Visualize first N splits to understand training window behavior."""
    
    asset_paths = get_asset_paths(asset, data_dir)
    config = load_config(config_path)
    
    # Load replay window data
    df = load_data(str(asset_paths.replay_window), config.data.date_col)
    
    splitter = create_splitter(config)
    
    print("="*80)
    print("WALK-FORWARD SPLIT VISUALIZATION")
    print("="*80)
    print(f"Asset: {asset}")
    print(f"Data Range: {df[config.data.date_col].min()} to {df[config.data.date_col].max()}")
    print(f"Total Days: {(df[config.data.date_col].max() - df[config.data.date_col].min()).days}")
    print(f"Total Rows: {len(df):,}")
    print()
    print(f"Strategy: {'Capped Expanding Window' if config.walkforward.expanding_window else 'Rolling Window'}")
    if config.walkforward.expanding_window:
        print(f"  - Expands from start but capped at {config.walkforward.max_training_window_days} days")
    else:
        print(f"  - Fixed window size: {config.walkforward.train_window_days} days")
    print(f"Test Window: {config.walkforward.test_window_days} days (1 trading week)")
    print(f"Retrain Frequency: {config.walkforward.retrain_frequency_days} days")
    print()
    
    print("="*80)
    print(f"FIRST {max_splits} SPLITS (showing training window behavior)")
    print("="*80)
    print()
    
    for i, (train_df, test_df) in enumerate(splitter.split(df, config.data.date_col)):
        if i >= max_splits:
            break
            
        # Count actual trading days (unique dates in data)
        train_trading_days = train_df[config.data.date_col].nunique()
        test_trading_days = test_df[config.data.date_col].nunique()
        
        print(f"Split {i+1}:")
        print(f"  Train: {train_df[config.data.date_col].min()} to {train_df[config.data.date_col].max()}")
        print(f"         {train_trading_days:4d} trading days | {len(train_df):6,} rows")
        print(f"  Test:  {test_df[config.data.date_col].min()} to {test_df[config.data.date_col].max()}")
        print(f"         {test_trading_days:4d} trading days | {len(test_df):6,} rows")
        
        # Show if window is capped
        if config.walkforward.expanding_window and train_trading_days >= config.walkforward.max_training_window_days:
            print(f"  Training window CAPPED at {config.walkforward.max_training_window_days} trading days")
        elif config.walkforward.expanding_window:
            print(f"  Training window expanding (not yet capped)")
        
        print()
    
    # Count total splits
    total_splits = sum(1 for _ in splitter.split(df, config.data.date_col))
    
    print("="*80)
    print(f"SUMMARY")
    print("="*80)
    print(f"Total Splits: {total_splits}")
    print(f"Model will be retrained {total_splits} times (once per split)")
    print()
    
    if config.walkforward.expanding_window:
        print("CAPPED EXPANDING WINDOW STRATEGY:")
        print(f"   • Early splits: Training window expands from start")
        print(f"   • Later splits: Training window capped at {config.walkforward.max_training_window_days} trading days")
        print(f"   • Benefit: Uses recent data without overfitting on old regimes")
        print(f"   • Every {config.walkforward.retrain_frequency_days} trading days: Full model retrain with updated data")
    else:
        print("ROLLING WINDOW STRATEGY:")
        print(f"   • All splits: Fixed {config.walkforward.train_window_days}-trading-day training window")
        print(f"   • Old data drops off as new data arrives")
        print(f"   • Every {config.walkforward.retrain_frequency_days} trading days: Full model retrain")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize walk-forward splits')
    parser.add_argument('--asset', required=True, help='Asset identifier (e.g., us30)')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--max-splits', type=int, default=20, help='Number of splits to show')
    args = parser.parse_args()
    
    visualize_splits(args.asset, args.config, args.data_dir, args.max_splits)
