"""
Master Runner Script

Runs complete pipeline for an asset: feature selection, training, and evaluation.

Usage:
    # Run full pipeline
    python scripts/run_pipeline.py --asset us30 --config config/production_config.yaml
    
    # Run specific stages
    python scripts/run_pipeline.py --asset us30 --config config/production_config.yaml --stages feature_selection
    python scripts/run_pipeline.py --asset us30 --config config/production_config.yaml --stages training
    python scripts/run_pipeline.py --asset us30 --config config/production_config.yaml --stages feature_selection,training
"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asset_utils import get_asset_paths


def run_feature_selection(asset: str, config: str, data_dir: str):
    """Run feature selection stage."""
    
    cmd = [
        sys.executable,
        'scripts/01_feature_selection.py',
        '--asset', asset,
        '--config', config,
        '--data-dir', data_dir
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print("\nError in feature selection stage")
        sys.exit(1)
    
    return True


def run_training(asset: str, config: str, data_dir: str, run_id: str = None, verbose: bool = False):
    """Run walk-forward training stage."""
    
    cmd = [
        sys.executable,
        'scripts/02_train.py',
        '--asset', asset,
        '--config', config,
        '--data-dir', data_dir
    ]
    
    if run_id:
        cmd.extend(['--run-id', run_id])
    
    if verbose:
        cmd.append('--verbose')
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print("\nError in training stage")
        sys.exit(1)
    
    return True


def run_holdout_evaluation(asset: str, results_dir: str):
    """Run holdout evaluation stage."""    
    # Find latest model for this asset
    models_dir = Path('models')
    asset_models = sorted([
        d for d in models_dir.iterdir()
        if d.is_dir() and d.name.startswith(f"{asset}_")
    ], key=lambda x: x.name, reverse=True)
    
    if not asset_models:
        print(f"No trained models found for asset {asset}")
        return False
    
    latest_run = asset_models[0].name
    print(f"Analyzing holdout set for run: {latest_run}")
    
    cmd = [
        sys.executable,
        'scripts/04_evaluate.py',
        '--run-id', latest_run,
        '--results-dir', results_dir
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print("\nError in holdout evaluation stage")
        sys.exit(1)
    
    return True


def main():
    """Run complete pipeline for an asset."""
    args = get_args()
    
    # Validate asset data exists
    try:
        asset_paths = get_asset_paths(args.asset, args.data_dir)
        print(f"Asset: {args.asset}")
        print()
        
        # Show concise data report with date ranges
        import pandas as pd
        from src.data_loader import load_data
        from src.config import load_config
        
        config = load_config(args.config)
        date_col = config.data.date_col
        
        def print_dataset_info(name: str, path: Path):
            df = pd.read_parquet(path)
            df[date_col] = pd.to_datetime(df[date_col])
            start_date = df[date_col].min().strftime('%Y-%m-%d')
            end_date = df[date_col].max().strftime('%Y-%m-%d')
            n_days = df[date_col].nunique()
            print(f"{name:15} {start_date} to {end_date}  ({n_days:3d} days)")
        
        print_dataset_info("Calibration:", asset_paths.calibration)
        print_dataset_info("Replay:", asset_paths.replay_window)
        print_dataset_info("Holdout:", asset_paths.holdout)
        
    except Exception as e:
        print(f"Error validating asset data: {e}")
        sys.exit(1)
    
    # Parse stages
    stages = [s.strip() for s in args.stages.split(',')]
    valid_stages = {'feature_selection', 'training', 'holdout'}
    invalid = set(stages) - valid_stages
    if invalid:
        print(f"Invalid stages: {invalid}")
        print(f"Valid stages: {valid_stages}")
        sys.exit(1)
    
    # Start pipeline
    start_time = datetime.now()
    print("\n" + "="*70)
    print(f"PIPELINE START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Run stages
    if 'feature_selection' in stages:
        run_feature_selection(args.asset, args.config, args.data_dir)
    
    if 'training' in stages:
        run_training(args.asset, args.config, args.data_dir, args.run_id, args.verbose)
    
    if 'holdout' in stages:
        #run_holdout_evaluation(args.asset, args.results_dir)
        pass
    
    # End pipeline
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\nPipeline complete! Duration: {duration}")
    

def get_args():
    parser = argparse.ArgumentParser(
        description='Run complete pipeline for an asset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--asset', required=True, help='Asset identifier (e.g., us30, us500)')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument(
        '--stages',
        default='feature_selection,training,holdout',
        help='Comma-separated list of stages to run (feature_selection, training, holdout)'
    )
    parser.add_argument('--run-id', help='Optional run ID for training')
    parser.add_argument('--verbose', action='store_true', help='Print detailed per-split information during training')
    
    return parser.parse_args()


if __name__ == '__main__':
    main()
