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
import numpy as np
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
from src.hyperparameter_tuner import HyperparameterTuner


def detect_quarter_boundary(split_idx: int, retrain_frequency: int, tune_every_quarters: int) -> bool:
    """
    Detect if we're at a quarter boundary (time to retune hyperparameters).
    
    Args:
        split_idx: Current split index (0-based)
        retrain_frequency: Days between retrains (typically 5)
        tune_every_quarters: Tune every N quarters
        
    Returns:
        True if this is the start of a tuning quarter
    """
    # ~63 trading days per quarter, retraining every 5 days = ~13 splits per quarter
    splits_per_quarter = 63 // retrain_frequency
    splits_per_tuning_period = splits_per_quarter * tune_every_quarters
    
    # Tune at start of every tuning period (split 0, 13, 26, 39, ...)
    return split_idx % splits_per_tuning_period == 0


def main():
    parser = argparse.ArgumentParser(description='Walk-forward training')
    parser.add_argument('--asset', required=True, help='Asset identifier (e.g., us30)')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--run-id', help='Optional run ID for saving model')
    parser.add_argument('--verbose', action='store_true', help='Print detailed per-split information')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("WALK-FORWARD TRAINING")
    print("="*70)
    
    asset_paths = get_asset_paths(args.asset, args.data_dir)
    config = load_config(args.config)
    registry_path = config.paths.feature_registry.replace('.json', f'_{args.asset}.json')
    registry = FeatureRegistry(registry_path)
    feature_cols = registry.get_feature_order()
    print(f"Loaded {len(feature_cols)} features from registry")
    
    # Load calibration (training burn-in) + replay_window (evaluation)
    calibration_df = load_data(str(asset_paths.calibration), config.data.date_col)
    replay_df = load_data(str(asset_paths.replay_window), config.data.date_col)
    
    # Validate features exist
    essential_cols = [config.data.date_col, config.data.group_col, config.data.label_col]
    if hasattr(config.data, 'pnl_col') and config.data.pnl_col:
        essential_cols.append(config.data.pnl_col)
    
    missing_features = set(feature_cols) - set(calibration_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Keep only essential columns + selected features
    calibration_df = calibration_df[essential_cols + feature_cols]
    replay_df = replay_df[essential_cols + feature_cols]
    
    # Concatenate: calibration provides initial training data, replay_window for evaluation
    # First model trains on calibration (last 252 days), tests on first days of replay_window
    df = pd.concat([calibration_df, replay_df], ignore_index=True).sort_values(config.data.date_col).reset_index(drop=True)
    print(f"Combined dataset: {len(calibration_df):,} calibration + {len(replay_df):,} replay = {len(df):,} total rows")
    
    splitter = create_splitter(config)
    
    all_predictions = []
    split_metrics = []
    final_model = None  # Will store the last trained model (on most recent window)
    
    # Count total splits for progress
    total_splits = sum(1 for _ in splitter.split(df, config.data.date_col))
    splitter = create_splitter(config)  # Recreate after counting
    
    # Initialize hyperparameters (either from config or will be tuned)
    current_params = config.get_model_params()
    tuning_sessions = []
    
    if not args.verbose:
        print(f"\nTraining {total_splits} walk-forward splits...", end='', flush=True)
    
    for i, (train_df, test_df) in enumerate(splitter.split(df, config.data.date_col)):
        # Check if we should run hyperparameter tuning
        if (config.hyperparameter_tuning.enabled and 
            detect_quarter_boundary(i, config.walkforward.retrain_frequency_days, 
                                   config.hyperparameter_tuning.tune_every_quarters)):
            
            # Get data for tuning (use current training data available up to this point)
            tuning_data = df[df[config.data.date_col] <= train_df[config.data.date_col].max()].copy()
            
            # Keep only last 252 days for tuning (or all data if less)
            unique_dates = tuning_data[config.data.date_col].unique()
            unique_dates = np.sort(unique_dates)
            if len(unique_dates) > 252:
                tuning_start_date = unique_dates[-252]
                tuning_data = tuning_data[tuning_data[config.data.date_col] >= tuning_start_date]
            
            # Start new line for tuning (interrupts progress)
            if not args.verbose:
                print(f"\n[Split {i+1}] ", end='', flush=True)
            
            # Run Optuna tuning
            tuner = HyperparameterTuner(
                feature_cols=feature_cols,
                label_col=config.data.label_col,
                group_col=config.data.group_col,
                pnl_col=config.data.pnl_col,
                n_trials=config.hyperparameter_tuning.n_trials,
                timeout_minutes=config.hyperparameter_tuning.timeout_minutes,
                study_name=f"{args.asset}_Q{len(tuning_sessions)+1}",
                verbose=config.hyperparameter_tuning.verbose
            )
            
            try:
                best_params, study = tuner.tune(
                    tuning_data,
                    date_col=config.data.date_col,
                    validation_days=config.hyperparameter_tuning.validation_days
                )
                
                # Update current params
                current_params = best_params
                
                # Save tuning results
                tuning_session = {
                    'quarter': len(tuning_sessions) + 1,
                    'split_idx': i,
                    'date': str(train_df[config.data.date_col].max()),
                    'best_params': best_params,
                    'best_score': study.best_value,
                    'n_trials': len(study.trials)
                }
                tuning_sessions.append(tuning_session)
                
                # Continue on same line
                if not args.verbose:
                    print()
                    print(f"Training {total_splits} walk-forward splits...", end='', flush=True)
            
            except Exception as e:
                if not args.verbose:
                    print(f"Tuning skipped: {str(e)[:50]}")
                    print(f"Training {total_splits} walk-forward splits...", end='', flush=True)
                else:
                    print(f"Warning: Hyperparameter tuning failed: {e}")
                    print(f"Continuing with previous parameters.")
        
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
        
        # Train model (keep last one as deployment model)
        model = train_ranker(
            train_df,
            feature_cols,
            config.data.label_col,
            config.data.group_col,
            current_params  # Use current (possibly tuned) params
        )
        final_model = model  # Update deployment model (last one wins)
        
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
            k = config.selection.top_k
            print(f"  Mean PnL (top-{k}): ${metrics[f'mean_selected_pnl_{k}']:.2f}")
            print(f"  Efficiency: {metrics[f'pnl_efficiency_{k}']:.2%}")
            print(f"  NDCG@{k}: {metrics[f'ndcg_{k}']:.4f}")
        
        # Store results
        all_predictions.append(test_df)
        
        # Convert metrics to JSON-serializable types (float/int instead of numpy types)
        json_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in metrics.items()}
        
        split_metrics.append({
            'split': i,
            'train_start': str(train_df[config.data.date_col].min()),
            'train_end': str(train_df[config.data.date_col].max()),
            'test_start': str(test_df[config.data.date_col].min()),
            'test_end': str(test_df[config.data.date_col].max()),
            **json_metrics
        })
    
    if not args.verbose:
        print(f" {total_splits}/{total_splits}")  # Complete the progress line
    
    print()
    
    # Combine all predictions
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Filter metrics to replay_window only (exclude any calibration overlap if it exists)
    replay_start_date = replay_df[config.data.date_col].min()
    replay_predictions = all_predictions_df[all_predictions_df[config.data.date_col] >= replay_start_date]
    
    # Overall metrics (on replay window only)
    overall_metrics = evaluate_predictions(
        replay_predictions,
        k=config.selection.top_k,
        group_col=config.data.group_col,
        label_col=config.data.label_col,
        pnl_col=config.data.pnl_col
    )
    print_metrics(overall_metrics, title="Replay Window Performance")
    
    # Use last walk-forward model for deployment (maintains windowed approach)
    print("\nDeployment model: Last walk-forward model (most recent 252-day window)")
    
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
            'total_train_samples': len(df),
            'hyperparameter_tuning_enabled': config.hyperparameter_tuning.enabled,
            'tuning_sessions': tuning_sessions,  # Save tuning history
            'final_hyperparameters': current_params  # Save final params used
        }
    )
    
    results_dir = Path(config.paths.results_dir) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all artifacts
    all_predictions_df.to_parquet(results_dir / 'predictions.parquet', index=False)
    with open(results_dir / 'split_metrics.json', 'w') as f:
        json.dump(split_metrics, f, indent=2)
    importance_df = final_model.get_feature_importance()
    importance_df.to_csv(results_dir / 'feature_importance.csv', index=False)
    
    print(f"\nSaved: models/{run_id}")
    if tuning_sessions:
        print(f"  • {len(tuning_sessions)} hyperparameter tuning sessions")
    print(f"  • {len(feature_cols)} features")
    print(f"  • {len(split_metrics)} walk-forward splits")


if __name__ == '__main__':
    main()
