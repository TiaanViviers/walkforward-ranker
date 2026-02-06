"""
Feature Selection Script

Performs feature selection on calibration data and saves feature registry.

Usage:
    python scripts/01_feature_selection.py --asset us30 --config config/production_config.yaml
"""

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data_loader import load_data, validate_data
from src.feature_selector import select_features
from src.feature_registry import FeatureRegistry
from src.asset_utils import get_asset_paths


def main():
    parser = argparse.ArgumentParser(description='Feature selection on calibration data')
    parser.add_argument('--asset', required=True, help='Asset identifier (e.g., us30)')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    args = parser.parse_args()
    
    print("="*70)
    print("FEATURE SELECTION")
    print("="*70)
    print(f"Asset: {args.asset}")
    
    asset_paths = get_asset_paths(args.asset, args.data_dir)
    config = load_config(args.config)
    df = load_data(str(asset_paths.calibration), config.data.date_col)
    all_features = config.get_all_features()
    print(f"Total features: {len(all_features)}")
    
    # Validate data
    validate_data(
        df,
        config.data.date_col,
        config.data.group_col,
        config.data.label_col,
        all_features
    )
    
    # Run feature selection
    print("\nRunning feature selection pipeline...")
    selected_features, selection_info = select_features(
        df,
        all_features,
        config.data.label_col,
        config.data.group_col,
        config.get_model_params(),
        correlation_threshold=config.feature_selection.correlation_threshold,
        min_importance_percentile=config.feature_selection.min_importance_percentile,
        variance_threshold=config.feature_selection.variance_threshold,
        removal_strategy=config.feature_selection.removal_strategy
    )
    
    # Create feature registry
    registry_path = config.paths.feature_registry.replace('.json', f'_{args.asset}.json')
    registry = FeatureRegistry(registry_path)
    registry.create(selected_features, df, version="1.0.0")
    
    print(f"\nFeature registry saved: {registry_path}")
    
    # Save selection info
    selection_info_path = Path(config.paths.feature_registry).parent / f'feature_selection_info_{args.asset}.json'
    with open(selection_info_path, 'w') as f:
        # Convert sets to lists for JSON serialization
        info_serializable = {
            k: (list(v) if isinstance(v, set) else v)
            for k, v in selection_info.items()
        }
        json.dump(info_serializable, f, indent=2)
    
    print(f"Selection info saved: {selection_info_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("FEATURE SELECTION SUMMARY")
    print("="*70)
    print(f"Strategy:             {selection_info['removal_strategy']}")
    print(f"Initial features:     {selection_info['initial_count']}")
    print(f"Flagged variance:     {len(selection_info['flagged_variance'])}")
    print(f"Flagged correlation:  {len(selection_info['flagged_correlation'])}")
    print(f"Flagged importance:   {len(selection_info['flagged_importance'])}")
    print(f"Removed features:     {len(selection_info['removed_features'])}")
    print(f"Final features:       {selection_info['final_count']}")
    print(f"Reduction:            {100 * (1 - selection_info['final_count'] / selection_info['initial_count']):.1f}%")
    print("="*70)
    
    print("\nTop 10 features by importance:")
    for i, feat_info in enumerate(selection_info['importance'][:10], 1):
        print(f"  {i:2d}. {feat_info['feature']:30s} {feat_info['importance']:10.2f}")
    
    # Print selected and removed features
    print("\n" + "="*70)
    print("SELECTED FEATURES ({} features)".format(len(selected_features)))
    print("="*70)
    for i, feat in enumerate(sorted(selected_features), 1):
        print(f"  {i:2d}. {feat}")
    
    removed_features = set(all_features) - set(selected_features)
    print("\n" + "="*70)
    print("REMOVED FEATURES ({} features)".format(len(removed_features)))
    print("="*70)
    for i, feat in enumerate(sorted(removed_features), 1):
        print(f"  {i:2d}. {feat}")


if __name__ == '__main__':
    main()
