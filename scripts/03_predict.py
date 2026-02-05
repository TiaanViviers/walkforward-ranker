"""
Production Inference Script

Loads trained model and predicts top-K configs for new data.

Usage:
    python scripts/03_predict.py --model-id 20260205_103000 --input data/today.parquet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_artifacts import ModelArtifact
from src.selector import select_top_k


def main():
    parser = argparse.ArgumentParser(description='Production inference')
    parser.add_argument('--model-id', required=True, help='Model run ID to use')
    parser.add_argument('--input', required=True, help='Path to input data')
    parser.add_argument('--output', help='Path to save predictions')
    parser.add_argument('--top-k', type=int, help='Number of configs to select (overrides model config)')
    args = parser.parse_args()
    
    print("="*70)
    print("PRODUCTION INFERENCE")
    print("="*70)
    
    # Load model artifact
    print(f"\nLoading model: {args.model_id}")
    artifact_mgr = ModelArtifact()
    artifact = artifact_mgr.load(args.model_id)
    
    model = artifact['model']
    feature_list = artifact['feature_list']
    config = artifact['config']
    
    print(f"Model timestamp: {artifact['metadata']['timestamp']}")
    print(f"Features: {len(feature_list)}")
    
    # Load input data
    print(f"\nLoading input data: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Data shape: {df.shape}")
    
    # Validate features
    missing_features = set(feature_list) - set(df.columns)
    if missing_features:
        raise ValueError(f"Input data missing required features: {missing_features}")
    
    # Ensure correct feature order
    X = df[feature_list]
    
    # Predict
    print("\nGenerating predictions...")
    scores = model.predict(X)
    df['predicted_score'] = scores
    
    # Rank configs
    df['predicted_rank'] = df['predicted_score'].rank(ascending=False, method='first')
    
    # Select top-K
    k = args.top_k if args.top_k else config['selection']['top_k']
    selected = select_top_k(df, 'predicted_score', k)
    
    print(f"\nTop-{k} Selected Configs:")
    print("-" * 70)
    
    # Display selected configs (without showing actual feature names)
    display_cols = ['predicted_rank', 'predicted_score']
    
    # Add config_id if available
    if 'config_id' in selected.columns:
        display_cols.insert(0, 'config_id')
    
    print(selected[display_cols].to_string(index=False))
    print("-" * 70)
    
    # Save predictions
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        selected.to_csv(output_path, index=False)
        print(f"\nPredictions saved: {output_path}")
    
    print("\nInference complete!")
    
    return selected


if __name__ == '__main__':
    main()
