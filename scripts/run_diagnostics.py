"""
Diagnostics Runner

Generates comprehensive performance analysis and visualizations for a trained model.

Usage:
    python scripts/run_diagnostics.py --run-id us30_20260208_210112
    python scripts/run_diagnostics.py --run-id us30_20260208_210112 --top-k 3
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from diagnostics.metrics import calculate_all_metrics, print_metrics_report
from diagnostics.equity_plots import plot_equity_curve, plot_drawdowns, plot_rolling_metrics
from diagnostics.regime_analysis import detect_volatility_regimes, analyze_by_regime, plot_regime_performance, print_regime_comparison


def main():
    parser = argparse.ArgumentParser(description='Generate diagnostics for trained model')
    parser.add_argument('--run-id', required=True, help='Run ID (e.g., us30_20260208_210112)')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--top-k', type=int, default=3, help='Top-K configs to analyze')
    parser.add_argument('--vol-feature', default='pnl_vol_5d', help='Volatility feature for regime detection')
    parser.add_argument('--capital-per-position', type=float, default=10000.0, 
                       help='Notional capital per position for returns calculation (default: $10,000)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DIAGNOSTICS ANALYSIS")
    print("="*70)
    print(f"Run ID: {args.run_id}")
    print(f"Top-K: {args.top_k}")
    
    # Setup paths
    results_dir = Path(args.results_dir) / args.run_id
    predictions_path = results_dir / 'predictions.parquet'
    split_metrics_path = results_dir / 'split_metrics.json'
    diagnostics_dir = results_dir / 'diagnostics'
    plots_dir = diagnostics_dir / 'plots'
    
    # Check if files exist
    if not predictions_path.exists():
        print(f"\nError: Predictions file not found at {predictions_path}")
        sys.exit(1)
    
    # Create diagnostics directories
    diagnostics_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    print(f"\nLoading predictions from: {predictions_path}")
    df = pd.read_parquet(predictions_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df):,} predictions")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique dates: {df['date'].nunique()}")
    
    # Calculate daily PnL for top-K selection
    print(f"\nCalculating daily PnL for top-{args.top_k} selection...")
    
    # Filter to top-K predicted configs per day
    top_k_df = (df.sort_values(['date', 'predicted_score'], ascending=[True, False])
                  .groupby('date')
                  .head(args.top_k))
    
    # Daily PnL = sum of PnL from top-K configs each day
    daily_pnl = top_k_df.groupby('date')['pnl'].sum().sort_index()
    dates = pd.Series(daily_pnl.index)
    
    print(f"Daily PnL series: {len(daily_pnl)} days")
    
    # ========================================================================
    # METRICS CALCULATION
    # ========================================================================
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    print(f"\nAssumptions:")
    print(f"  Capital per position: ${args.capital_per_position:,.0f}")
    print(f"  Top-{args.top_k} total capital: ${args.capital_per_position * args.top_k:,.0f}/day")
    print(f"  Method: Fixed notional (no compounding)")
    
    metrics = calculate_all_metrics(daily_pnl, capital_per_position=args.capital_per_position)
    
    # Print metrics report
    print_metrics_report(metrics)
    
    # Save metrics to JSON
    metrics_path = diagnostics_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")
    
    # ========================================================================
    # REGIME ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("REGIME ANALYSIS")
    print("="*70)
    
    try:
        # Detect regimes
        print(f"\nDetecting volatility regimes using '{args.vol_feature}'...")
        regimes = detect_volatility_regimes(top_k_df, vol_feature=args.vol_feature)
        
        # Add regimes to df
        top_k_df['regime'] = regimes.values
        
        # Map regimes to daily data for plotting
        daily_regimes = top_k_df.groupby('date')['regime'].first()
        
        print(f"  Low vol:  {(regimes == 'low').sum():>6,} predictions")
        print(f"  Med vol:  {(regimes == 'med').sum():>6,} predictions")
        print(f"  High vol: {(regimes == 'high').sum():>6,} predictions")
        
        # Analyze performance by regime
        regime_metrics = analyze_by_regime(top_k_df, date_col='date', pnl_col='pnl', regime_col='regime')
        
        # Print comparison
        print_regime_comparison(regime_metrics)
        
        # Save regime metrics
        regime_metrics_path = diagnostics_dir / 'regime_metrics.json'
        # Convert to JSON-serializable format
        regime_metrics_json = {
            regime: {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in metrics_dict.items()}
            for regime, metrics_dict in regime_metrics.items()
        }
        with open(regime_metrics_path, 'w') as f:
            json.dump(regime_metrics_json, f, indent=2)
        print(f"\nSaved regime metrics: {regime_metrics_path}")
        
    except Exception as e:
        print(f"\nWarning: Could not perform regime analysis: {e}")
        print("Continuing without regime detection...")
        daily_regimes = None
        regime_metrics = None
    
    # ========================================================================
    # PLOT GENERATION
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    print()
    
    # 1. Equity Curve with Regime Background
    print("\nEquity curve...")
    plot_equity_curve(
        daily_pnl,
        dates,
        regime_colors=daily_regimes if daily_regimes is not None else None,
        save_path=plots_dir / 'equity_curve.png',
        title=f'Equity Curve - Top-{args.top_k} Selection'
    )
    plt.close()
    
    # 2. Drawdown Chart
    print(" Drawdown analysis...")
    plot_drawdowns(
        daily_pnl,
        dates,
        save_path=plots_dir / 'drawdowns.png'
    )
    plt.close()
    
    # 3. Rolling Metrics
    print(" Rolling metrics...")
    plot_rolling_metrics(
        daily_pnl,
        dates,
        windows=[30, 60, 90],
        save_path=plots_dir / 'rolling_metrics.png'
    )
    plt.close()
    
    # 4. Regime Performance Comparison
    if regime_metrics is not None:
        print(" Regime comparison...")
        plot_regime_performance(
            regime_metrics,
            save_path=plots_dir / 'regime_performance.png'
        )
        plt.close()
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {diagnostics_dir}")
    print(f"   Metrics:  {diagnostics_dir / 'metrics.json'}")
    if regime_metrics is not None:
        print(f"   Regimes:  {diagnostics_dir / 'regime_metrics.json'}")
    print(f"  Plots:    {plots_dir}")
    print()


if __name__ == '__main__':
    main()
