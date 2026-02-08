"""
Regime Analysis

Analyzes performance across different market regimes (volatility, trend, etc).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple


def detect_volatility_regimes(
    df: pd.DataFrame,
    vol_feature: str = 'pnl_vol_5d',
    n_bins: int = 3
) -> pd.Series:
    """
    Detect volatility regimes based on a volatility feature.
    
    Args:
        df: DataFrame with predictions and features
        vol_feature: Name of volatility feature column
        n_bins: Number of regime bins (3 = low/med/high)
        
    Returns:
        Series with regime labels ('low', 'med', 'high')
    """
    if vol_feature not in df.columns:
        # Fallback: use PnL volatility if available
        if 'pnl' in df.columns:
            daily_vol = df.groupby('date')['pnl'].std()
            # Map back to original df
            df['_temp_vol'] = df['date'].map(daily_vol)
            vol_feature = '_temp_vol'
        else:
            raise ValueError(f"Volatility feature '{vol_feature}' not found in dataframe")
    
    # Bin volatility into regimes
    quantiles = df[vol_feature].quantile([0.33, 0.67]).values
    
    def classify_regime(vol):
        if pd.isna(vol):
            return 'unknown'
        if vol <= quantiles[0]:
            return 'low'
        elif vol <= quantiles[1]:
            return 'med'
        else:
            return 'high'
    
    regimes = df[vol_feature].apply(classify_regime)
    
    # Clean up temp column if created
    if vol_feature == '_temp_vol':
        df.drop('_temp_vol', axis=1, inplace=True)
    
    return regimes


def analyze_by_regime(
    df: pd.DataFrame,
    date_col: str = 'date',
    pnl_col: str = 'pnl',
    regime_col: str = 'regime'
) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance metrics for each regime.
    
    Args:
        df: DataFrame with predictions, PnL, and regime labels
        date_col: Date column name
        pnl_col: PnL column name
        regime_col: Regime label column name
        
    Returns:
        Dictionary of metrics per regime
    """
    from diagnostics.metrics import calculate_all_metrics
    
    regime_metrics = {}
    
    for regime in df[regime_col].unique():
        regime_df = df[df[regime_col] == regime]
        daily_pnl = regime_df.groupby(date_col)[pnl_col].sum()
        
        metrics = calculate_all_metrics(daily_pnl)
        metrics['n_days'] = len(daily_pnl)
        metrics['n_trades'] = len(regime_df)
        
        regime_metrics[regime] = metrics
    
    return regime_metrics


def plot_regime_performance(
    regime_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot performance comparison across regimes.
    
    Args:
        regime_metrics: Dictionary of metrics per regime
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    regimes = list(regime_metrics.keys())
    colors = {'low': '#2E86AB', 'med': '#F18F01', 'high': '#E74C3C'}
    regime_colors = [colors.get(r, '#888888') for r in regimes]
    
    # Metrics to plot
    metrics_to_plot = [
        ('mean_daily_pnl', 'Mean Daily PnL ($)', True),
        ('sharpe_ratio', 'Sharpe Ratio', False),
        ('win_rate', 'Win Rate', True),
        ('profit_factor', 'Profit Factor', False),
        ('max_drawdown', 'Max Drawdown ($)', True),
        ('expectancy', 'Expectancy ($)', True),
    ]
    
    for idx, (metric, label, is_currency) in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [regime_metrics[r][metric] for r in regimes]
        
        bars = ax.bar(regimes, values, color=regime_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if is_currency and 'rate' not in metric.lower():
                label_text = f'${val:,.0f}'
            elif 'rate' in metric.lower():
                label_text = f'{val*100:.1f}%'
            else:
                label_text = f'{val:.2f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label_text, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Volatility Regime', fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add zero line for reference
        if metric in ['mean_daily_pnl', 'max_drawdown', 'expectancy']:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    return fig


def print_regime_comparison(regime_metrics: Dict[str, Dict[str, float]]) -> None:
    """Print formatted regime comparison table."""
    print("\n" + "="*70)
    print("PERFORMANCE BY VOLATILITY REGIME")
    print("="*70)
    
    regimes = list(regime_metrics.keys())
    
    # Key metrics to compare
    metrics = [
        ('n_days', 'Days', '.0f'),
        ('mean_daily_pnl', 'Mean PnL', ',.2f'),
        ('sharpe_ratio', 'Sharpe', '.2f'),
        ('win_rate', 'Win Rate', '.1%'),
        ('profit_factor', 'Profit Factor', '.2f'),
        ('max_drawdown', 'Max DD', ',.0f'),
    ]
    
    # Print header
    print(f"\n{'Metric':<20}", end='')
    for regime in regimes:
        print(f"{regime.upper():>15}", end='')
    print()
    print("-" * 70)
    
    # Print each metric
    for metric_key, metric_name, fmt in metrics:
        print(f"{metric_name:<20}", end='')
        for regime in regimes:
            value = regime_metrics[regime][metric_key]
            if 'rate' in metric_key.lower():
                print(f"{value:>15{fmt}}", end='')
            elif 'pnl' in metric_key.lower() or 'drawdown' in metric_key.lower():
                print(f"${value:>14{fmt}}", end='')
            else:
                print(f"{value:>15{fmt}}", end='')
        print()
    
    print("="*70)
