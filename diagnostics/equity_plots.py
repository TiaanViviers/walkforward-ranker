"""
Equity Curve and Performance Visualization

Creates publication-quality charts for strategy performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, Tuple


def plot_equity_curve(
    daily_pnl: pd.Series,
    dates: pd.Series,
    regime_colors: Optional[pd.Series] = None,
    save_path: Optional[Path] = None,
    title: str = "Equity Curve"
) -> plt.Figure:
    """
    Plot equity curve with optional regime-colored background.
    
    Args:
        daily_pnl: Daily PnL series
        dates: Date series aligned with daily_pnl
        regime_colors: Optional series of regime labels ('low', 'med', 'high')
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    
    # Calculate cumulative PnL
    cumulative_pnl = daily_pnl.cumsum()
    
    # Plot regime background if provided
    if regime_colors is not None:
        colors_map = {'low': '#e8f4f8', 'med': '#fff9e6', 'high': '#ffe6e6'}
        
        current_regime = None
        start_idx = 0
        
        for i, regime in enumerate(regime_colors):
            if regime != current_regime:
                if current_regime is not None:
                    color = colors_map.get(current_regime, '#ffffff')
                    ax1.axvspan(dates.iloc[start_idx], dates.iloc[i], 
                               alpha=0.3, color=color, zorder=0)
                current_regime = regime
                start_idx = i
        
        # Final regime span
        if current_regime is not None:
            color = colors_map.get(current_regime, '#ffffff')
            ax1.axvspan(dates.iloc[start_idx], dates.iloc[-1], 
                       alpha=0.3, color=color, zorder=0)
    
    # Plot cumulative PnL
    ax1.plot(dates, cumulative_pnl, linewidth=2, color='#2E86AB', label='Cumulative PnL')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_ylabel('Cumulative PnL ($)', fontsize=11, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Plot daily PnL bars
    colors = ['#00A86B' if x > 0 else '#E74C3C' for x in daily_pnl]
    ax2.bar(dates, daily_pnl, color=colors, alpha=0.6, width=1)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    ax2.set_ylabel('Daily PnL ($)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add regime legend if provided
    if regime_colors is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e8f4f8', alpha=0.3, label='Low Vol'),
            Patch(facecolor='#fff9e6', alpha=0.3, label='Med Vol'),
            Patch(facecolor='#ffe6e6', alpha=0.3, label='High Vol')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    return fig


def plot_drawdowns(
    daily_pnl: pd.Series,
    dates: pd.Series,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot underwater (drawdown) chart.
    
    Args:
        daily_pnl: Daily PnL series
        dates: Date series
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Calculate drawdown
    cumulative = daily_pnl.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    drawdown_pct = (drawdown / running_max * 100).fillna(0)
    
    # Plot
    ax.fill_between(dates, drawdown_pct, 0, color='#E74C3C', alpha=0.5)
    ax.plot(dates, drawdown_pct, color='#C0392B', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Styling
    ax.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_title('Underwater Chart (Drawdown Over Time)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Annotate max drawdown
    max_dd_val = drawdown_pct.min()
    # Get integer position of min drawdown
    max_dd_pos = drawdown_pct.values.argmin()
    max_dd_date = dates.iloc[max_dd_pos]
    
    ax.annotate(f'Max DD: {max_dd_val:.1f}%',
                xy=(max_dd_date, max_dd_val),
                xytext=(10, -30), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    return fig


def plot_rolling_metrics(
    daily_pnl: pd.Series,
    dates: pd.Series,
    windows: list = [30, 60, 90],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot rolling performance metrics.
    
    Args:
        daily_pnl: Daily PnL series
        dates: Date series
        windows: List of rolling window sizes
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Rolling Sharpe
    ax = axes[0]
    for window, color in zip(windows, colors):
        rolling_mean = daily_pnl.rolling(window).mean()
        rolling_std = daily_pnl.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        ax.plot(dates, rolling_sharpe, label=f'{window}d', color=color, linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=1, color='green', linestyle=':', alpha=0.3, linewidth=1, label='Sharpe=1')
    ax.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax.set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Rolling Win Rate
    ax = axes[1]
    for window, color in zip(windows, colors):
        rolling_wr = daily_pnl.rolling(window).apply(lambda x: (x > 0).sum() / len(x))
        ax.plot(dates, rolling_wr * 100, label=f'{window}d', color=color, linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='50%')
    ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Rolling Win Rate', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Rolling Mean PnL
    ax = axes[2]
    for window, color in zip(windows, colors):
        rolling_mean = daily_pnl.rolling(window).mean()
        ax.plot(dates, rolling_mean, label=f'{window}d', color=color, linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel('Mean PnL ($)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_title('Rolling Mean Daily PnL', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axes
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    return fig
