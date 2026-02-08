"""
Diagnostics and Analysis Tools

Provides comprehensive performance analysis and visualization for ranking models.
"""

from diagnostics.metrics import calculate_all_metrics
from diagnostics.equity_plots import plot_equity_curve, plot_drawdowns, plot_rolling_metrics
from diagnostics.regime_analysis import analyze_by_regime, plot_regime_performance

__all__ = [
    'calculate_all_metrics',
    'plot_equity_curve',
    'plot_drawdowns',
    'plot_rolling_metrics',
    'analyze_by_regime',
    'plot_regime_performance',
]
