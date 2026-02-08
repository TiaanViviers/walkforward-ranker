"""
Performance Metrics Calculation

Calculates comprehensive risk and return metrics for strategy analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def calculate_all_metrics(daily_pnl: pd.Series, capital_per_position: float = 10000.0) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics using institutional standards.
    
    Args:
        daily_pnl: Series of daily PnL values (raw dollars)
        capital_per_position: Notional capital per position (default $10,000)
        
    Returns:
        Dictionary of all calculated metrics (returns-based and dollar-based)
    """
    metrics = {}
    
    # Store assumption
    metrics['capital_per_position'] = capital_per_position
    
    # Calculate returns from PnL (institutional standard)
    daily_returns = daily_pnl / capital_per_position
    
    # Basic stats (returns) - store as percentages for display
    metrics['total_days'] = len(daily_pnl)
    metrics['mean_daily_return'] = daily_returns.mean() * 100  # Convert to %
    metrics['median_daily_return'] = daily_returns.median() * 100  # Convert to %
    metrics['std_daily_return'] = daily_returns.std() * 100  # Convert to %
    metrics['total_return_pct'] = daily_returns.sum() * 100  # Total % return
    
    # Basic stats (raw dollars - for reference)
    metrics['total_return_dollars'] = daily_pnl.sum()
    metrics['mean_daily_pnl'] = daily_pnl.mean()
    metrics['median_daily_pnl'] = daily_pnl.median()
    metrics['std_daily_pnl'] = daily_pnl.std()
    
    # Win/loss stats
    wins = daily_pnl[daily_pnl > 0]
    losses = daily_pnl[daily_pnl < 0]
    
    metrics['win_rate'] = len(wins) / len(daily_pnl) if len(daily_pnl) > 0 else 0
    metrics['avg_win'] = wins.mean() if len(wins) > 0 else 0
    metrics['avg_loss'] = losses.mean() if len(losses) > 0 else 0
    metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
    
    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else 0
    
    # Expectancy
    metrics['expectancy'] = metrics['win_rate'] * metrics['avg_win'] + (1 - metrics['win_rate']) * metrics['avg_loss']
    
    # Risk-adjusted returns (institutional standard - using decimal returns)
    mean_return_decimal = daily_returns.mean()
    std_return_decimal = daily_returns.std()
    
    if std_return_decimal > 0:
        metrics['sharpe_ratio'] = (mean_return_decimal / std_return_decimal) * np.sqrt(252)
    else:
        metrics['sharpe_ratio'] = 0
    
    # Sortino ratio (downside deviation only)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    if downside_std > 0:
        metrics['sortino_ratio'] = (mean_return_decimal / downside_std) * np.sqrt(252)
    else:
        metrics['sortino_ratio'] = 0
    
    # Drawdown analysis
    cumulative = daily_pnl.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    
    metrics['max_drawdown'] = drawdown.min()
    metrics['max_drawdown_pct'] = (drawdown.min() / running_max.max() * 100) if running_max.max() > 0 else 0
    
    # Drawdown duration
    is_underwater = drawdown < 0
    underwater_periods = []
    current_period = 0
    for underwater in is_underwater:
        if underwater:
            current_period += 1
        else:
            if current_period > 0:
                underwater_periods.append(current_period)
            current_period = 0
    if current_period > 0:
        underwater_periods.append(current_period)
    
    metrics['max_drawdown_duration'] = max(underwater_periods) if underwater_periods else 0
    metrics['avg_drawdown_duration'] = np.mean(underwater_periods) if underwater_periods else 0
    
    # Calmar ratio (annualized return / max drawdown %)
    if metrics['max_drawdown'] != 0:
        annual_return_pct = mean_return_decimal * 252 * 100  # Annualized % return
        max_dd_pct = abs(metrics['max_drawdown_pct'])
        metrics['calmar_ratio'] = annual_return_pct / max_dd_pct if max_dd_pct > 0 else 0
    else:
        metrics['calmar_ratio'] = 0
    
    # Losing streak analysis
    metrics['max_losing_streak'] = calculate_max_losing_streak(daily_pnl)
    metrics['max_winning_streak'] = calculate_max_winning_streak(daily_pnl)
    
    # Value at Risk (VaR)
    metrics['var_95'] = daily_pnl.quantile(0.05)  # 95% VaR
    metrics['var_99'] = daily_pnl.quantile(0.01)  # 99% VaR
    
    # Conditional VaR (CVaR / Expected Shortfall)
    metrics['cvar_95'] = daily_pnl[daily_pnl <= metrics['var_95']].mean()
    metrics['cvar_99'] = daily_pnl[daily_pnl <= metrics['var_99']].mean()
    
    # Ulcer Index (pain from drawdowns)
    squared_drawdowns = (drawdown ** 2).mean()
    metrics['ulcer_index'] = np.sqrt(squared_drawdowns)
    
    return metrics


def calculate_max_losing_streak(daily_pnl: pd.Series) -> int:
    """Calculate maximum consecutive losing days."""
    max_streak = 0
    current_streak = 0
    
    for pnl in daily_pnl:
        if pnl < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


def calculate_max_winning_streak(daily_pnl: pd.Series) -> int:
    """Calculate maximum consecutive winning days."""
    max_streak = 0
    current_streak = 0
    
    for pnl in daily_pnl:
        if pnl > 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


def calculate_rolling_metrics(
    daily_pnl: pd.Series,
    window: int = 30
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        daily_pnl: Series of daily PnL values
        window: Rolling window size in days
        
    Returns:
        DataFrame with rolling metrics
    """
    rolling_df = pd.DataFrame(index=daily_pnl.index)
    
    # Rolling mean and std
    rolling_df['mean'] = daily_pnl.rolling(window).mean()
    rolling_df['std'] = daily_pnl.rolling(window).std()
    
    # Rolling Sharpe
    rolling_df['sharpe'] = (rolling_df['mean'] / rolling_df['std']) * np.sqrt(252)
    
    # Rolling cumulative return
    rolling_df['cumulative'] = daily_pnl.rolling(window).sum()
    
    # Rolling win rate
    rolling_df['win_rate'] = daily_pnl.rolling(window).apply(lambda x: (x > 0).sum() / len(x))
    
    return rolling_df


def print_metrics_report(metrics: Dict[str, float]) -> None:
    """Print formatted metrics report to console."""
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    print("\nRETURNS (Percentage)")
    print(f"  Total Return:        {metrics['total_return_pct']:>12.2f}%")
    print(f"  Mean Daily Return:   {metrics['mean_daily_return']:>12.4f}%")
    print(f"  Median Daily Return: {metrics['median_daily_return']:>12.4f}%")
    print(f"  Std Daily Return:    {metrics['std_daily_return']:>12.4f}%")
    
    print("\nRETURNS (Dollars)")
    print(f"  Total Return:        ${metrics['total_return_dollars']:>12,.2f}")
    print(f"  Mean Daily PnL:      ${metrics['mean_daily_pnl']:>12,.2f}")
    print(f"  Median Daily PnL:    ${metrics['median_daily_pnl']:>12,.2f}")
    print(f"  Std Daily PnL:       ${metrics['std_daily_pnl']:>12,.2f}")
    
    print("\nWIN/LOSS")
    print(f"  Win Rate:            {metrics['win_rate']:>12.2%}")
    print(f"  Avg Win:             ${metrics['avg_win']:>12,.2f}")
    print(f"  Avg Loss:            ${metrics['avg_loss']:>12,.2f}")
    print(f"  Win/Loss Ratio:      {metrics['win_loss_ratio']:>12.2f}x")
    print(f"  Profit Factor:       {metrics['profit_factor']:>12.2f}")
    print(f"  Expectancy:          ${metrics['expectancy']:>12,.2f}")
    
    print("\nRISK-ADJUSTED (Returns-Based)")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>12.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>12.2f}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>12.2f}")
    
    print("\nDRAWDOWN")
    print(f"  Max Drawdown:        ${metrics['max_drawdown']:>12,.2f}")
    print(f"  Max Drawdown %:      {metrics['max_drawdown_pct']:>12.1f}%")
    print(f"  Max DD Duration:     {metrics['max_drawdown_duration']:>12.0f} days")
    print(f"  Avg DD Duration:     {metrics['avg_drawdown_duration']:>12.1f} days")
    print(f"  Ulcer Index:         {metrics['ulcer_index']:>12.2f}")
    
    print("\nSTREAKS")
    print(f"  Max Losing Streak:   {metrics['max_losing_streak']:>12.0f} days")
    print(f"  Max Winning Streak:  {metrics['max_winning_streak']:>12.0f} days")
    
    print("\nRISK (VaR)")
    print(f"  VaR 95%:             ${metrics['var_95']:>12,.2f}")
    print(f"  VaR 99%:             ${metrics['var_99']:>12,.2f}")
    print(f"  CVaR 95%:            ${metrics['cvar_95']:>12,.2f}")
    print(f"  CVaR 99%:            ${metrics['cvar_99']:>12,.2f}")
    
    print("\n" + "="*70)
