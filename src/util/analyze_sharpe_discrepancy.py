"""
Analyze Sharpe Ratio Discrepancy

This script analyzes why there's a difference between our manual calculation
and Backtrader's Sharpe ratio calculation.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import backtrader as bt

def load_json_data(filename):
    """Load JSON data from results file"""
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_backtrader_methodology():
    """
    Analyze how Backtrader calculates Sharpe ratio
    """
    print("Backtrader Sharpe Ratio Analysis")
    print("=" * 50)
    
    # Load the JSON data
    json_file = "results/BTCUSDT_1h_20230101_20250501_RSIBBVolumeEntryMixin_RSIBBExitMixin_20250621_095351.json"
    data = load_json_data(json_file)
    
    trades = data['trades']
    backtrader_sharpe = data['analyzers']['sharpe']['sharperatio']
    
    print(f"Backtrader Sharpe Ratio: {backtrader_sharpe:.6f}")
    
    # Key differences in Backtrader's calculation:
    print("\nBacktrader Sharpe Ratio Methodology:")
    print("1. Uses log returns instead of simple returns")
    print("2. Calculates returns on a per-bar basis, not per-trade")
    print("3. Uses different annualization factors")
    print("4. May use different risk-free rate handling")
    
    return backtrader_sharpe

def calculate_sharpe_with_log_returns(trades, initial_capital=1000.0, risk_free_rate=0.01):
    """
    Calculate Sharpe ratio using log returns (closer to Backtrader's method)
    """
    if not trades:
        return 0.0
    
    # Create a portfolio value series
    portfolio_values = []
    dates = []
    current_value = initial_capital
    
    # Group trades by date
    daily_pnl = {}
    
    for trade in trades:
        exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
        trade_date = exit_time.date()
        
        if trade_date not in daily_pnl:
            daily_pnl[trade_date] = 0.0
        
        daily_pnl[trade_date] += trade['net_pnl']
    
    # Create portfolio value series
    dates = sorted(daily_pnl.keys())
    portfolio_values = [initial_capital]
    
    for date in dates:
        current_value += daily_pnl[date]
        portfolio_values.append(current_value)
    
    # Calculate log returns
    log_returns = []
    for i in range(1, len(portfolio_values)):
        log_return = np.log(portfolio_values[i] / portfolio_values[i-1])
        log_returns.append(log_return)
    
    if not log_returns:
        return 0.0
    
    returns = np.array(log_returns)
    
    # Calculate metrics
    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    # Annualize (assuming daily data)
    annualized_return = avg_return * 252
    annualized_volatility = std_return * np.sqrt(252)
    
    # Calculate Sharpe ratio
    if annualized_volatility == 0:
        return 0.0
    
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'avg_daily_return': avg_return,
        'daily_volatility': std_return,
        'method': 'log_returns'
    }

def calculate_sharpe_per_bar_basis(trades, initial_capital=1000.0, risk_free_rate=0.01):
    """
    Calculate Sharpe ratio on a per-bar basis (like Backtrader)
    """
    if not trades:
        return 0.0
    
    # Create a time series of portfolio values
    # We need to interpolate between trades to get daily values
    
    # Get all unique dates from trades
    all_dates = set()
    for trade in trades:
        exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
        all_dates.add(exit_time.date())
    
    # Create a complete date range
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    # Create daily portfolio values
    daily_values = {}
    current_value = initial_capital
    
    # Initialize all dates with the current value
    current_date = start_date
    while current_date <= end_date:
        daily_values[current_date] = current_value
        current_date += timedelta(days=1)
    
    # Apply trade P&Ls on their exit dates
    for trade in trades:
        exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
        exit_date = exit_time.date()
        
        # Update all subsequent dates with the new value
        current_date = exit_date
        while current_date <= end_date:
            daily_values[current_date] += trade['net_pnl']
            current_date += timedelta(days=1)
    
    # Calculate daily returns
    dates = sorted(daily_values.keys())
    returns = []
    
    for i in range(1, len(dates)):
        prev_value = daily_values[dates[i-1]]
        curr_value = daily_values[dates[i]]
        
        if prev_value > 0:
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
    
    if not returns:
        return 0.0
    
    returns = np.array(returns)
    
    # Calculate metrics
    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    # Annualize
    annualized_return = avg_return * 252
    annualized_volatility = std_return * np.sqrt(252)
    
    # Calculate Sharpe ratio
    if annualized_volatility == 0:
        return 0.0
    
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'avg_daily_return': avg_return,
        'daily_volatility': std_return,
        'method': 'per_bar_basis',
        'num_days': len(returns)
    }

def main():
    """Main analysis function"""
    print("Sharpe Ratio Discrepancy Analysis")
    print("=" * 60)
    
    # Load data
    json_file = "results/BTCUSDT_1h_20230101_20250501_RSIBBVolumeEntryMixin_RSIBBExitMixin_20250621_095351.json"
    data = load_json_data(json_file)
    trades = data['trades']
    backtrader_sharpe = data['analyzers']['sharpe']['sharperatio']
    
    print(f"Backtrader Sharpe Ratio: {backtrader_sharpe:.6f}")
    print(f"Number of trades: {len(trades)}")
    
    # Test different calculation methods
    print("\n" + "-" * 40)
    print("Method 1: Simple Returns (Trade-based)")
    print("-" * 40)
    
    # Re-run the simple returns calculation
    daily_pnl = {}
    for trade in trades:
        exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
        trade_date = exit_time.date()
        if trade_date not in daily_pnl:
            daily_pnl[trade_date] = 0.0
        daily_pnl[trade_date] += trade['net_pnl']
    
    dates = sorted(daily_pnl.keys())
    current_capital = 1000.0
    daily_returns = []
    
    for date in dates:
        daily_return = daily_pnl[date] / current_capital
        daily_returns.append(daily_return)
        current_capital += daily_pnl[date]
    
    returns = np.array(daily_returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    annualized_return = avg_return * 252
    annualized_volatility = std_return * np.sqrt(252)
    sharpe_simple = (annualized_return - 0.01) / annualized_volatility
    
    print(f"Simple Returns Sharpe: {sharpe_simple:.6f}")
    print(f"Annualized Return: {annualized_return:.6f}")
    print(f"Annualized Volatility: {annualized_volatility:.6f}")
    
    print("\n" + "-" * 40)
    print("Method 2: Log Returns")
    print("-" * 40)
    
    result_log = calculate_sharpe_with_log_returns(trades)
    print(f"Log Returns Sharpe: {result_log['sharpe_ratio']:.6f}")
    print(f"Annualized Return: {result_log['annualized_return']:.6f}")
    print(f"Annualized Volatility: {result_log['annualized_volatility']:.6f}")
    
    print("\n" + "-" * 40)
    print("Method 3: Per-Bar Basis")
    print("-" * 40)
    
    result_bar = calculate_sharpe_per_bar_basis(trades)
    print(f"Per-Bar Sharpe: {result_bar['sharpe_ratio']:.6f}")
    print(f"Annualized Return: {result_bar['annualized_return']:.6f}")
    print(f"Annualized Volatility: {result_bar['annualized_volatility']:.6f}")
    print(f"Number of days: {result_bar['num_days']}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Backtrader:           {backtrader_sharpe:.6f}")
    print(f"Simple Returns:       {sharpe_simple:.6f}")
    print(f"Log Returns:          {result_log['sharpe_ratio']:.6f}")
    print(f"Per-Bar Basis:        {result_bar['sharpe_ratio']:.6f}")
    
    print("\nPossible reasons for the discrepancy:")
    print("1. Backtrader may use a different annualization factor")
    print("2. Backtrader may handle the risk-free rate differently")
    print("3. Backtrader may use a different return calculation method")
    print("4. Backtrader may include periods with no trades differently")
    print("5. Backtrader may use a different volatility calculation")

if __name__ == "__main__":
    main() 