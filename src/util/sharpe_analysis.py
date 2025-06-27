"""
Analyze Sharpe Ratio Discrepancy

This script analyzes why there's a difference between our manual calculation
and Backtrader's Sharpe ratio calculation.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_json_data(filename):
    """Load JSON data from results file"""
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_sharpe_with_log_returns(trades, initial_capital=1000.0, risk_free_rate=0.01):
    """
    Calculate Sharpe ratio using log returns (closer to Backtrader's method)
    """
    if not trades:
        return 0.0
    
    # Create a portfolio value series
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
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Backtrader:           {backtrader_sharpe:.6f}")
    print(f"Simple Returns:       {sharpe_simple:.6f}")
    print(f"Log Returns:          {result_log['sharpe_ratio']:.6f}")
    
    print("\nPossible reasons for the discrepancy:")
    print("1. Backtrader may use a different annualization factor")
    print("2. Backtrader may handle the risk-free rate differently")
    print("3. Backtrader may use a different return calculation method")
    print("4. Backtrader may include periods with no trades differently")
    print("5. Backtrader may use a different volatility calculation")

if __name__ == "__main__":
    main() 