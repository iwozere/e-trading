"""
Test Sharpe Ratio Calculation

This script tests the Sharpe ratio calculation to understand why the values might be low.
It recreates the calculation using the same parameters as the optimizer.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_json_data(filename):
    """Load JSON data from results file"""
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_sharpe_ratio_manual(trades, initial_capital=1000.0, risk_free_rate=0.01):
    """
    Calculate Sharpe ratio manually to verify the calculation
    
    Sharpe Ratio = (Return - Risk Free Rate) / Standard Deviation of Returns
    """
    if not trades:
        return 0.0
    
    # Calculate daily returns from trades
    daily_returns = []
    current_capital = initial_capital
    
    # Group trades by date and calculate daily P&L
    daily_pnl = {}
    
    for trade in trades:
        # Parse entry and exit times
        entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
        exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
        
        # Use exit date as the date for the trade
        trade_date = exit_time.date()
        
        if trade_date not in daily_pnl:
            daily_pnl[trade_date] = 0.0
        
        daily_pnl[trade_date] += trade['net_pnl']
    
    # Calculate daily returns
    dates = sorted(daily_pnl.keys())
    for i, date in enumerate(dates):
        daily_return = daily_pnl[date] / current_capital
        daily_returns.append(daily_return)
        current_capital += daily_pnl[date]
    
    if not daily_returns:
        return 0.0
    
    # Convert to numpy array
    returns = np.array(daily_returns)
    
    # Calculate metrics
    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # Sample standard deviation
    
    # Annualize returns and volatility
    # Assuming daily data, multiply by 252 (trading days per year)
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
        'risk_free_rate': risk_free_rate,
        'num_trading_days': len(daily_returns),
        'total_return': np.sum(returns),
        'daily_returns': daily_returns
    }

def test_backtrader_sharpe_calculation():
    """Test how Backtrader calculates Sharpe ratio"""
    print("Testing Sharpe Ratio Calculation")
    print("=" * 50)
    
    # Load the JSON data
    json_file = "results/BTCUSDT_1h_20230101_20250501_RSIBBVolumeEntryMixin_RSIBBExitMixin_20250621_095351.json"
    data = load_json_data(json_file)
    
    # Extract trades
    trades = data['trades']
    print(f"Number of trades: {len(trades)}")
    
    # Get configuration
    initial_capital = 1000.0
    risk_free_rate = 0.01  # 1% as set in optimizer.json
    
    print(f"Initial capital: ${initial_capital}")
    print(f"Risk-free rate: {risk_free_rate:.3f} ({risk_free_rate*100:.1f}%)")
    
    # Calculate Sharpe ratio manually
    result = calculate_sharpe_ratio_manual(trades, initial_capital, risk_free_rate)
    
    print("\nManual Calculation Results:")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.6f}")
    print(f"Annualized Return: {result['annualized_return']:.6f} ({result['annualized_return']*100:.2f}%)")
    print(f"Annualized Volatility: {result['annualized_volatility']:.6f} ({result['annualized_volatility']*100:.2f}%)")
    print(f"Average Daily Return: {result['avg_daily_return']:.6f} ({result['avg_daily_return']*100:.4f}%)")
    print(f"Daily Volatility: {result['daily_volatility']:.6f} ({result['daily_volatility']*100:.4f}%)")
    print(f"Total Return: {result['total_return']:.6f} ({result['total_return']*100:.2f}%)")
    print(f"Number of Trading Days: {result['num_trading_days']}")
    
    # Compare with Backtrader result
    backtrader_sharpe = data['analyzers']['sharpe']['sharperatio']
    print(f"\nBacktrader Sharpe Ratio: {backtrader_sharpe:.6f}")
    print(f"Difference: {abs(result['sharpe_ratio'] - backtrader_sharpe):.6f}")
    
    # Analyze why Sharpe ratio might be low
    print("\nAnalysis:")
    print(f"Strategy Performance: {result['annualized_return']*100:.2f}% annual return")
    print(f"Risk: {result['annualized_volatility']*100:.2f}% annual volatility")
    print(f"Risk-Adjusted Return: {result['sharpe_ratio']:.3f}")
    
    if result['sharpe_ratio'] < 1.0:
        print("\nSharpe ratio is low because:")
        if result['annualized_return'] < risk_free_rate:
            print("- Strategy return is below risk-free rate")
        if result['annualized_volatility'] > 0.5:
            print("- High volatility is penalizing the ratio")
        if result['annualized_return'] < 0.1:
            print("- Low absolute returns")
    
    # Show some trade statistics
    net_pnls = [trade['net_pnl'] for trade in trades]
    print(f"\nTrade Statistics:")
    print(f"Average trade P&L: ${np.mean(net_pnls):.2f}")
    print(f"Trade P&L std dev: ${np.std(net_pnls):.2f}")
    print(f"Min trade P&L: ${min(net_pnls):.2f}")
    print(f"Max trade P&L: ${max(net_pnls):.2f}")
    
    return result

def test_different_risk_free_rates():
    """Test how different risk-free rates affect Sharpe ratio"""
    print("\n" + "=" * 50)
    print("Testing Different Risk-Free Rates")
    print("=" * 50)
    
    json_file = "results/BTCUSDT_1h_20230101_20250501_RSIBBVolumeEntryMixin_RSIBBExitMixin_20250621_095351.json"
    data = load_json_data(json_file)
    trades = data['trades']
    
    risk_free_rates = [0.0, 0.01, 0.02, 0.03, 0.05]
    
    for rfr in risk_free_rates:
        result = calculate_sharpe_ratio_manual(trades, 1000.0, rfr)
        print(f"Risk-free rate: {rfr*100:.1f}% -> Sharpe: {result['sharpe_ratio']:.3f}")

if __name__ == "__main__":
    test_backtrader_sharpe_calculation()
    test_different_risk_free_rates() 