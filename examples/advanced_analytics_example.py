#!/usr/bin/env python3
"""
Advanced Analytics Example
==========================

This example demonstrates the comprehensive advanced analytics system including:
- Performance metrics calculation
- Monte Carlo simulations
- Risk analysis (VaR, CVaR)
- Strategy comparison and ranking
- Automated reporting
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.analytics.advanced_analytics import (
    AdvancedAnalytics, 
    StrategyComparator, 
    Trade, 
    PerformanceMetrics
)


def create_sample_trades() -> List[Dict[str, Any]]:
    """Create sample trade data for demonstration"""
    
    # Sample trade data
    trades_data = [
        {
            "entry_time": "2024-01-01T09:00:00Z",
            "exit_time": "2024-01-01T10:30:00Z",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "entry_price": 45000.0,
            "exit_price": 45500.0,
            "quantity": 0.1,
            "pnl": 50.0,
            "commission": 0.5,
            "net_pnl": 49.5,
            "exit_reason": "take_profit"
        },
        {
            "entry_time": "2024-01-01T11:00:00Z",
            "exit_time": "2024-01-01T12:15:00Z",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "entry_price": 45500.0,
            "exit_price": 45200.0,
            "quantity": 0.1,
            "pnl": 30.0,
            "commission": 0.5,
            "net_pnl": 29.5,
            "exit_reason": "stop_loss"
        },
        {
            "entry_time": "2024-01-01T14:00:00Z",
            "exit_time": "2024-01-01T15:45:00Z",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "entry_price": 3200.0,
            "exit_price": 3250.0,
            "quantity": 1.0,
            "pnl": 50.0,
            "commission": 3.25,
            "net_pnl": 46.75,
            "exit_reason": "take_profit"
        },
        {
            "entry_time": "2024-01-02T09:00:00Z",
            "exit_time": "2024-01-02T10:00:00Z",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "entry_price": 45200.0,
            "exit_price": 44800.0,
            "quantity": 0.1,
            "pnl": -40.0,
            "commission": 0.5,
            "net_pnl": -40.5,
            "exit_reason": "stop_loss"
        },
        {
            "entry_time": "2024-01-02T11:30:00Z",
            "exit_time": "2024-01-02T13:00:00Z",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "entry_price": 44800.0,
            "exit_price": 45200.0,
            "quantity": 0.1,
            "pnl": 40.0,
            "commission": 0.5,
            "net_pnl": 39.5,
            "exit_reason": "take_profit"
        },
        {
            "entry_time": "2024-01-02T14:30:00Z",
            "exit_time": "2024-01-02T16:00:00Z",
            "symbol": "ETHUSDT",
            "side": "SELL",
            "entry_price": 3250.0,
            "exit_price": 3280.0,
            "quantity": 1.0,
            "pnl": -30.0,
            "commission": 3.28,
            "net_pnl": -33.28,
            "exit_reason": "stop_loss"
        },
        {
            "entry_time": "2024-01-03T09:00:00Z",
            "exit_time": "2024-01-03T10:30:00Z",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "entry_price": 45200.0,
            "exit_price": 45800.0,
            "quantity": 0.1,
            "pnl": 60.0,
            "commission": 0.5,
            "net_pnl": 59.5,
            "exit_reason": "take_profit"
        },
        {
            "entry_time": "2024-01-03T11:00:00Z",
            "exit_time": "2024-01-03T12:00:00Z",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "entry_price": 45800.0,
            "exit_price": 45600.0,
            "quantity": 0.1,
            "pnl": -20.0,
            "commission": 0.5,
            "net_pnl": -20.5,
            "exit_reason": "stop_loss"
        },
        {
            "entry_time": "2024-01-03T14:00:00Z",
            "exit_time": "2024-01-03T15:30:00Z",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "entry_price": 3280.0,
            "exit_price": 3320.0,
            "quantity": 1.0,
            "pnl": 40.0,
            "commission": 3.32,
            "net_pnl": 36.68,
            "exit_reason": "take_profit"
        },
        {
            "entry_time": "2024-01-04T09:00:00Z",
            "exit_time": "2024-01-04T10:00:00Z",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "entry_price": 45600.0,
            "exit_price": 45300.0,
            "quantity": 0.1,
            "pnl": 30.0,
            "commission": 0.5,
            "net_pnl": 29.5,
            "exit_reason": "take_profit"
        }
    ]
    
    return trades_data


def create_strategy_trades(strategy_name: str, base_trades: List[Dict], 
                          performance_multiplier: float = 1.0) -> List[Dict]:
    """Create trades for a specific strategy with performance variation"""
    
    strategy_trades = []
    for i, trade in enumerate(base_trades):
        # Modify trade data for strategy variation
        modified_trade = trade.copy()
        modified_trade["strategy"] = strategy_name
        
        # Apply performance multiplier
        modified_trade["net_pnl"] *= performance_multiplier
        modified_trade["pnl"] *= performance_multiplier
        
        # Add some randomness
        import random
        random_factor = random.uniform(0.8, 1.2)
        modified_trade["net_pnl"] *= random_factor
        modified_trade["pnl"] *= random_factor
        
        strategy_trades.append(modified_trade)
    
    return strategy_trades


def demonstrate_basic_analytics():
    """Demonstrate basic performance metrics calculation"""
    
    print("=" * 60)
    print("BASIC PERFORMANCE ANALYTICS")
    print("=" * 60)
    
    # Create analytics instance
    analytics = AdvancedAnalytics(risk_free_rate=0.02)
    
    # Add sample trades
    trades_data = create_sample_trades()
    analytics.add_trades(trades_data)
    
    # Calculate metrics
    metrics = analytics.calculate_metrics()
    
    # Display results
    print(f"📊 Performance Summary:")
    print(f"   Total Trades: {metrics.total_trades}")
    print(f"   Win Rate: {metrics.win_rate:.2f}%")
    print(f"   Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   Total Return: ${metrics.total_return:.2f}")
    print(f"   Total Return %: {metrics.total_return_pct:.2f}%")
    print()
    
    print(f"📈 Risk Metrics:")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"   Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"   Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"   VaR (95%): ${metrics.var_95:.2f}")
    print(f"   CVaR (95%): ${metrics.cvar_95:.2f}")
    print()
    
    print(f"🎯 Trade Analysis:")
    print(f"   Average Win: ${metrics.avg_win:.2f}")
    print(f"   Average Loss: ${metrics.avg_loss:.2f}")
    print(f"   Largest Win: ${metrics.largest_win:.2f}")
    print(f"   Largest Loss: ${metrics.largest_loss:.2f}")
    print(f"   Max Consecutive Wins: {metrics.max_consecutive_wins}")
    print(f"   Max Consecutive Losses: {metrics.max_consecutive_losses}")
    print()
    
    print(f"⏱️ Time Analysis:")
    print(f"   Average Trade Duration: {metrics.avg_trade_duration}")
    print(f"   Average Trades per Day: {metrics.avg_trades_per_day:.2f}")
    print()
    
    print(f"💰 Additional Metrics:")
    print(f"   Kelly Criterion: {metrics.kelly_criterion:.4f}")
    print(f"   Expectancy: ${metrics.expectancy:.2f}")
    print(f"   Recovery Factor: {metrics.recovery_factor:.2f}")
    print(f"   Payoff Ratio: {metrics.payoff_ratio:.2f}")
    
    return analytics


def demonstrate_monte_carlo_simulation():
    """Demonstrate Monte Carlo simulation"""
    
    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION")
    print("=" * 60)
    
    # Create analytics instance
    analytics = AdvancedAnalytics()
    trades_data = create_sample_trades()
    analytics.add_trades(trades_data)
    
    # Run Monte Carlo simulation
    print("🔄 Running Monte Carlo simulation (10,000 scenarios)...")
    simulation_results = analytics.run_monte_carlo_simulation(
        n_simulations=10000,
        n_trades=100
    )
    
    if "error" in simulation_results:
        print(f"❌ Simulation Error: {simulation_results['error']}")
        return
    
    print(f"📊 Simulation Results:")
    print(f"   Mean Return: ${simulation_results['mean_return']:.2f}")
    print(f"   Standard Deviation: ${simulation_results['std_return']:.2f}")
    print(f"   VaR (95%): ${simulation_results['var_95']:.2f}")
    print(f"   CVaR (95%): ${simulation_results['cvar_95']:.2f}")
    print(f"   Probability of Profit: {simulation_results['prob_profit']:.2f}%")
    print(f"   Min Return: ${simulation_results['min_return']:.2f}")
    print(f"   Max Return: ${simulation_results['max_return']:.2f}")
    print()
    
    print(f"📈 Percentile Analysis:")
    percentiles = simulation_results['percentiles']
    for p, value in percentiles.items():
        print(f"   {p}th Percentile: ${value:.2f}")
    
    # Risk assessment
    print(f"\n⚠️ Risk Assessment:")
    if simulation_results['prob_profit'] > 60:
        print("   ✅ High probability of profit (>60%)")
    elif simulation_results['prob_profit'] > 40:
        print("   ⚠️ Moderate probability of profit (40-60%)")
    else:
        print("   ❌ Low probability of profit (<40%)")
    
    if simulation_results['var_95'] > -100:
        print("   ✅ Low downside risk (VaR > -$100)")
    else:
        print("   ⚠️ High downside risk (VaR < -$100)")


def demonstrate_strategy_comparison():
    """Demonstrate strategy comparison and ranking"""
    
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON & RANKING")
    print("=" * 60)
    
    # Create strategy comparator
    comparator = StrategyComparator()
    
    # Create different strategies with varying performance
    base_trades = create_sample_trades()
    
    strategies = [
        ("RSI_BB_Strategy", 1.0),      # Base performance
        ("MACD_Strategy", 1.2),        # 20% better
        ("Supertrend_Strategy", 0.8),  # 20% worse
        ("Bollinger_Strategy", 1.1),   # 10% better
        ("Ichimoku_Strategy", 0.9),    # 10% worse
    ]
    
    print("🔄 Creating strategy analytics...")
    
    for strategy_name, performance_multiplier in strategies:
        # Create trades for this strategy
        strategy_trades = create_strategy_trades(strategy_name, base_trades, performance_multiplier)
        
        # Create analytics instance for this strategy
        strategy_analytics = AdvancedAnalytics()
        strategy_analytics.add_trades(strategy_trades)
        
        # Add to comparator
        comparator.add_strategy(strategy_name, strategy_analytics)
    
    # Compare strategies
    print("📊 Strategy Comparison Table:")
    comparison_df = comparator.compare_strategies()
    print(comparison_df.to_string(index=False))
    print()
    
    # Rank strategies
    print("🏆 Strategy Rankings:")
    rankings = comparator.rank_strategies()
    for strategy, rank in rankings.items():
        print(f"   {rank}. {strategy}")
    
    # Get best strategy
    best_strategy = min(rankings.items(), key=lambda x: x[1])[0]
    print(f"\n🥇 Best Strategy: {best_strategy}")
    
    # Get detailed metrics for best strategy
    best_analytics = comparator.strategies[best_strategy]
    best_metrics = best_analytics.calculate_metrics()
    
    print(f"📈 Best Strategy Metrics:")
    print(f"   Sharpe Ratio: {best_metrics.sharpe_ratio:.2f}")
    print(f"   Win Rate: {best_metrics.win_rate:.2f}%")
    print(f"   Profit Factor: {best_metrics.profit_factor:.2f}")
    print(f"   Max Drawdown: {best_metrics.max_drawdown_pct:.2f}%")


def demonstrate_automated_reporting():
    """Demonstrate automated report generation"""
    
    print("\n" + "=" * 60)
    print("AUTOMATED REPORT GENERATION")
    print("=" * 60)
    
    # Create analytics instance
    analytics = AdvancedAnalytics()
    trades_data = create_sample_trades()
    analytics.add_trades(trades_data)
    
    # Calculate metrics
    analytics.calculate_metrics()
    
    # Generate reports
    print("📄 Generating performance reports...")
    
    try:
        report_path = analytics.generate_performance_report("reports")
        print(f"✅ Reports generated successfully!")
        print(f"📁 Report location: {report_path}")
        
        # List generated files
        if os.path.exists("reports"):
            print(f"\n📋 Generated Files:")
            for file in os.listdir("reports"):
                if file.startswith("performance_report_"):
                    file_path = os.path.join("reports", file)
                    file_size = os.path.getsize(file_path)
                    print(f"   📄 {file} ({file_size} bytes)")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        print("💡 Make sure you have reportlab and openpyxl installed:")
        print("   pip install reportlab openpyxl")


def demonstrate_risk_analysis():
    """Demonstrate comprehensive risk analysis"""
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE RISK ANALYSIS")
    print("=" * 60)
    
    # Create analytics instance
    analytics = AdvancedAnalytics()
    trades_data = create_sample_trades()
    analytics.add_trades(trades_data)
    
    # Calculate metrics
    metrics = analytics.calculate_metrics()
    
    print(f"🎯 Risk Assessment:")
    print()
    
    # Drawdown analysis
    print(f"📉 Drawdown Analysis:")
    if metrics.max_drawdown_pct < 10:
        print("   ✅ Excellent: Max drawdown < 10%")
    elif metrics.max_drawdown_pct < 20:
        print("   ⚠️ Good: Max drawdown 10-20%")
    elif metrics.max_drawdown_pct < 30:
        print("   ⚠️ Moderate: Max drawdown 20-30%")
    else:
        print("   ❌ High: Max drawdown > 30%")
    
    # Sharpe ratio analysis
    print(f"\n📊 Risk-Adjusted Returns:")
    if metrics.sharpe_ratio > 2.0:
        print("   ✅ Excellent: Sharpe ratio > 2.0")
    elif metrics.sharpe_ratio > 1.0:
        print("   ⚠️ Good: Sharpe ratio 1.0-2.0")
    elif metrics.sharpe_ratio > 0.5:
        print("   ⚠️ Moderate: Sharpe ratio 0.5-1.0")
    else:
        print("   ❌ Poor: Sharpe ratio < 0.5")
    
    # Win rate analysis
    print(f"\n🎯 Win Rate Analysis:")
    if metrics.win_rate > 60:
        print("   ✅ Excellent: Win rate > 60%")
    elif metrics.win_rate > 50:
        print("   ⚠️ Good: Win rate 50-60%")
    elif metrics.win_rate > 40:
        print("   ⚠️ Moderate: Win rate 40-50%")
    else:
        print("   ❌ Poor: Win rate < 40%")
    
    # Profit factor analysis
    print(f"\n💰 Profit Factor Analysis:")
    if metrics.profit_factor > 2.0:
        print("   ✅ Excellent: Profit factor > 2.0")
    elif metrics.profit_factor > 1.5:
        print("   ⚠️ Good: Profit factor 1.5-2.0")
    elif metrics.profit_factor > 1.2:
        print("   ⚠️ Moderate: Profit factor 1.2-1.5")
    else:
        print("   ❌ Poor: Profit factor < 1.2")
    
    # Consecutive losses analysis
    print(f"\n🔴 Consecutive Losses:")
    if metrics.max_consecutive_losses <= 3:
        print("   ✅ Excellent: Max consecutive losses ≤ 3")
    elif metrics.max_consecutive_losses <= 5:
        print("   ⚠️ Good: Max consecutive losses 4-5")
    elif metrics.max_consecutive_losses <= 7:
        print("   ⚠️ Moderate: Max consecutive losses 6-7")
    else:
        print("   ❌ High: Max consecutive losses > 7")
    
    # Kelly criterion analysis
    print(f"\n🎲 Kelly Criterion:")
    if metrics.kelly_criterion > 0.25:
        print("   ✅ Excellent: Kelly criterion > 0.25")
    elif metrics.kelly_criterion > 0.1:
        print("   ⚠️ Good: Kelly criterion 0.1-0.25")
    elif metrics.kelly_criterion > 0.05:
        print("   ⚠️ Moderate: Kelly criterion 0.05-0.1")
    else:
        print("   ❌ Poor: Kelly criterion < 0.05")


def demonstrate_integration_with_alert_system():
    """Demonstrate integration with alert system"""
    
    print("\n" + "=" * 60)
    print("INTEGRATION WITH ALERT SYSTEM")
    print("=" * 60)
    
    # Create analytics instance
    analytics = AdvancedAnalytics()
    trades_data = create_sample_trades()
    analytics.add_trades(trades_data)
    
    # Calculate metrics
    metrics = analytics.calculate_metrics()
    
    print("🔔 Simulating alert system integration...")
    
    # Simulate alert conditions
    alerts = []
    
    if metrics.max_drawdown_pct > 15:
        alerts.append(f"🚨 High Drawdown Alert: {metrics.max_drawdown_pct:.2f}%")
    
    if metrics.sharpe_ratio < 1.0:
        alerts.append(f"📉 Low Sharpe Ratio Alert: {metrics.sharpe_ratio:.2f}")
    
    if metrics.win_rate < 50:
        alerts.append(f"🎯 Low Win Rate Alert: {metrics.win_rate:.2f}%")
    
    if metrics.max_consecutive_losses > 5:
        alerts.append(f"🔴 High Consecutive Losses Alert: {metrics.max_consecutive_losses}")
    
    if metrics.profit_factor < 1.5:
        alerts.append(f"💰 Low Profit Factor Alert: {metrics.profit_factor:.2f}")
    
    # Display alerts
    if alerts:
        print("⚠️ Generated Alerts:")
        for alert in alerts:
            print(f"   {alert}")
    else:
        print("✅ No alerts generated - all metrics within acceptable ranges")
    
    # Simulate performance metrics for alert system
    print(f"\n📊 Performance Metrics for Alert System:")
    performance_metrics = {
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "daily_pnl": metrics.total_return_pct / 4,  # Approximate daily PnL
        "max_consecutive_losses": metrics.max_consecutive_losses,
        "sharpe_ratio": metrics.sharpe_ratio,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor
    }
    
    for metric, value in performance_metrics.items():
        print(f"   {metric}: {value:.2f}")


def main():
    """Main function to run all demonstrations"""
    
    print("🚀 ADVANCED ANALYTICS SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This example demonstrates the comprehensive advanced analytics system")
    print("including performance metrics, Monte Carlo simulations, risk analysis,")
    print("strategy comparison, and automated reporting.")
    print()
    
    try:
        # Create reports directory
        os.makedirs("reports", exist_ok=True)
        
        # Run all demonstrations
        analytics = demonstrate_basic_analytics()
        demonstrate_monte_carlo_simulation()
        demonstrate_strategy_comparison()
        demonstrate_risk_analysis()
        demonstrate_automated_reporting()
        demonstrate_integration_with_alert_system()
        
        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The advanced analytics system provides comprehensive analysis")
        print("capabilities for trading strategy evaluation and optimization.")
        print()
        print("📁 Check the 'reports/' directory for generated reports.")
        print("📊 Use the metrics and insights for strategy improvement.")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        print("💡 Make sure all required dependencies are installed:")
        print("   pip install pandas numpy matplotlib seaborn scipy")
        print("   pip install reportlab openpyxl")


if __name__ == "__main__":
    main()
