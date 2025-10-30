"""
Telegram Management Example Usage

Demonstrates how to use the Telegram management module's main functions.
"""
from src.management.telegram.commands import trading_commands, analysis_commands, portfolio_commands, admin_commands
from src.management.telegram.notifications import trade_alerts, risk_alerts, performance_reports, market_updates
from src.management.telegram.integrations import voice_commands, chart_generator, sentiment_analysis

# --- Commands ---
print(trading_commands.start_trading(user_id=12345))
print(trading_commands.stop_trading(user_id=12345))
print(trading_commands.trading_status(user_id=12345))

print(analysis_commands.send_chart(user_id=12345, symbol='BTCUSDT'))
print(analysis_commands.send_report(user_id=12345, report_type='daily'))

print(portfolio_commands.get_portfolio_overview(user_id=12345))

print(admin_commands.add_admin(user_id=12345, new_admin_id=67890))
print(admin_commands.remove_admin(user_id=12345, admin_id=67890))
print(admin_commands.list_admins())

# --- Notifications ---
trade_info = {'symbol': 'BTCUSDT', 'side': 'buy', 'price': 30000, 'qty': 0.1}
print(trade_alerts.send_trade_alert(user_id=12345, trade_info=trade_info))

risk_info = {'type': 'drawdown', 'message': 'Max drawdown exceeded!'}
print(risk_alerts.send_risk_alert(user_id=12345, risk_info=risk_info))

report = {'pnl': 500, 'win_rate': 0.6}
print(performance_reports.send_performance_report(user_id=12345, report=report, period='weekly'))

update = {'headline': 'BTC breaks $30k', 'details': 'Strong buying pressure.'}
print(market_updates.send_market_update(user_id=12345, update=update))

# --- Integrations ---
print(voice_commands.process_voice_command(user_id=12345, audio_data='audio_file.ogg'))
print(chart_generator.generate_chart(symbol='BTCUSDT', timeframe='4h'))
print(sentiment_analysis.analyze_sentiment(message='I am bullish on BTC!')) 