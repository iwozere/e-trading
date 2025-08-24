import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.frontend.telegram.screener.business_logic import handle_command
from src.frontend.telegram.command_parser import ParsedCommand, parse_command
from src.frontend.telegram import db
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


async def process_report_notifications(result, notification_manager, message, user_email):
    channels = ["telegram"]
    if result.get("email", False):
        channels.append("email")

    _logger.info("Processing %d reports for channels: %s", len(result["reports"]), channels)

    # Email notifications
    if "email" in channels:
        for report in result["reports"]:
            attachments = None
            if report.get("chart_bytes"):
                attachments = {f"{report['ticker']}_chart.png": report["chart_bytes"]}
            if report.get("error"):
                await notification_manager.send_notification(
                    notification_type="ERROR",
                    title=f"Report Error for {report['ticker']}",
                    message=f"No data for {report['ticker']}: {report['error']}",
                    priority="NORMAL",
                    channels=["email"],
                    email_receiver=user_email
                )
            else:
                await notification_manager.send_notification(
                    notification_type="INFO",
                    title=f"Report for {report['ticker']}",
                    message=report["message"],
                    attachments=attachments,
                    priority="NORMAL",
                    channels=["email"],
                    email_receiver=user_email
                )
    # Telegram notifications - Send directly to avoid queue issues
    _logger.info("Starting Telegram notifications for %d reports", len(result["reports"]))
    for i, report in enumerate(result["reports"]):
        _logger.info("Sending Telegram notification %d/%d for ticker: %s", i+1, len(result["reports"]), report.get("ticker"))
        if "telegram" in channels:
            attachments = None
            if report.get("chart_bytes"):
                attachments = {f"{report['ticker']}_chart.png": report["chart_bytes"]}

            try:
                if report.get("error"):
                    # Send error notification directly
                    await notification_manager.channels["telegram"].send(
                        notification_manager._create_notification(
                            notification_type="ERROR",
                            title=f"Report Error for {report['ticker']}",
                            message=f"No data for {report['ticker']}: {report['error']}",
                            data={
                                "channels": ["telegram"],
                                "telegram_chat_id": message.chat.id,
                                "reply_to_message_id": message.message_id
                            }
                        )
                    )
                    _logger.info("Successfully sent error notification for %s", report['ticker'])
                else:
                    # Send success notification directly
                    await notification_manager.channels["telegram"].send(
                        notification_manager._create_notification(
                            notification_type="INFO",
                            title=f"Report for {report['ticker']}",
                            message=report["message"],
                            data={
                                "channels": ["telegram"],
                                "telegram_chat_id": message.chat.id,
                                "reply_to_message_id": message.message_id,
                                "attachments": attachments
                            }
                        )
                    )
                    _logger.info("Successfully sent Telegram notification for %s", report['ticker'])
            except Exception as e:
                _logger.exception("Error sending notification for %s", report['ticker'])
                # Try fallback without attachment
                try:
                    await notification_manager.channels["telegram"].send(
                        notification_manager._create_notification(
                            notification_type="INFO",
                            title=f"Report for {report['ticker']}",
                            message=report["message"] + "\n\n[Chart could not be sent due to an error.]",
                            data={
                                "channels": ["telegram"],
                                "telegram_chat_id": message.chat.id,
                                "reply_to_message_id": message.message_id
                            }
                        )
                    )
                    _logger.info("Successfully sent fallback Telegram notification for %s", report['ticker'])
                except Exception as e2:
                    _logger.exception("Error sending fallback notification for %s", report['ticker'])


def send_screener_email(email: str, report, config):
    """Send screener results via email."""
    try:
        # Create email content
        subject = f"📊 {config.list_type.replace('_', ' ').title()} Screener Results - {len(report.top_results)} Stocks Found"

        # Build email body
        body = f"""
        <h2>🎯 {config.list_type.replace('_', ' ').title()} Screener Results</h2>
        <p><strong>Analysis Summary:</strong></p>
        <ul>
            <li>FMP Pre-filtered: {len(report.fmp_results.get('fmp_results', [])) if hasattr(report, 'fmp_results') and report.fmp_results and 'fmp_results' in report.fmp_results else 'N/A'} stocks</li>
            <li>Processed: {report.total_tickers_processed} tickers</li>
            <li>Found: {len(report.top_results)} matching stocks</li>
        </ul>

        <h3>📋 Top Results:</h3>
        """

        for i, result in enumerate(report.top_results[:20], 1):  # Show top 20
            body += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 8px; background-color: #f9f9f9;">
                <h4 style="color: #2c3e50; margin-top: 0;">#{i}: {result.ticker}</h4>
                <p style="font-size: 16px; font-weight: bold; color: #e74c3c;">
                    <strong>Score:</strong> {result.composite_score:.1f}/10 |
                    <strong>Recommendation:</strong> {result.recommendation}
                </p>
            """

            # Fundamental Analysis
            if result.fundamentals:
                body += "<div style='margin: 10px 0;'>"
                body += "<h5 style='color: #27ae60; margin: 10px 0;'>📊 Fundamental Analysis:</h5>"

                # Check if fundamentals has any data
                has_fundamental_data = False

                if result.fundamentals.current_price:
                    body += f"<p><strong>Current Price:</strong> ${result.fundamentals.current_price:.2f}</p>"
                    has_fundamental_data = True
                if result.fundamentals.pe_ratio:
                    recommendation = _get_pe_recommendation(result.fundamentals.pe_ratio)
                    body += f"<p><strong>P/E Ratio:</strong> {result.fundamentals.pe_ratio:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.forward_pe:
                    recommendation = _get_pe_recommendation(result.fundamentals.forward_pe)
                    body += f"<p><strong>Forward P/E:</strong> {result.fundamentals.forward_pe:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.price_to_book:
                    recommendation = _get_pb_recommendation(result.fundamentals.price_to_book)
                    body += f"<p><strong>P/B Ratio:</strong> {result.fundamentals.price_to_book:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.price_to_sales:
                    recommendation = _get_ps_recommendation(result.fundamentals.price_to_sales)
                    body += f"<p><strong>P/S Ratio:</strong> {result.fundamentals.price_to_sales:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.peg_ratio:
                    recommendation = _get_peg_recommendation(result.fundamentals.peg_ratio)
                    body += f"<p><strong>PEG Ratio:</strong> {result.fundamentals.peg_ratio:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.return_on_equity:
                    recommendation = _get_roe_recommendation(result.fundamentals.return_on_equity)
                    body += f"<p><strong>ROE:</strong> {result.fundamentals.return_on_equity:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.return_on_assets:
                    recommendation = _get_roa_recommendation(result.fundamentals.return_on_assets)
                    body += f"<p><strong>ROA:</strong> {result.fundamentals.return_on_assets:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.debt_to_equity:
                    recommendation = _get_debt_equity_recommendation(result.fundamentals.debt_to_equity)
                    body += f"<p><strong>Debt/Equity:</strong> {result.fundamentals.debt_to_equity:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.current_ratio:
                    recommendation = _get_current_ratio_recommendation(result.fundamentals.current_ratio)
                    body += f"<p><strong>Current Ratio:</strong> {result.fundamentals.current_ratio:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.quick_ratio:
                    recommendation = _get_quick_ratio_recommendation(result.fundamentals.quick_ratio)
                    body += f"<p><strong>Quick Ratio:</strong> {result.fundamentals.quick_ratio:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.operating_margin:
                    recommendation = _get_margin_recommendation(result.fundamentals.operating_margin)
                    body += f"<p><strong>Operating Margin:</strong> {result.fundamentals.operating_margin:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.profit_margin:
                    recommendation = _get_margin_recommendation(result.fundamentals.profit_margin)
                    body += f"<p><strong>Profit Margin:</strong> {result.fundamentals.profit_margin:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.revenue_growth:
                    recommendation = _get_growth_recommendation(result.fundamentals.revenue_growth)
                    body += f"<p><strong>Revenue Growth:</strong> {result.fundamentals.revenue_growth:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.net_income_growth:
                    recommendation = _get_growth_recommendation(result.fundamentals.net_income_growth)
                    body += f"<p><strong>Net Income Growth:</strong> {result.fundamentals.net_income_growth:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.free_cash_flow:
                    body += f"<p><strong>Free Cash Flow:</strong> ${result.fundamentals.free_cash_flow:,.0f}</p>"
                    has_fundamental_data = True
                if result.fundamentals.dividend_yield:
                    recommendation = _get_dividend_recommendation(result.fundamentals.dividend_yield)
                    body += f"<p><strong>Dividend Yield:</strong> {(result.fundamentals.dividend_yield / 100.0):.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.payout_ratio:
                    recommendation = _get_payout_recommendation(result.fundamentals.payout_ratio)
                    body += f"<p><strong>Payout Ratio:</strong> {result.fundamentals.payout_ratio:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.market_cap:
                    body += f"<p><strong>Market Cap:</strong> ${result.fundamentals.market_cap:,.0f}</p>"
                    has_fundamental_data = True
                if result.fundamentals.enterprise_value:
                    body += f"<p><strong>Enterprise Value:</strong> ${result.fundamentals.enterprise_value:,.0f}</p>"
                    has_fundamental_data = True
                if result.fundamentals.sector:
                    body += f"<p><strong>Sector:</strong> {result.fundamentals.sector}</p>"
                    has_fundamental_data = True
                if result.fundamentals.industry:
                    body += f"<p><strong>Industry:</strong> {result.fundamentals.industry}</p>"
                    has_fundamental_data = True
                if result.fundamentals.country:
                    body += f"<p><strong>Country:</strong> {result.fundamentals.country}</p>"
                    has_fundamental_data = True

                # Debug: Show if no fundamental data was found
                if not has_fundamental_data:
                    body += "<p><em>No fundamental data available for this ticker</em></p>"

                body += "</div>"

            # Technical Analysis
            if result.technicals:
                body += "<div style='margin: 10px 0;'>"
                body += "<h5 style='color: #3498db; margin: 10px 0;'>📈 Technical Analysis:</h5>"

                if hasattr(result.technicals, 'rsi') and result.technicals.rsi is not None:
                    recommendation = _get_rsi_recommendation(result.technicals.rsi)
                    body += f"<p><strong>RSI:</strong> {result.technicals.rsi:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'macd') and result.technicals.macd is not None:
                    recommendation = _get_macd_recommendation(result.technicals.macd, result.technicals.macd_signal if hasattr(result.technicals, 'macd_signal') else None)
                    body += f"<p><strong>MACD:</strong> {result.technicals.macd:.4f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'macd_signal') and result.technicals.macd_signal is not None:
                    body += f"<p><strong>MACD Signal:</strong> {result.technicals.macd_signal:.4f}</p>"
                if hasattr(result.technicals, 'sma_50') and result.technicals.sma_50 is not None:
                    recommendation = _get_sma_recommendation(result.fundamentals.current_price if result.fundamentals else None, result.technicals.sma_50)
                    body += f"<p><strong>SMA 50:</strong> ${result.technicals.sma_50:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'sma_200') and result.technicals.sma_200 is not None:
                    recommendation = _get_sma_recommendation(result.fundamentals.current_price if result.fundamentals else None, result.technicals.sma_200)
                    body += f"<p><strong>SMA 200:</strong> ${result.technicals.sma_200:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'ema_12') and result.technicals.ema_12 is not None:
                    body += f"<p><strong>EMA 12:</strong> ${result.technicals.ema_12:.2f}</p>"
                if hasattr(result.technicals, 'ema_26') and result.technicals.ema_26 is not None:
                    body += f"<p><strong>EMA 26:</strong> ${result.technicals.ema_26:.2f}</p>"
                if hasattr(result.technicals, 'bb_upper') and result.technicals.bb_upper is not None:
                    recommendation = _get_bollinger_recommendation(result.fundamentals.current_price if result.fundamentals else None, result.technicals.bb_upper, result.technicals.bb_lower if hasattr(result.technicals, 'bb_lower') else None)
                    body += f"<p><strong>Bollinger Upper:</strong> ${result.technicals.bb_upper:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'bb_lower') and result.technicals.bb_lower is not None:
                    body += f"<p><strong>Bollinger Lower:</strong> ${result.technicals.bb_lower:.2f}</p>"
                if hasattr(result.technicals, 'bb_middle') and result.technicals.bb_middle is not None:
                    body += f"<p><strong>Bollinger Middle:</strong> ${result.technicals.bb_middle:.2f}</p>"
                if hasattr(result.technicals, 'adx') and result.technicals.adx is not None:
                    recommendation = _get_adx_recommendation(result.technicals.adx)
                    body += f"<p><strong>ADX:</strong> {result.technicals.adx:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'atr') and result.technicals.atr is not None:
                    body += f"<p><strong>ATR:</strong> {result.technicals.atr:.2f}</p>"
                if hasattr(result.technicals, 'stoch_k') and result.technicals.stoch_k is not None:
                    recommendation = _get_stochastic_recommendation(result.technicals.stoch_k)
                    body += f"<p><strong>Stochastic K:</strong> {result.technicals.stoch_k:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'stoch_d') and result.technicals.stoch_d is not None:
                    body += f"<p><strong>Stochastic D:</strong> {result.technicals.stoch_d:.2f}</p>"
                if hasattr(result.technicals, 'williams_r') and result.technicals.williams_r is not None:
                    recommendation = _get_williams_r_recommendation(result.technicals.williams_r)
                    body += f"<p><strong>Williams %R:</strong> {result.technicals.williams_r:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'cci') and result.technicals.cci is not None:
                    recommendation = _get_cci_recommendation(result.technicals.cci)
                    body += f"<p><strong>CCI:</strong> {result.technicals.cci:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'roc') and result.technicals.roc is not None:
                    recommendation = _get_roc_recommendation(result.technicals.roc)
                    body += f"<p><strong>ROC:</strong> {result.technicals.roc:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'mfi') and result.technicals.mfi is not None:
                    recommendation = _get_mfi_recommendation(result.technicals.mfi)
                    body += f"<p><strong>MFI:</strong> {result.technicals.mfi:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"

                body += "</div>"

            # DCF Analysis
            if result.dcf_valuation and result.dcf_valuation.fair_value:
                dcf = result.dcf_valuation
                if result.fundamentals and result.fundamentals.current_price:
                    current_price = result.fundamentals.current_price
                    fair_value = dcf.fair_value
                    upside = ((fair_value - current_price) / current_price) * 100
                    body += "<div style='margin: 10px 0;'>"
                    body += "<h5 style='color: #f39c12; margin: 10px 0;'>💰 DCF Analysis:</h5>"
                    body += f"<p><strong>Fair Value:</strong> ${fair_value:.2f}</p>"
                    body += f"<p><strong>Current Price:</strong> ${current_price:.2f}</p>"
                    body += f"<p><strong>Upside Potential:</strong> {upside:+.1f}%</p>"
                    body += "</div>"

            body += "</div>"

        if len(report.top_results) > 20:
            body += f"<p><em>... and {len(report.top_results) - 20} more results</em></p>"

        # Send email using existing email infrastructure
        from src.notification.emailer import EmailNotifier
        email_notifier = EmailNotifier()
        email_notifier.send_email(email, subject, body)

        _logger.info("Screener results sent via email to %s", email)

    except Exception as e:
        _logger.exception("Error sending screener email to %s", email)
        raise


def _get_recommendation_color(recommendation: str) -> str:
    """Get color for recommendation."""
    if recommendation == "BUY":
        return "#27ae60"  # Green
    elif recommendation == "SELL":
        return "#e74c3c"  # Red
    elif recommendation == "HOLD":
        return "#f39c12"  # Orange
    else:
        return "#95a5a6"  # Gray


def _get_pe_recommendation(pe_ratio: float) -> str:
    """Get recommendation for P/E ratio."""
    if pe_ratio <= 15:
        return "BUY"
    elif pe_ratio <= 25:
        return "HOLD"
    else:
        return "SELL"


def _get_pb_recommendation(pb_ratio: float) -> str:
    """Get recommendation for P/B ratio."""
    if pb_ratio <= 1.5:
        return "BUY"
    elif pb_ratio <= 3.0:
        return "HOLD"
    else:
        return "SELL"


def _get_ps_recommendation(ps_ratio: float) -> str:
    """Get recommendation for P/S ratio."""
    if ps_ratio <= 1.0:
        return "BUY"
    elif ps_ratio <= 3.0:
        return "HOLD"
    else:
        return "SELL"


def _get_peg_recommendation(peg_ratio: float) -> str:
    """Get recommendation for PEG ratio."""
    if peg_ratio <= 1.0:
        return "BUY"
    elif peg_ratio <= 1.5:
        return "HOLD"
    else:
        return "SELL"


def _get_roe_recommendation(roe: float) -> str:
    """Get recommendation for ROE."""
    if roe >= 0.15:
        return "BUY"
    elif roe >= 0.10:
        return "HOLD"
    else:
        return "SELL"


def _get_roa_recommendation(roa: float) -> str:
    """Get recommendation for ROA."""
    if roa >= 0.05:
        return "BUY"
    elif roa >= 0.03:
        return "HOLD"
    else:
        return "SELL"


def _get_debt_equity_recommendation(debt_equity: float) -> str:
    """Get recommendation for Debt/Equity ratio."""
    if debt_equity <= 0.5:
        return "BUY"
    elif debt_equity <= 1.0:
        return "HOLD"
    else:
        return "SELL"


def _get_current_ratio_recommendation(current_ratio: float) -> str:
    """Get recommendation for Current Ratio."""
    if current_ratio >= 2.0:
        return "BUY"
    elif current_ratio >= 1.5:
        return "HOLD"
    else:
        return "SELL"


def _get_quick_ratio_recommendation(quick_ratio: float) -> str:
    """Get recommendation for Quick Ratio."""
    if quick_ratio >= 1.0:
        return "BUY"
    elif quick_ratio >= 0.8:
        return "HOLD"
    else:
        return "SELL"


def _get_margin_recommendation(margin: float) -> str:
    """Get recommendation for Operating/Profit Margin."""
    if margin >= 0.15:
        return "BUY"
    elif margin >= 0.10:
        return "HOLD"
    else:
        return "SELL"


def _get_growth_recommendation(growth: float) -> str:
    """Get recommendation for Revenue/Net Income Growth."""
    if growth >= 0.10:
        return "BUY"
    elif growth >= 0.05:
        return "HOLD"
    else:
        return "SELL"


def _get_dividend_recommendation(dividend_yield: float) -> str:
    """Get recommendation for Dividend Yield."""
    # dividend_yield is now in percentage format (e.g., 4.0 for 4%)
    if dividend_yield >= 4.0:
        return "BUY"
    elif dividend_yield >= 2.0:
        return "HOLD"
    else:
        return "SELL"


def _get_payout_recommendation(payout_ratio: float) -> str:
    """Get recommendation for Payout Ratio."""
    if payout_ratio <= 0.50:
        return "BUY"
    elif payout_ratio <= 0.75:
        return "HOLD"
    else:
        return "SELL"


def _get_rsi_recommendation(rsi: float) -> str:
    """Get recommendation for RSI."""
    if rsi <= 30:
        return "BUY"
    elif rsi >= 70:
        return "SELL"
    else:
        return "HOLD"


def _get_macd_recommendation(macd: float, macd_signal: float = None) -> str:
    """Get recommendation for MACD."""
    if macd_signal is not None:
        if macd > macd_signal:
            return "BUY"
        else:
            return "SELL"
    else:
        if macd > 0:
            return "BUY"
        else:
            return "SELL"


def _get_sma_recommendation(current_price: float, sma: float) -> str:
    """Get recommendation for SMA."""
    if current_price is None:
        return "HOLD"
    if current_price > sma:
        return "BUY"
    else:
        return "SELL"


def _get_bollinger_recommendation(current_price: float, bb_upper: float, bb_lower: float = None) -> str:
    """Get recommendation for Bollinger Bands."""
    if current_price is None:
        return "HOLD"
    if current_price <= bb_lower:
        return "BUY"
    elif current_price >= bb_upper:
        return "SELL"
    else:
        return "HOLD"


def _get_adx_recommendation(adx: float) -> str:
    """Get recommendation for ADX."""
    if adx >= 25:
        return "BUY"
    else:
        return "HOLD"


def _get_stochastic_recommendation(stoch_k: float) -> str:
    """Get recommendation for Stochastic."""
    if stoch_k <= 20:
        return "BUY"
    elif stoch_k >= 80:
        return "SELL"
    else:
        return "HOLD"


def _get_williams_r_recommendation(williams_r: float) -> str:
    """Get recommendation for Williams %R."""
    if williams_r <= -80:
        return "BUY"
    elif williams_r >= -20:
        return "SELL"
    else:
        return "HOLD"


def _get_cci_recommendation(cci: float) -> str:
    """Get recommendation for CCI."""
    if cci <= -100:
        return "BUY"
    elif cci >= 100:
        return "SELL"
    else:
        return "HOLD"


def _get_roc_recommendation(roc: float) -> str:
    """Get recommendation for ROC."""
    if roc > 0:
        return "BUY"
    else:
        return "SELL"


def _get_mfi_recommendation(mfi: float) -> str:
    """Get recommendation for MFI."""
    if mfi <= 20:
        return "BUY"
    elif mfi >= 80:
        return "SELL"
    else:
        return "HOLD"

async def process_report_command(message, telegram_user_id, args, notification_manager):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = handle_command(parsed)
        user_email = result.get("user_email")
        if result.get("status") == "ok" and "reports" in result:
            await process_report_notifications(result, notification_manager, message, user_email)
        else:
            channels = ["telegram"]
            if result.get("email", False):
                channels.append("email")
            await notification_manager.send_notification(
                notification_type="ERROR",
                title=result.get("title", "Report"),
                message=result.get("message", "No message"),
                priority="NORMAL",
                channels=channels,
                telegram_chat_id=message.chat.id,
                reply_to_message_id=message.message_id,
                email_receiver=user_email if "email" in channels else None
            )
    except Exception as e:
        _logger.exception("Error in report command")
        return {
            "status": "error",
            "message": f"Error processing report command: {str(e)}"
        }

async def process_help_command(message, telegram_user_id, message_text=None, notification_manager=None):
    try:
        # Parse command to check for -email flag
        if message_text:
            parsed = parse_command(message_text)
            email_flag = parsed.args.get("email", False)
        else:
            email_flag = False

        # Get user info for email
        user_email = None
        if email_flag:
            user_status = db.get_user_status(telegram_user_id)
            if user_status and user_status.get("verified"):
                user_email = user_status.get("email")
            else:
                email_flag = False  # Don't send email if user not verified

        # Get help content
        help_content = get_comprehensive_help_content()

        channels = ["telegram"]
        if email_flag and user_email:
            channels.append("email")

        # Send Telegram notification
        await notification_manager.send_notification(
            notification_type="INFO",
            title="Alkotrader Bot - Complete Help Guide",
            message=help_content,
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

        # Send email if requested and user is verified
        if email_flag and user_email:
            await notification_manager.send_notification(
                notification_type="INFO",
                title="Alkotrader Bot - Complete Help Guide",
                message=help_content,
                priority="NORMAL",
                channels=["email"],
                email_receiver=user_email
            )

    except Exception as e:
        _logger.exception("Error in help command")
        return {
            "status": "error",
            "message": f"Error processing help command: {str(e)}"
        }

def get_comprehensive_help_content():
    """Get comprehensive help content for the bot."""
    return """
🤖 ALKOTRADER BOT - COMPLETE COMMAND GUIDE

📋 QUICK START
1. Send /start to begin
2. Register with /register your@email.com
3. Verify your email with the code sent to you
4. Request approval with /request_approval
5. Start using the bot's features!

📊 REPORT COMMANDS
/report TICKER [flags] - Generate comprehensive analysis
Examples:
• /report AAPL - Basic report for Apple
• /report TSLA -email - Report sent to email
• /report BTCUSDT -indicators=RSI,MACD -period=6mo
• /report MSFT -interval=1h -provider=yf

JSON Configuration (Advanced):
• /report -config='{"report_type":"analysis","tickers":["AAPL","MSFT"],"period":"1y","indicators":["RSI","MACD"],"email":true}'
• /report -config='{"report_type":"analysis","tickers":["TSLA"],"period":"6mo","interval":"1h","indicators":["RSI","MACD","BollingerBands"],"include_fundamentals":false}'

Flags:
• -email - Send report to your verified email
• -indicators=RSI,MACD,BollingerBands - Specify indicators
• -period=1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
• -interval=1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
• -provider=yf,alpha_vantage,polygon
• -config=JSON_STRING - Use JSON configuration for advanced options

🚨 ALERT COMMANDS
Price Alerts:
• /alerts add TICKER PRICE CONDITION [flags]
• /alerts add AAPL 150.00 above -email
• /alerts add BTCUSDT 50000 below

Indicator Alerts (Advanced):
• /alerts add_indicator TICKER CONFIG_JSON [flags]
• /alerts add_indicator AAPL '{"type":"indicator","indicator":"RSI","parameters":{"period":14},"condition":{"operator":"<","value":30},"alert_action":"BUY","timeframe":"15m"}' -email

Alert Management:
• /alerts - List all alerts
• /alerts edit ALERT_ID [PRICE] [CONDITION] [flags]
• /alerts delete ALERT_ID
• /alerts pause ALERT_ID
• /alerts resume ALERT_ID

⏰ SCHEDULE COMMANDS
Simple Schedules:
• /schedules add TICKER TIME [flags]
• /schedules add AAPL 09:00 -email
• /schedules add TSLA 16:30 -indicators=RSI,MACD

JSON Schedules (Advanced):
• /schedules add_json CONFIG_JSON
• /schedules add_json '{"type":"report","ticker":"AAPL","scheduled_time":"09:00","period":"1y","interval":"1d","email":true}'

Screener Schedules:
• /schedules screener LIST_TYPE [TIME] [flags]
• /schedules screener us_small_cap 09:00 -email
• /schedules screener us_large_cap -indicators=PE,PB,ROE

Schedule Management:
• /schedules - List all schedules
• /schedules edit SCHEDULE_ID [TIME] [flags]
• /schedules delete SCHEDULE_ID
• /schedules pause SCHEDULE_ID
• /schedules resume SCHEDULE_ID

🔧 UTILITY COMMANDS
Account Management:
• /start - Start the bot
• /help - Show this help message
• /register your@email.com - Register email
• /verify CODE - Verify email
• /request_approval - Request admin approval
• /info - Show account information

📈 TECHNICAL INDICATORS
Available: RSI, MACD, Bollinger Bands, SMA, EMA, ADX, ATR, Stochastic, WilliamsR
Fundamental: P/E, P/B, ROE, ROA, Debt/Equity, Current Ratio, EPS, Revenue, Profit Margin

⚙️ ADVANCED FEATURES
JSON configurations for complex reports, alerts and schedules
Multiple timeframe support (5m, 15m, 1h, 4h, 1d)
Fundamental screening for undervalued stocks
Email notifications for all features

📞 SUPPORT
• Use /help command in the bot
• Contact admin for account issues
• Check admin panel for detailed help

🎯 PRO TIPS:
• Use -email flag for reports and alerts
• Combine multiple indicators for better analysis
• Set up daily schedules for regular monitoring
• Use fundamental screener for stock discovery
• Configure alerts with specific timeframes
• Use JSON configuration for complex report setups

For more detailed help, visit the admin panel help page!
"""

async def process_info_command(message, telegram_user_id, notification_manager):
    try:
        # Parse command to check for -email flag
        parsed = parse_command(message.text)
        email_flag = parsed.args.get("email", False)

        # Get user info for email
        user_email = None
        if email_flag:
            user_status = db.get_user_status(telegram_user_id)
            if user_status and user_status.get("verified"):
                user_email = user_status.get("email")
            else:
                email_flag = False  # Don't send email if user not verified

        # Add telegram_user_id to parsed command arguments
        parsed.args["telegram_user_id"] = telegram_user_id

        # Get info content
        result = handle_command(parsed)

        channels = ["telegram"]
        if email_flag and user_email:
            channels.append("email")

        # Send Telegram notification
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Info"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

        # Send email if requested and user is verified
        if email_flag and user_email:
            await notification_manager.send_notification(
                notification_type="INFO" if result["status"] == "ok" else "ERROR",
                title=result.get("title", "Info"),
                message=result.get("message", "No message"),
                priority="NORMAL",
                channels=["email"],
                email_receiver=user_email
            )

    except Exception as e:
        _logger.exception("Error in info command")
        return {
            "status": "error",
            "message": f"Error processing info command: {str(e)}"
        }

async def process_register_command(message, telegram_user_id, args, notification_manager):
    try:
        email = args[1].strip() if len(args) > 1 else None
        language = args[2].strip().lower() if len(args) > 2 else None
        parsed = ParsedCommand(command="register", args={"telegram_user_id": telegram_user_id, "email": email, "language": language})
        result = handle_command(parsed)
        channels = ["telegram"]

        # Send Telegram notification
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Register"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

        # Send verification email if registration was successful
        if result["status"] == "ok" and "email_verification" in result:
            verification_info = result["email_verification"]
            await notification_manager.send_notification(
                notification_type="INFO",
                title="Your Alkotrader Email Verification Code",
                message=f"Hello,\n\nThank you for registering your email with the Alkotrader Telegram bot.\n\nYour verification code is: {verification_info['code']}\n\nThis code is valid for 1 hour. If you did not request this, please ignore this email.\n\nBest regards,\nAlkotrader Team",
                priority="NORMAL",
                channels=["email"],
                email_receiver=verification_info["email"]
            )

    except Exception as e:
        _logger.exception("Error in register command")
        return {
            "status": "error",
            "message": f"Error processing register command: {str(e)}"
        }

async def process_verify_command(message, telegram_user_id, args, notification_manager):
    try:
        code = args[1] if len(args) > 1 else None
        parsed = ParsedCommand(command="verify", args={"telegram_user_id": telegram_user_id, "code": code})
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Verify"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in verify command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Verify Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_request_approval_command(message, telegram_user_id, args, notification_manager):
    """Process /request_approval command"""
    try:
        # Parse command
        command_text = " ".join(args)
        parsed = parse_command(command_text)
        parsed.args["telegram_user_id"] = telegram_user_id

        # Process command
        result = handle_command(parsed)

        # Send response to user
        if result["status"] == "ok":
            await notification_manager.send_notification(
                notification_type="INFO",
                title=result.get("title", "Approval Request"),
                message=result["message"],
                priority="NORMAL",
                channels=["telegram"],
                telegram_chat_id=message.chat.id,
                reply_to_message_id=message.message_id
            )

            # Notify admins about the approval request
            if result.get("notify_admins"):
                await notification_manager.send_notification(
                    notification_type="INFO",
                    title="New Approval Request",
                    message=f"User {result['user_id']} ({result['email']}) has requested approval for restricted features.",
                    priority="HIGH",
                    channels=["telegram"],
                    telegram_chat_id=message.chat.id  # Send to admin chat
                )
        else:
            await notification_manager.send_notification(
                notification_type="ERROR",
                title="Approval Request Error",
                message=result["message"],
                priority="NORMAL",
                channels=["telegram"],
                telegram_chat_id=message.chat.id,
                reply_to_message_id=message.message_id
            )

    except Exception as e:
        _logger.exception("Error processing approval request: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Error",
            message="An error occurred while processing your approval request.",
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_language_command(message, telegram_user_id, args, notification_manager):
    try:
        lang = args[1].strip().lower() if len(args) > 1 else None
        parsed = ParsedCommand(command="language", args={"telegram_user_id": telegram_user_id, "language": lang})
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Language"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in language command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Language Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_admin_command(message, telegram_user_id, args, notification_manager):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")

        # Send response to admin
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Admin"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

        # Handle broadcast functionality
        if result["status"] == "ok" and "broadcast" in result:
            broadcast_info = result["broadcast"]
            broadcast_message = broadcast_info["message"]
            user_ids = broadcast_info["users"]

            # Send broadcast to all users
            for user_id in user_ids:
                try:
                    await notification_manager.send_notification(
                        notification_type="INFO",
                        title="Alkotrader Announcement",
                        message=broadcast_message,
                        priority="NORMAL",
                        channels=["telegram"],
                        telegram_chat_id=int(user_id)
                    )
                except Exception as e:
                    _logger.error("Error sending broadcast to user %s: %s", user_id, e)

    except Exception as e:
        _logger.exception("Error in admin command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Admin Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_alerts_command(message, telegram_user_id, args, notification_manager):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Alerts"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in alerts command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Alerts Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_schedules_command(message, telegram_user_id, args, notification_manager):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Schedules"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in schedules command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Schedules Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_feedback_command(message, telegram_user_id, args, notification_manager):
    try:
        feedback_text = args[1] if len(args) > 1 else None
        parsed = ParsedCommand(command="feedback", args={"telegram_user_id": telegram_user_id, "feedback": feedback_text})
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Feedback"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in feedback command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Feedback Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_feature_command(message, telegram_user_id, args, notification_manager):
    """Process /feature command"""
    try:
        # Parse command
        parsed = parse_command(" ".join(args))
        parsed.args["telegram_user_id"] = telegram_user_id

        # Execute business logic
        result = handle_command(parsed)

        # Send response
        if result["status"] == "ok":
            await message.answer(result["message"])
        else:
            await message.answer(f"❌ {result['message']}")

    except Exception as e:
        _logger.exception("Error processing feature command")
        await message.answer("❌ Error processing feature request. Please try again.")


async def process_screener_command(message, telegram_user_id, args, notification_manager):
    """Process /screener command"""
    try:
        # Parse command
        parsed = parse_command(" ".join(args))
        parsed.args["telegram_user_id"] = telegram_user_id

        # Execute business logic
        result = handle_command(parsed)

        # Send response
        if result["status"] == "success":
            if "report" in result:
                # Send formatted message to Telegram
                await message.answer(result["message"], parse_mode='Markdown')
            else:
                # Email sent confirmation
                await message.answer(result["message"])
        else:
            await message.answer(f"❌ {result['message']}")

    except Exception as e:
        _logger.exception("Error processing screener command")
        await message.answer("❌ Error processing screener request. Please try again.")


async def process_unknown_command(message, telegram_user_id, notification_manager, help_text):
    try:
        parsed = ParsedCommand(command="unknown", args={"telegram_user_id": telegram_user_id, "text": message.text})

        # Create a user-friendly unknown command message
        unknown_command = message.text.split()[0] if message.text else "unknown"
        unknown_message = f"❓ Unknown command: {unknown_command}\n\nI don't recognize this command. Please use /help to see all available commands and their usage."

        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Unknown Command",
            message=unknown_message,
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in unknown command handler: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Unknown Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
