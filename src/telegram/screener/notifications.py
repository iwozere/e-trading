import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.telegram.screener.business_logic import handle_command, get_service_instances
from src.telegram.command_parser import ParsedCommand, parse_command
from src.notification.logger import setup_logger
from src.notification.service.client import MessageType, MessagePriority
from src.common.recommendation.engine import RecommendationEngine
_logger = setup_logger(__name__)

# Initialize unified recommendation engine
recommendation_engine = RecommendationEngine()


async def process_report_notifications(result, notification_client, message, user_email):
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
                await notification_client.send_notification(
                    notification_type=MessageType.ERROR,
                    title=f"Report Error for {report['ticker']}",
                    message=f"No data for {report['ticker']}: {report['error']}",
                    priority=MessagePriority.NORMAL,
                    channels=["email"],
                    email_receiver=user_email,
                    attachments=attachments
                )
            else:
                await notification_client.send_notification(
                    notification_type=MessageType.INFO,
                    title=f"Report for {report['ticker']}",
                    message=report["message"],
                    attachments=attachments,
                    priority=MessagePriority.NORMAL,
                    channels=["email"],
                    email_receiver=user_email
                )

    # Telegram notifications - Send directly to avoid queue issues
    _logger.info("Starting Telegram notifications for %d reports", len(result["reports"]))
    for i, report in enumerate(result["reports"]):
        _logger.info("Sending Telegram notification %d/%d for ticker: %s", i + 1, len(result["reports"]), report.get("ticker"))
        if "telegram" in channels:
            attachments = None
            if report.get("chart_bytes"):
                attachments = {f"{report['ticker']}_chart.png": report["chart_bytes"]}

            try:
                if report.get("error"):
                    # Send error notification directly
                    await notification_client.send_notification(
                        notification_type=MessageType.ERROR,
                        title=f"Report Error for {report['ticker']}",
                        message=f"No data for {report['ticker']}: {report['error']}",
                        priority=MessagePriority.NORMAL,
                        channels=["telegram"],
                        recipient_id=str(message.chat.id),
                        data={"reply_to_message_id": message.message_id}
                    )
                    _logger.info("Successfully sent error notification for %s", report['ticker'])
                else:
                    # Create a more concise message for Telegram to avoid length issues
                    telegram_message = _create_telegram_friendly_message(report["message"], report.get("ticker", "Unknown"))

                    # Send success notification directly
                    await notification_client.send_notification(
                        notification_type=MessageType.INFO,
                        title=f"Report for {report['ticker']}",
                        message=telegram_message,
                        priority=MessagePriority.NORMAL,
                        channels=["telegram"],
                        recipient_id=str(message.chat.id),
                        attachments=attachments,
                        data={"reply_to_message_id": message.message_id}
                    )
                    _logger.info("Successfully sent Telegram notification for %s", report['ticker'])
            except Exception:
                _logger.exception("Error sending notification for %s", report['ticker'])
                # Try fallback without attachment
                try:
                    telegram_message = _create_telegram_friendly_message(report["message"], report.get("ticker", "Unknown"))
                    await notification_client.send_notification(
                        notification_type=MessageType.INFO,
                        title=f"Report for {report['ticker']}",
                        message=telegram_message + "\n\n[Chart could not be sent due to an error.]",
                        priority=MessagePriority.NORMAL,
                        channels=["telegram"],
                        recipient_id=str(message.chat.id),
                        data={"reply_to_message_id": message.message_id}
                    )
                    _logger.info("Successfully sent fallback Telegram notification for %s", report['ticker'])
                except Exception:
                    _logger.exception("Error sending fallback notification for %s", report['ticker'])


def _create_telegram_friendly_message(message: str, ticker: str) -> str:
    """
    Create a more concise message for Telegram to avoid length limits.

    Args:
        message: Original message content
        ticker: Stock ticker symbol

    Returns:
        Concise message suitable for Telegram
    """
    # If message is already short enough, return as is
    if len(message) <= 3000:
        return message

    # For very long messages, create a summary
    lines = message.split('\n')
    summary_lines = []

    # Keep the header/title
    for line in lines[:5]:  # Keep first 5 lines (usually header)
        if line.strip():
            summary_lines.append(line)

    # Add a summary indicator
    summary_lines.append("\n📊 **Summary Report** (Full details sent via email)")

    # Try to extract key metrics
    key_metrics = []
    for line in lines:
        if any(keyword in line.lower() for keyword in ['price:', 'pe ratio:', 'roe:', 'rsi:', 'macd:']):
            if len(key_metrics) < 8:  # Limit to 8 key metrics
                key_metrics.append(line.strip())

    if key_metrics:
        summary_lines.append("\n**Key Metrics:**")
        summary_lines.extend(key_metrics[:8])

    summary_lines.append(f"\n💡 Use `/report {ticker} -email` for complete analysis")

    return '\n'.join(summary_lines)


def _create_telegram_friendly_help(help_content: str) -> str:
    """
    Create a more concise help message for Telegram to avoid length limits.

    Args:
        help_content: Original help content

    Returns:
        Concise help message suitable for Telegram
    """
    # If help content is already short enough, return as is
    if len(help_content) <= 3000:
        return help_content

    # For very long help content, create a summary
    lines = help_content.split('\n')
    summary_lines = []

    # Keep the header
    for line in lines[:10]:  # Keep first 10 lines (header and quick start)
        if line.strip():
            summary_lines.append(line)

    # Add a summary indicator
    summary_lines.append("\n📋 **Quick Command Reference**")

    # Extract key command categories
    command_categories = []
    for line in lines:
        if line.startswith('📊') or line.startswith('🚨') or line.startswith('⏰') or line.startswith('🔍') or line.startswith('🔧'):
            if len(command_categories) < 10:  # Limit categories
                command_categories.append(line.strip())

    if command_categories:
        summary_lines.extend(command_categories)

    summary_lines.append("\n💡 **For complete help with examples, use:** `/help -email`")
    summary_lines.append("📞 **Need support?** Contact admin or check admin panel")

    return '\n'.join(summary_lines)


def send_screener_email(email: str, report, config):
    """Send screener results via email."""
    try:
        # Create email content
        # Use screener name if available, otherwise fall back to list_type
        screener_name = getattr(config, 'screener_name', None)
        if screener_name:
            title = screener_name.replace('_', ' ').title()
        else:
            title = config.list_type.replace('_', ' ').title()

        subject = f"📊 {title} Screener Results - {len(report.top_results)} Stocks Found"

        # Build email body
        body = f"""
        <h2>🎯 {title} Screener Results</h2>
        <p><strong>Analysis Summary:</strong></p>
        <ul>
            <li>FMP Pre-filtered: {len(report.fmp_results.get('fmp_results', [])) if hasattr(report, 'fmp_results') and report.fmp_results and 'fmp_results' in report.fmp_results else 'N/A'} stocks</li>
            <li>Processed: {report.total_tickers_processed} tickers</li>
            <li>Found: {len(report.top_results)} matching stocks</li>
        </ul>

        <h3>📋 Top Results:</h3>
        """

        for i, result in enumerate(report.top_results[:20], 1):  # Show top 20
            # Get company name from fundamentals if available
            company_name = ""
            if result.fundamentals and hasattr(result.fundamentals, 'company_name') and result.fundamentals.company_name:
                company_name = f" - {result.fundamentals.company_name}"
            elif result.fundamentals and hasattr(result.fundamentals, 'sector') and result.fundamentals.sector:
                company_name = f" - {result.fundamentals.sector}"

            body += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 8px; background-color: #f9f9f9;">
                <h4 style="color: #2c3e50; margin-top: 0;">#{i}: {result.ticker}{company_name}</h4>
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
                    recommendation = _get_unified_recommendation("PE_RATIO", result.fundamentals.pe_ratio)
                    body += f"<p><strong>P/E Ratio:</strong> {result.fundamentals.pe_ratio:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.forward_pe:
                    recommendation = _get_unified_recommendation("FORWARD_PE", result.fundamentals.forward_pe)
                    body += f"<p><strong>Forward P/E:</strong> {result.fundamentals.forward_pe:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.price_to_book:
                    recommendation = _get_unified_recommendation("PB_RATIO", result.fundamentals.price_to_book)
                    body += f"<p><strong>P/B Ratio:</strong> {result.fundamentals.price_to_book:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.price_to_sales:
                    recommendation = _get_unified_recommendation("PS_RATIO", result.fundamentals.price_to_sales)
                    body += f"<p><strong>P/S Ratio:</strong> {result.fundamentals.price_to_sales:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.peg_ratio:
                    recommendation = _get_unified_recommendation("PEG_RATIO", result.fundamentals.peg_ratio)
                    body += f"<p><strong>PEG Ratio:</strong> {result.fundamentals.peg_ratio:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.return_on_equity:
                    recommendation = _get_unified_recommendation("ROE", result.fundamentals.return_on_equity)
                    body += f"<p><strong>ROE:</strong> {result.fundamentals.return_on_equity:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.return_on_assets:
                    recommendation = _get_unified_recommendation("ROA", result.fundamentals.return_on_assets)
                    body += f"<p><strong>ROA:</strong> {result.fundamentals.return_on_assets:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.debt_to_equity:
                    recommendation = _get_unified_recommendation("DEBT_TO_EQUITY", result.fundamentals.debt_to_equity)
                    body += f"<p><strong>Debt/Equity:</strong> {result.fundamentals.debt_to_equity:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.current_ratio:
                    recommendation = _get_unified_recommendation("CURRENT_RATIO", result.fundamentals.current_ratio)
                    body += f"<p><strong>Current Ratio:</strong> {result.fundamentals.current_ratio:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.quick_ratio:
                    recommendation = _get_unified_recommendation("QUICK_RATIO", result.fundamentals.quick_ratio)
                    body += f"<p><strong>Quick Ratio:</strong> {result.fundamentals.quick_ratio:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.operating_margin:
                    recommendation = _get_unified_recommendation("OPERATING_MARGIN", result.fundamentals.operating_margin)
                    body += f"<p><strong>Operating Margin:</strong> {result.fundamentals.operating_margin:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.profit_margin:
                    recommendation = _get_unified_recommendation("PROFIT_MARGIN", result.fundamentals.profit_margin)
                    body += f"<p><strong>Profit Margin:</strong> {result.fundamentals.profit_margin:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.revenue_growth:
                    recommendation = _get_unified_recommendation("REVENUE_GROWTH", result.fundamentals.revenue_growth)
                    body += f"<p><strong>Revenue Growth:</strong> {result.fundamentals.revenue_growth:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.net_income_growth:
                    recommendation = _get_unified_recommendation("NET_INCOME_GROWTH", result.fundamentals.net_income_growth)
                    body += f"<p><strong>Net Income Growth:</strong> {result.fundamentals.net_income_growth:.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.free_cash_flow:
                    body += f"<p><strong>Free Cash Flow:</strong> ${result.fundamentals.free_cash_flow:,.0f}</p>"
                    has_fundamental_data = True
                if result.fundamentals.dividend_yield:
                    recommendation = _get_unified_recommendation("DIVIDEND_YIELD", result.fundamentals.dividend_yield)
                    body += f"<p><strong>Dividend Yield:</strong> {(result.fundamentals.dividend_yield / 100.0):.2%} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                    has_fundamental_data = True
                if result.fundamentals.payout_ratio:
                    recommendation = _get_unified_recommendation("PAYOUT_RATIO", result.fundamentals.payout_ratio)
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
                    recommendation = _get_unified_recommendation("RSI", result.technicals.rsi)
                    body += f"<p><strong>RSI:</strong> {result.technicals.rsi:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'macd') and result.technicals.macd is not None:
                    context = {}
                    if hasattr(result.technicals, 'macd_signal') and result.technicals.macd_signal is not None:
                        context['macd_signal'] = result.technicals.macd_signal
                    if hasattr(result.technicals, 'macd_histogram') and result.technicals.macd_histogram is not None:
                        context['macd_histogram'] = result.technicals.macd_histogram
                    recommendation = _get_unified_recommendation("MACD", result.technicals.macd, context)
                    body += f"<p><strong>MACD:</strong> {result.technicals.macd:.4f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'macd_signal') and result.technicals.macd_signal is not None:
                    body += f"<p><strong>MACD Signal:</strong> {result.technicals.macd_signal:.4f}</p>"
                if hasattr(result.technicals, 'sma_fast') and result.technicals.sma_fast is not None:
                    context = {'current_price': result.fundamentals.current_price if result.fundamentals else None}
                    recommendation = _get_unified_recommendation("SMA_FAST", result.technicals.sma_fast, context)
                    body += f"<p><strong>SMA Fast:</strong> ${result.technicals.sma_fast:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"

                if hasattr(result.technicals, 'sma_slow') and result.technicals.sma_slow is not None:
                    context = {'current_price': result.fundamentals.current_price if result.fundamentals else None}
                    recommendation = _get_unified_recommendation("SMA_SLOW", result.technicals.sma_slow, context)
                    body += f"<p><strong>SMA Slow:</strong> ${result.technicals.sma_slow:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"

                if hasattr(result.technicals, 'ema_fast') and result.technicals.ema_fast is not None:
                    body += f"<p><strong>EMA Fast:</strong> ${result.technicals.ema_fast:.2f}</p>"

                if hasattr(result.technicals, 'ema_slow') and result.technicals.ema_slow is not None:
                    body += f"<p><strong>EMA Slow:</strong> ${result.technicals.ema_slow:.2f}</p>"

                if hasattr(result.technicals, 'bb_upper') and result.technicals.bb_upper is not None:
                    context = {
                        'current_price': result.fundamentals.current_price if result.fundamentals else None,
                        'bb_upper': result.technicals.bb_upper,
                        'bb_lower': result.technicals.bb_lower if hasattr(result.technicals, 'bb_lower') else None
                    }
                    recommendation = _get_unified_recommendation("BB_UPPER", result.technicals.bb_upper, context)
                    body += f"<p><strong>Bollinger Upper:</strong> ${result.technicals.bb_upper:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'bb_lower') and result.technicals.bb_lower is not None:
                    body += f"<p><strong>Bollinger Lower:</strong> ${result.technicals.bb_lower:.2f}</p>"
                if hasattr(result.technicals, 'bb_middle') and result.technicals.bb_middle is not None:
                    body += f"<p><strong>Bollinger Middle:</strong> ${result.technicals.bb_middle:.2f}</p>"
                if hasattr(result.technicals, 'adx') and result.technicals.adx is not None:
                    recommendation = _get_unified_recommendation("ADX", result.technicals.adx)
                    body += f"<p><strong>ADX:</strong> {result.technicals.adx:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'atr') and result.technicals.atr is not None:
                    body += f"<p><strong>ATR:</strong> {result.technicals.atr:.2f}</p>"
                if hasattr(result.technicals, 'stoch_k') and result.technicals.stoch_k is not None:
                    recommendation = _get_unified_recommendation("STOCH_K", result.technicals.stoch_k)
                    body += f"<p><strong>Stochastic K:</strong> {result.technicals.stoch_k:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'stoch_d') and result.technicals.stoch_d is not None:
                    body += f"<p><strong>Stochastic D:</strong> {result.technicals.stoch_d:.2f}</p>"
                if hasattr(result.technicals, 'williams_r') and result.technicals.williams_r is not None:
                    recommendation = _get_unified_recommendation("WILLIAMS_R", result.technicals.williams_r)
                    body += f"<p><strong>Williams %R:</strong> {result.technicals.williams_r:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'cci') and result.technicals.cci is not None:
                    recommendation = _get_unified_recommendation("CCI", result.technicals.cci)
                    body += f"<p><strong>CCI:</strong> {result.technicals.cci:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'roc') and result.technicals.roc is not None:
                    recommendation = _get_unified_recommendation("ROC", result.technicals.roc)
                    body += f"<p><strong>ROC:</strong> {result.technicals.roc:.2f} <span style='color: {_get_recommendation_color(recommendation)};'>({recommendation})</span></p>"
                if hasattr(result.technicals, 'mfi') and result.technicals.mfi is not None:
                    recommendation = _get_unified_recommendation("MFI", result.technicals.mfi)
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

    except Exception:
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


def _get_unified_recommendation(indicator: str, value: float, context: dict = None) -> str:
    """Get recommendation using unified recommendation engine."""
    try:
        if value is None:
            return "HOLD"

        # Get recommendation from unified engine
        recommendation = recommendation_engine.get_legacy_recommendation(indicator, value, context)
        return recommendation[0]  # Return just the recommendation string
    except Exception as e:
        _logger.warning("Error getting recommendation for %s: %s", indicator, e)
        return "HOLD"




async def process_report_command(message, telegram_user_id, args, notification_client):
    """
    Process /report command using service layer for all operations.

    This handler delegates to business logic which uses:
    - telegram_service for user data and audit logging
    - indicator_service for technical and fundamental analysis

    Service instances are managed globally in business_logic module.
    """
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = await handle_command(parsed)
        user_email = result.get("user_email")
        if result.get("status") == "ok" and "reports" in result:
            await process_report_notifications(result, notification_client, message, user_email)
        else:
            channels = ["telegram"]
            if result.get("email", False):
                channels.append("email")
            from src.notification.service.client import MessageType, MessagePriority
            await notification_client.send_notification(
                notification_type=MessageType.ERROR,
                title=result.get("title", "Report"),
                message=result.get("message", "No message"),
                priority=MessagePriority.NORMAL,
                channels=channels,
                recipient_id=str(message.chat.id),
                email_receiver=user_email if "email" in channels else None,
                data={"reply_to_message_id": message.message_id}
            )
    except Exception as e:
        _logger.exception("Error in report command")
        return {
            "status": "error",
            "message": f"Error processing report command: {str(e)}"
        }

async def process_help_command(message, telegram_user_id, message_text=None, notification_client=None):
    """
    Process /help command using service layer for user data access.

    Service instances are accessed through business_logic.get_service_instances()
    which provides the telegram_service and indicator_service instances
    initialized in bot.py.
    """
    try:
        # Parse command to check for -email flag
        if message_text:
            parsed = parse_command(message_text)
            email_flag = parsed.args.get("email", False)
        else:
            email_flag = False

        # Get user info for email using service layer
        user_email = None
        if email_flag:
            telegram_svc, _ = get_service_instances()
            if telegram_svc:
                user_status = telegram_svc.get_user_status(telegram_user_id)
                if user_status and user_status.get("verified"):
                    user_email = user_status.get("email")
                else:
                    email_flag = False  # Don't send email if user not verified
            else:
                email_flag = False  # Service not available

        # Get help content
        help_content = get_comprehensive_help_content()

        channels = ["telegram"]
        if email_flag and user_email:
            channels.append("email")

        # Create a more concise help message for Telegram
        telegram_help = _create_telegram_friendly_help(help_content)

        # Send Telegram notification
        await notification_client.send_notification(
            notification_type=MessageType.INFO,
            title="Alkotrader Bot - Complete Help Guide",
            message=telegram_help,
            priority=MessagePriority.NORMAL,
            channels=["telegram"],
            recipient_id=str(message.chat.id),
            data={"reply_to_message_id": message.message_id}
        )

        # Send email if requested and user is verified
        if email_flag and user_email:
            await notification_client.send_notification(
                notification_type=MessageType.INFO,
                title="Alkotrader Bot - Complete Help Guide",
                message=help_content,
                priority=MessagePriority.NORMAL,
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

🔍 SCREENER COMMANDS
Immediate Screening:
• /screener SCREENER_NAME [flags] - Run predefined screener immediately
• /screener financial_stocks - Screen financial sector stocks
• /screener small_cap_value -email - Screen small-cap value stocks and send to email
• /screener six_stocks - Screen Swiss stocks on SIX exchange
• /screener mid_cap_stocks -email - Screen mid-cap stocks and send to email
• /screener large_cap_stocks - Screen large-cap stocks
• /screener extra_large_cap_stocks -email - Screen mega-cap stocks and send to email

Available Predefined Screeners:
• conservative_value - Conservative value stocks with low risk
• growth_at_reasonable_price - Growth stocks with reasonable valuations
• dividend_aristocrats - High-quality dividend-paying stocks
• deep_value - Deep value stocks with very low valuations
• quality_growth - High-quality growth stocks with strong fundamentals
• small_cap_value - Small-cap value stocks with growth potential
• defensive_stocks - Defensive stocks with low volatility
• momentum_quality - Quality stocks with positive momentum
• international_value - International value stocks
• tech_growth - Technology growth stocks with reasonable valuations
• financial_stocks - Financial sector stocks with strong fundamentals
• mid_cap_stocks - Mid-cap stocks ($2B - $200B market cap)
• large_cap_stocks - Large-cap stocks ($200B+ market cap)
• six_stocks - Swiss stocks listed on SIX exchange
• extra_large_cap_stocks - Extra large-cap stocks ($500B+ market cap)

Custom JSON Configuration:
• /screener '{"screener_type":"hybrid","list_type":"us_medium_cap",...}' - Use custom JSON config

Flags:
• -email - Send screener results to your verified email

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

async def process_info_command(message, telegram_user_id, notification_client):
    """
    Process /info command using service layer through business logic.

    All database operations are handled through the business logic layer
    which uses the service instances initialized in bot.py.
    """
    try:
        # Parse command to check for -email flag
        parsed = parse_command(message.text)
        email_flag = parsed.args.get("email", False)

        # Get user info for email using service layer
        user_email = None
        if email_flag:
            telegram_svc, _ = get_service_instances()
            if telegram_svc:
                user_status = telegram_svc.get_user_status(telegram_user_id)
                if user_status and user_status.get("verified"):
                    user_email = user_status.get("email")
                else:
                    email_flag = False  # Don't send email if user not verified
            else:
                email_flag = False  # Service not available

        # Add telegram_user_id to parsed command arguments
        parsed.args["telegram_user_id"] = telegram_user_id

        # Get info content
        result = await handle_command(parsed)

        channels = ["telegram"]
        if email_flag and user_email:
            channels.append("email")

        # Send Telegram notification
        await notification_client.send_notification(
            notification_type=MessageType.INFO if result["status"] == "ok" else MessageType.ERROR,
            title=result.get("title", "Info"),
            message=result.get("message", "No message"),
            priority=MessagePriority.NORMAL,
            channels=["telegram"],
            recipient_id=str(message.chat.id),
            data={"reply_to_message_id": message.message_id}
        )

        # Send email if requested and user is verified
        if email_flag and user_email:
            await notification_client.send_notification(
                notification_type=MessageType.INFO if result["status"] == "ok" else MessageType.ERROR,
                title=result.get("title", "Info"),
                message=result.get("message", "No message"),
                priority=MessagePriority.NORMAL,
                channels=["email"],
                email_receiver=user_email
            )

    except Exception as e:
        _logger.exception("Error in info command")
        return {
            "status": "error",
            "message": f"Error processing info command: {str(e)}"
        }

async def process_register_command(message, telegram_user_id, args, notification_client):
    try:
        email = args[1].strip() if len(args) > 1 else None
        language = args[2].strip().lower() if len(args) > 2 else None
        parsed = ParsedCommand(command="register", args={"telegram_user_id": telegram_user_id, "email": email, "language": language})
        result = await handle_command(parsed)
        channels = ["telegram"]

        # Send Telegram notification
        await notification_client.send_notification(
            notification_type=MessageType.INFO if result["status"] == "ok" else MessageType.ERROR,
            title=result.get("title", "Register"),
            message=result.get("message", "No message"),
            priority=MessagePriority.NORMAL,
            channels=["telegram"],
            recipient_id=str(message.chat.id),
            data={"reply_to_message_id": message.message_id}
        )

        # Send verification email if registration was successful
        if result["status"] == "ok" and "email_verification" in result:
            verification_info = result["email_verification"]
            await notification_client.send_notification(
                notification_type=MessageType.INFO,
                title="Your Alkotrader Email Verification Code",
                message=f"Hello,\n\nThank you for registering your email with the Alkotrader Telegram bot.\n\nYour verification code is: {verification_info['code']}\n\nThis code is valid for 1 hour. If you did not request this, please ignore this email.\n\nBest regards,\nAlkotrader Team",
                priority=MessagePriority.NORMAL,
                channels=["email"],
                email_receiver=verification_info["email"]
            )

    except Exception as e:
        _logger.exception("Error in register command")
        return {
            "status": "error",
            "message": f"Error processing register command: {str(e)}"
        }

async def process_verify_command(message, telegram_user_id, args, notification_client):
    try:
        code = args[1] if len(args) > 1 else None
        parsed = ParsedCommand(command="verify", args={"telegram_user_id": telegram_user_id, "code": code})
        result = await handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_client.send_notification(
            notification_type=MessageType.INFO if result["status"] == "ok" else MessageType.ERROR,
            title=result.get("title", "Verify"),
            message=result.get("message", "No message"),
            priority=MessagePriority.NORMAL,
            channels=channels,
            recipient_id=str(message.chat.id),
            email_receiver=user_email if "email" in channels else None,
            data={"reply_to_message_id": message.message_id}
        )
    except Exception:
        _logger.exception("Error in verify command: ")
        await notification_client.send_notification(
            notification_type=MessageType.ERROR,
            title="Verify Command Error",
            message="An error occurred while processing your request.",
            priority=MessagePriority.CRITICAL,
            channels=["telegram"],
            recipient_id=str(message.chat.id),
            data={"reply_to_message_id": message.message_id}
        )

async def process_request_approval_command(message, telegram_user_id, args, notification_client):
    """Process /request_approval command"""
    try:
        # Parse command
        command_text = " ".join(args)
        parsed = parse_command(command_text)
        parsed.args["telegram_user_id"] = telegram_user_id

        # Process command
        result = await handle_command(parsed)

        # Send response to user
        if result["status"] == "ok":
            await notification_client.send_notification(
                notification_type=MessageType.INFO,
                title=result.get("title", "Approval Request"),
                message=result["message"],
                priority=MessagePriority.NORMAL,
                channels=["telegram"],
                recipient_id=str(message.chat.id),
                data={"reply_to_message_id": message.message_id}
            )

            # Notify admins about the approval request
            if result.get("notify_admins"):
                await notification_client.send_notification(
                    notification_type=MessageType.INFO,
                    title="New Approval Request",
                    message=f"User {result['user_id']} ({result['email']}) has requested approval for restricted features.",
                    priority=MessagePriority.HIGH,
                    channels=["telegram"],
                    recipient_id=str(message.chat.id)  # Send to admin chat
                )
        else:
            await notification_client.send_notification(
                notification_type=MessageType.ERROR,
                title="Approval Request Error",
                message=result["message"],
                priority=MessagePriority.NORMAL,
                channels=["telegram"],
                recipient_id=str(message.chat.id),
                data={"reply_to_message_id": message.message_id}
            )

    except Exception:
        _logger.exception("Error processing approval request: ")
        await notification_client.send_notification(
            notification_type=MessageType.ERROR,
            title="Error",
            message="An error occurred while processing your approval request.",
            priority=MessagePriority.NORMAL,
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_language_command(message, telegram_user_id, args, notification_client):
    try:
        lang = args[1].strip().lower() if len(args) > 1 else None
        parsed = ParsedCommand(command="language", args={"telegram_user_id": telegram_user_id, "language": lang})
        result = await handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_client.send_notification(
            notification_type=MessageType.INFO if result["status"] == "ok" else MessageType.ERROR,
            title=result.get("title", "Language"),
            message=result.get("message", "No message"),
            priority=MessagePriority.NORMAL,
            channels=channels,
            recipient_id=str(message.chat.id),
            email_receiver=user_email if "email" in channels else None,
            data={"reply_to_message_id": message.message_id}
        )
    except Exception:
        _logger.exception("Error in language command: ")
        await notification_client.send_notification(
            notification_type=MessageType.ERROR,
            title="Language Command Error",
            message="An error occurred while processing your request.",
            priority=MessagePriority.CRITICAL,
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_admin_command(message, telegram_user_id, args, notification_client):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = await handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")

        # Send response to admin
        await notification_client.send_notification(
            notification_type=MessageType.INFO if result["status"] == "ok" else MessageType.ERROR,
            title=result.get("title", "Admin"),
            message=result.get("message", "No message"),
            priority=MessagePriority.NORMAL,
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
                    await notification_client.send_notification(
                        notification_type=MessageType.INFO,
                        title="Alkotrader Announcement",
                        message=broadcast_message,
                        priority=MessagePriority.NORMAL,
                        channels=["telegram"],
                        recipient_id=str(user_id)
                    )
                except Exception:
                    _logger.exception("Error sending broadcast to user %s:", user_id)

    except Exception:
        _logger.exception("Error in admin command: ")
        await notification_client.send_notification(
            notification_type=MessageType.ERROR,
            title="Admin Command Error",
            message="An error occurred while processing your request.",
            priority=MessagePriority.CRITICAL,
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_alerts_command(message, telegram_user_id, args, notification_client):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = await handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_client.send_notification(
            notification_type=MessageType.INFO if result["status"] == "ok" else MessageType.ERROR,
            title=result.get("title", "Alerts"),
            message=result.get("message", "No message"),
            priority=MessagePriority.NORMAL,
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception:
        _logger.exception("Error in alerts command: ")
        await notification_client.send_notification(
            notification_type=MessageType.ERROR,
            title="Alerts Command Error",
            message="An error occurred while processing your request.",
            priority=MessagePriority.CRITICAL,
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_schedules_command(message, telegram_user_id, args, notification_client):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = await handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_client.send_notification(
            notification_type=MessageType.INFO if result["status"] == "ok" else MessageType.ERROR,
            title=result.get("title", "Schedules"),
            message=result.get("message", "No message"),
            priority=MessagePriority.NORMAL,
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception:
        _logger.exception("Error in schedules command: ")
        await notification_client.send_notification(
            notification_type=MessageType.ERROR,
            title="Schedules Command Error",
            message="An error occurred while processing your request.",
            priority=MessagePriority.CRITICAL,
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_feedback_command(message, telegram_user_id, args, notification_client):
    try:
        feedback_text = args[1] if len(args) > 1 else None
        parsed = ParsedCommand(command="feedback", args={"telegram_user_id": telegram_user_id, "feedback": feedback_text})
        result = await handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_client.send_notification(
            notification_type=MessageType.INFO if result["status"] == "ok" else MessageType.ERROR,
            title=result.get("title", "Feedback"),
            message=result.get("message", "No message"),
            priority=MessagePriority.NORMAL,
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception:
        _logger.exception("Error in feedback command: ")
        await notification_client.send_notification(
            notification_type=MessageType.ERROR,
            title="Feedback Command Error",
            message="An error occurred while processing your request.",
            priority=MessagePriority.CRITICAL,
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_feature_command(message, telegram_user_id, args, notification_client):
    """Process /feature command"""
    try:
        # Parse command
        parsed = parse_command(" ".join(args))
        parsed.args["telegram_user_id"] = telegram_user_id

        # Execute business logic
        result = await handle_command(parsed)

        # Send response
        if result["status"] == "ok":
            await message.answer(result["message"])
        else:
            await message.answer(f"❌ {result['message']}")

    except Exception:
        _logger.exception("Error processing feature command")
        await message.answer("❌ Error processing feature request. Please try again.")


async def process_screener_command(message, telegram_user_id, args, notification_client):
    """Process /screener command"""
    try:
        # Parse command
        parsed = parse_command(" ".join(args))
        parsed.args["telegram_user_id"] = telegram_user_id

        # Execute business logic
        result = await handle_command(parsed)

        # Send response
        if result["status"] == "success":
            if "report" in result:
                # Create a more concise message for Telegram to avoid length issues
                telegram_message = _create_telegram_friendly_message(result["message"], "Screener Results")

                # Send formatted message to Telegram
                await message.answer(telegram_message, parse_mode='Markdown')
            else:
                # Email sent confirmation
                await message.answer(result["message"])
        else:
            await message.answer(f"❌ {result['message']}")

    except Exception:
        _logger.exception("Error processing screener command")
        await message.answer("❌ Error processing screener request. Please try again.")


async def process_unknown_command(message, telegram_user_id, notification_client, help_text):
    try:
        parsed = ParsedCommand(command="unknown", args={"telegram_user_id": telegram_user_id, "text": message.text})

        # Create a user-friendly unknown command message
        unknown_command = message.text.split()[0] if message.text else "unknown"
        unknown_message = f"❓ Unknown command: {unknown_command}\n\nI don't recognize this command. Please use /help to see all available commands and their usage."

        await notification_client.send_notification(
            notification_type=MessageType.ERROR,
            title="Unknown Command",
            message=unknown_message,
            priority=MessagePriority.NORMAL,
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception:
        _logger.exception("Error in unknown command handler: ")
        await notification_client.send_notification(
            notification_type=MessageType.ERROR,
            title="Unknown Command Error",
            message="An error occurred while processing your request.",
            priority=MessagePriority.CRITICAL,
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
