import argparse
import logging
import os
import sys

# Ensure the root directory is in sys.path to allow relative imports if run as script
# But better to run as module: python -m src.ml.pipeline.p13_bdsh.run_p13
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ml.pipeline.p13_bdsh.p13_pipeline import P13Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run VIX-Threshold Scaling Strategy (Pipeline p13)")
    parser.add_argument("--tickers", type=str, default="SPY", help="Comma-separated list of tickers (e.g. SPY,QQQ,TLT)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--live", action="store_true", help="Print today's production summary")
    parser.add_argument(
        "--output", type=str, choices=["text", "json"], default="text", help="Output format for summary"
    )
    parser.add_argument("--notify", action="store_true", help="Send notification via Telegram")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Logging level
    loglevel = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=loglevel, format="%(asctime)s - %(levelname)s - %(message)s")

    # Convert tickers to list
    ticker_list = [t.strip().upper() for t in args.tickers.split(",")]

    # Initialize and run pipeline
    pipeline = P13Pipeline(tickers=ticker_list, start_date=args.start, end_date=args.end)

    try:
        pipeline.run()

        if args.live:
            # Load the state we just saved
            state = pipeline.load_state()

            # Structured summary for potential API/Notification use
            summary_info = []
            for ticker in ticker_list:
                if ticker in state:
                    s = state[ticker]
                    status = "EXIT / NEUTRAL"
                    if s["active_exposure"] > 0:
                        tier = "3 (Max)" if s["active_exposure"] > 0.9 else ("2" if s["active_exposure"] > 0.4 else "1")
                        status = f"HOLD - Tier {tier} Active"
                    if s["in_cooldown"]:
                        status = "COOLDOWN (Stop-Loss triggered)"

                    summary_info.append(
                        {
                            "ticker": ticker,
                            "vix_z": s["current_vix_z"],
                            "status": status,
                            "stop_loss": s["stop_loss_price"],
                        }
                    )

            if args.output == "json":
                import json

                print(json.dumps(summary_info, indent=4))
            else:
                print("\n" + "=" * 80)
                print(" PRODUCTION SIGNALING SUMMARY")
                print("=" * 80)
                for item in summary_info:
                    print(
                        f"Ticker: [{item['ticker']}] | Current VIX Z: [{item['vix_z']:.2f}] | Status: [{item['status']}] | Stop-Loss: [${item['stop_loss']:.2f}]"
                    )
                print("=" * 80 + "\n")

            # Send to Telegram if requested
            if args.notify:
                try:
                    from src.notification.notification_service import NotificationService

                    service = NotificationService()
                    message = "📊 *P13 Daily Signals*\n\n"
                    for item in summary_info:
                        message += f"• *{item['ticker']}*: {item['status']}\n  (VIX Z: {item['vix_z']:.2f}, SL: ${item['stop_loss']:.2f})\n"

                    service.send_telegram(message)
                    logging.info("Telegram notification sent.")
                except Exception as ne:
                    logging.error(f"Failed to send notification: {ne}")

    except Exception as e:
        logging.error(f"Pipeline crashed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
