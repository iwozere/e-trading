#!/usr/bin/env python3
"""
Sentiment Coverage Report for the Short Squeeze Detection Pipeline

Reports per-provider ticker coverage across the candidate universe: what fraction of tickers a
provider (StockTwits, Bluesky, Hacker News, ...) actually returned data for, the median
mentions_24h among tickers it did cover, and how many tickers fall below a minimum-mentions
threshold.

sentiment-spec-rev2.md is explicit that this should run *before* tuning any sentiment thresholds
(§1.4, §2.10): "If Bluesky coverage on your actual candidate universe is under ~20%, the honest
conclusion is that the sentiment term should carry little or no weight in squeeze_score -- and
knowing that is worth more than a feature that looks populated but is mostly imputed neutrals."

This reads *stored* deep-scan history (ss_deep_metrics.raw_payload / .sentiment_data_quality) --
it does not make live API calls. It reports on what the pipeline's scheduled runs actually saw
over the requested window, not a fresh one-off collection.

Usage:
    python coverage_report.py [options]

Examples:
    # Report over the last 7 days against the latest weekly screener universe
    python coverage_report.py

    # Report over the last 30 days
    python coverage_report.py --days 30

    # Report against an explicit ticker list instead of the screener universe
    python coverage_report.py --universe NVDA,TSLA,GME,AMC

    # Custom minimum-mentions threshold and JSON output
    python coverage_report.py --min-mentions 10 --output-dir /tmp/reports
"""

import argparse
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_MIN_MENTIONS = 5


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Report per-provider sentiment coverage across the candidate universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if __doc__ and "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="screener_snapshot",
        help=(
            "'screener_snapshot' (default) to use the latest weekly screener universe, "
            "or a comma-separated ticker list (e.g. NVDA,TSLA,GME)"
        ),
    )
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default: 7)")
    parser.add_argument(
        "--min-mentions",
        type=int,
        default=DEFAULT_MIN_MENTIONS,
        help=f"Mentions threshold below which a ticker is flagged low-confidence (default: {DEFAULT_MIN_MENTIONS})",
    )
    parser.add_argument("--output-dir", type=str, help="Directory to save a JSON report (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def resolve_universe(universe_arg: str, service: ShortSqueezeService) -> List[str]:
    """Resolve --universe into a concrete ticker list."""
    if universe_arg == "screener_snapshot":
        tickers = service.get_universe_from_latest_screener_snapshot()
        _logger.info("Resolved universe from latest screener snapshot: %d tickers", len(tickers))
        return tickers
    tickers = [t.strip().upper() for t in universe_arg.split(",") if t.strip()]
    _logger.info("Using explicit universe: %d tickers", len(tickers))
    return tickers


def compute_coverage(
    universe: List[str], rows: List[Dict[str, Any]], min_mentions: int
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-provider coverage stats from stored deep-scan rows.

    Each row's ``raw_payload`` is ``{provider: {..., "mentions": int, ...}}`` (see
    ``collect_sentiment_async.py``'s ``raw_payload.update(summaries)``), covering every provider
    uniformly -- retail and tech_discourse alike, since both signal classes flow through the same
    per-ticker summary fetch.

    Returns:
        ``{provider: {ticker_coverage_pct, median_mentions_24h, tickers_below_min_mentions,
        tickers_with_zero_mentions, tickers_seen}}``
    """
    universe_set = {t.upper() for t in universe}

    # ticker -> provider -> best mentions count seen across the window
    per_ticker_provider_mentions: Dict[str, Dict[str, int]] = {}
    providers_seen: set[str] = set()

    for row in rows:
        ticker = str(row.get("ticker", "")).upper()
        raw_payload = row.get("raw_payload") or {}
        for provider, summary in raw_payload.items():
            if not isinstance(summary, dict) or "error" in summary:
                continue
            mentions = summary.get("mentions")
            if mentions is None:
                continue
            providers_seen.add(provider)
            bucket = per_ticker_provider_mentions.setdefault(ticker, {})
            # Keep the max seen for this ticker/provider across the window -- a single good day
            # of coverage counts as "covered", matching the spec's ticker_coverage_pct intent.
            bucket[provider] = max(int(mentions), bucket.get(provider, 0))

    report: Dict[str, Dict[str, Any]] = {}
    for provider in sorted(providers_seen):
        covered_tickers = [
            t for t in universe_set if provider in per_ticker_provider_mentions.get(t, {})
        ]
        mention_counts = [per_ticker_provider_mentions[t][provider] for t in covered_tickers]

        report[provider] = {
            "tickers_seen": len(covered_tickers),
            "ticker_coverage_pct": round(100.0 * len(covered_tickers) / len(universe_set), 1) if universe_set else 0.0,
            "median_mentions_24h": statistics.median(mention_counts) if mention_counts else 0,
            "tickers_with_zero_mentions": sum(1 for m in mention_counts if m == 0),
            "tickers_below_min_mentions": sum(1 for m in mention_counts if m < min_mentions),
        }

    return report


def print_report(universe: List[str], days: int, min_mentions: int, report: Dict[str, Dict[str, Any]]) -> None:
    """Log a human-readable coverage report."""
    _logger.info("=== SENTIMENT COVERAGE REPORT ===")
    _logger.info("Universe size: %d tickers | Window: last %d days | min_mentions: %d", len(universe), days, min_mentions)

    if not report:
        _logger.warning("No sentiment data found in ss_deep_metrics for this window -- has daily_deep_scan run recently?")
        return

    for provider, stats in report.items():
        _logger.info(
            "  %-12s coverage=%5.1f%%  seen=%4d/%-4d  median_mentions_24h=%-6s  zero=%-4d  below_min=%-4d",
            provider,
            stats["ticker_coverage_pct"],
            stats["tickers_seen"],
            len(universe),
            stats["median_mentions_24h"],
            stats["tickers_with_zero_mentions"],
            stats["tickers_below_min_mentions"],
        )
        if stats["ticker_coverage_pct"] < 20.0:
            _logger.warning(
                "  %s: coverage under 20%% -- per spec §2.10, this provider likely deserves "
                "little or no weight in squeeze_score until coverage improves",
                provider,
            )

    _logger.info("=== END COVERAGE REPORT ===")


def save_json_report(
    output_dir: str, universe: List[str], days: int, min_mentions: int, report: Dict[str, Dict[str, Any]]
) -> None:
    """Save the report as JSON if an output directory was given."""
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_file = output_path / f"coverage_report_{date.today().isoformat()}.json"

    payload = {
        "generated_at": date.today().isoformat(),
        "universe_size": len(universe),
        "days": days,
        "min_mentions": min_mentions,
        "providers": report,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    _logger.info("JSON report saved to: %s", json_file)


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    if args.days <= 0:
        _logger.error("--days must be positive")
        return 1

    try:
        service = ShortSqueezeService()

        universe = resolve_universe(args.universe, service)
        if not universe:
            _logger.error("Resolved universe is empty -- nothing to report on")
            return 1

        since_date = date.today() - timedelta(days=args.days)
        rows = service.get_deep_scan_metrics_since(since_date)
        _logger.info("Loaded %d deep-scan rows since %s", len(rows), since_date)

        report = compute_coverage(universe, rows, args.min_mentions)
        print_report(universe, args.days, args.min_mentions, report)

        if args.output_dir:
            save_json_report(args.output_dir, universe, args.days, args.min_mentions, report)

        return 0

    except KeyboardInterrupt:
        _logger.warning("Interrupted by user")
        return 130
    except Exception:
        _logger.exception("Coverage report failed:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
