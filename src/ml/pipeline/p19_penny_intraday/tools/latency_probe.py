"""
P19 intraday data-feed probe.

Measures the *free-tier* capability matrix that decides the P19 architecture
(spec §13 rate budget, open question #1): for Finnhub and Polygon —

  * is the real-time **quote** endpoint reachable on the free key?
  * are intraday **candles / aggregates** available on free, or premium-gated?
  * what throughput (req/min) before a 429 rate-limit?
  * how stale is the latest bar/quote timestamp?

NOTE: real-time *latency* is only meaningful **during US market hours**
(Mon–Fri ~13:30–20:00 UTC). Run then to get true staleness; off-hours the
"staleness" just reflects time since the last session close. Throughput and
endpoint-availability are valid any time.

Usage:
    python -m src.ml.pipeline.p19_penny_intraday.tools.latency_probe
"""

import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Real small-cap penny names seen this session.
TICKERS = [
    "ILLR",
    "CPOP",
    "SCAG",
    "QTEX",
    "NTCL",
    "CAST",
    "LASE",
    "MNTS",
    "DPRO",
    "TGHL",
    "XOS",
    "FTHM",
    "BOLD",
    "UNCY",
    "JRSH",
]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        env = PROJECT_ROOT / "config" / "donotshare" / ".env"
        load_dotenv(dotenv_path=env, override=False)
    except Exception:
        _logger.warning("python-dotenv not available / .env not loaded")


def _age(ts_epoch: float | None) -> str:
    if not ts_epoch:
        return "n/a"
    delta = datetime.now(UTC) - datetime.fromtimestamp(ts_epoch, tz=UTC)
    return f"{delta.total_seconds() / 60:.1f} min ago"


def probe_finnhub(key: str) -> None:
    print("\n=== FINNHUB (free) ===")
    if not key:
        print("  no FINNHUB_API_KEY — skipped")
        return

    # 1) real-time quote endpoint + throughput
    lat = []
    last_ts = None
    rate_limited_at = None
    for i, sym in enumerate(TICKERS, 1):
        t0 = time.monotonic()
        try:
            r = requests.get("https://finnhub.io/api/v1/quote", params={"symbol": sym, "token": key}, timeout=10)
        except Exception as e:
            print(f"  quote {sym}: ERROR {e}")
            continue
        dt = (time.monotonic() - t0) * 1000
        lat.append(dt)
        if r.status_code == 429:
            rate_limited_at = i
            print(f"  quote {sym}: 429 rate-limited after {i} calls")
            break
        if r.status_code != 200:
            print(f"  quote {sym}: HTTP {r.status_code}")
            continue
        j = r.json()
        last_ts = j.get("t") or last_ts
        if i <= 3:
            print(f"  quote {sym}: price={j.get('c')} ts={_age(j.get('t'))} ({dt:.0f}ms)")
    if lat:
        print(
            f"  quote: {len(lat)} calls, median {sorted(lat)[len(lat) // 2]:.0f}ms, "
            f"rate-limited={'yes @' + str(rate_limited_at) if rate_limited_at else 'no'}"
        )
        print(f"  latest quote timestamp staleness: {_age(last_ts)}")

    # 2) intraday candles (free or premium-gated?)
    now = int(time.time())
    r = requests.get(
        "https://finnhub.io/api/v1/stock/candle",
        params={"symbol": "AAPL", "resolution": "5", "from": now - 3 * 86400, "to": now, "token": key},
        timeout=10,
    )
    print(
        f"  intraday candle (5m): HTTP {r.status_code} "
        f"{'(premium-gated on free)' if r.status_code in (401, 403) else r.text[:60]}"
    )


def probe_polygon(key: str) -> None:
    print("\n=== POLYGON (free) ===")
    if not key:
        print("  no POLYGON_API_KEY — skipped")
        return

    # intraday aggregates + rate limit (free ~5/min)
    end = datetime.now(UTC).date()
    start = end - timedelta(days=5)
    rate_limited_at = None
    last_bar = None
    ok = 0
    for i, sym in enumerate(TICKERS[:8], 1):
        url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/5/minute/{start}/{end}"
        try:
            r = requests.get(url, params={"apiKey": key, "limit": 5, "sort": "desc"}, timeout=10)
        except Exception as e:
            print(f"  aggs {sym}: ERROR {e}")
            continue
        if r.status_code == 429:
            rate_limited_at = i
            print(f"  aggs {sym}: 429 rate-limited after {i} calls")
            break
        if r.status_code != 200:
            print(f"  aggs {sym}: HTTP {r.status_code} {r.text[:60]}")
            continue
        ok += 1
        res = r.json().get("results") or []
        if res:
            last_bar = res[0].get("t", 0) / 1000.0
        if i <= 3:
            print(f"  aggs {sym}: {len(res)} bars, last bar {_age(last_bar)}")
    print(f"  aggs: {ok} ok, rate-limited={'yes @' + str(rate_limited_at) if rate_limited_at else 'no'}")
    if last_bar:
        print(f"  latest 5m bar staleness: {_age(last_bar)} (free tier is typically 15-min delayed)")

    # real-time snapshot (usually premium)
    r = requests.get(
        "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/AAPL", params={"apiKey": key}, timeout=10
    )
    print(
        f"  snapshot (real-time): HTTP {r.status_code} "
        f"{'(premium-gated on free)' if r.status_code in (401, 403) else 'ok'}"
    )


def probe_ibkr() -> None:
    """
    Probe the IBKR paper Gateway (run on the Pi, during market hours).

    Confirms p19's chosen primary feed: delayed 5m bars that **carry volume**.
    Reports connection, last-bar staleness, and whether volume is present.
    """
    print("\n=== IBKR GATEWAY (paper, delayed) ===")
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PAPER_PORT", "4002"))
    client_id = 19  # unique to p19; avoid clashing with running bots
    try:
        from ib_async import IB, Stock
    except ImportError:
        try:
            from ib_insync import IB, Stock
        except Exception:
            print("  ib_async/ib_insync not installed — skipped")
            return

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
    except Exception as e:
        print(f"  connect {host}:{port} FAILED: {e}")
        print("  (run this on the Pi where the Gateway Docker is reachable)")
        return

    print(f"  connected {host}:{port} (clientId={client_id})")
    ib.reqMarketDataType(3)  # 3 = delayed (free)
    for sym in TICKERS[:5]:
        try:
            bars = ib.reqHistoricalData(
                Stock(sym, "SMART", "USD"),
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=2,
            )
        except Exception as e:
            print(f"  {sym}: reqHistoricalData ERROR {e}")
            continue
        if not bars:
            print(f"  {sym}: no bars (no subscription / no data)")
            continue
        last = bars[-1]
        ts = last.date if isinstance(last.date, datetime) else None
        age = _age(ts.timestamp()) if ts else "n/a"
        has_vol = getattr(last, "volume", 0) and last.volume > 0
        print(
            f"  {sym}: {len(bars)} bars, last close={last.close} vol={last.volume} "
            f"({age}) volume_present={'YES' if has_vol else 'no'}"
        )
    ib.disconnect()
    print("  → if volume_present=YES, IBKR delivers real RVOL (15-min delayed) — primary feed confirmed")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="P19 feed probe")
    ap.add_argument("--ibkr", action="store_true", help="Probe the IBKR paper Gateway (run on the Pi)")
    ap.add_argument("--rest", action="store_true", help="Probe Finnhub + Polygon REST tiers")
    args = ap.parse_args()

    _load_env()
    print(f"P19 feed probe @ {datetime.now(UTC):%Y-%m-%d %H:%M UTC %A}")
    print("(latency/staleness only meaningful during market hours Mon-Fri 13:30-20:00 UTC)")
    # default (no flags) = REST probe, as before
    if args.ibkr:
        probe_ibkr()
    if args.rest or not args.ibkr:
        probe_finnhub(os.getenv("FINNHUB_API_KEY", ""))
        probe_polygon(os.getenv("POLYGON_API_KEY", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
