"""
P19 Watchlist Builder (Phase 1, spec §4.1).

Runs once pre-market. Merges the daily watchlist from:
  (a) the latest P17 dated output — Tier B/C + explosive candidates,
  (b) pre-market gappers / most-active < price cap (IBKR market scanner),
  (c) optional manual pins,
then applies the hard filters (§7), dedups, ranks by priority, caps to the IBKR
market-data line budget (N ≤ ~100, §13.2), and writes
``results/p19_penny_intraday/{date}/watchlist.json`` with per-name baseline context.

P17-sourced entries carry full baseline context from the candidates CSV, so the
core build is deterministic and testable without a live feed. The gappers source
needs the IBKR Gateway and degrades to an empty list when it is unreachable.
"""

import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.notification.logger import setup_logger
from src.ml.pipeline.p19_penny_intraday.config import P19Config
from src.ml.pipeline.p19_penny_intraday.models.watchlist_entry import WatchlistEntry

_logger = setup_logger(__name__)

DEFAULT_P17_RESULTS_DIR = "results/p17_penny_stocks"
DEFAULT_OUTPUT_DIR = "results/p19_penny_intraday"

_P17_TIERS = {"A", "B", "C"}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if f == f else default   # guard NaN
    except (TypeError, ValueError):
        return default


def _b(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "1.0", "yes")


class WatchlistBuilder:
    """Build and persist the daily intraday watchlist."""

    def __init__(
        self,
        config: P19Config,
        target_date: str,
        p17_results_dir: str = DEFAULT_P17_RESULTS_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.cfg = config
        self.target_date = target_date
        self.p17_results_dir = p17_results_dir
        self.output_dir = output_dir

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Build, write, and return a summary dict."""
        entries = self.build()
        path = self.write(entries)
        sources = {s: sum(1 for e in entries if e.source == s)
                   for s in ("p17", "gapper", "manual")}
        _logger.info("Watchlist %s: %d names %s → %s",
                     self.target_date, len(entries), sources, path)
        return {"date": self.target_date, "count": len(entries),
                "sources": sources, "path": str(path)}

    def build(self) -> List[WatchlistEntry]:
        """Merge sources → filter → rank → cap."""
        merged: Dict[str, WatchlistEntry] = {}
        # Precedence: manual > p17 > gapper (richer/explicit context wins on dup).
        if self.cfg.use_gappers:
            self._merge(merged, self._from_gappers())
        if self.cfg.use_p17_watchlist:
            self._merge(merged, self._from_p17())
        self._merge(merged, self._from_manual())

        kept = [e for e in merged.values() if self._passes_filters(e)]
        for e in kept:
            e.priority = self._priority(e)
        kept.sort(key=lambda e: e.priority, reverse=True)
        cap = max(1, self.cfg.feed_config.watchlist_cap)
        return kept[:cap]

    # ── Sources ────────────────────────────────────────────────────────────

    def _from_p17(self) -> List[WatchlistEntry]:
        """Tier B/C + explosive names from the latest P17 candidates CSV ≤ target_date."""
        csv = self._latest_p17_csv()
        if not csv:
            _logger.warning("No P17 candidates CSV found under %s", self.p17_results_dir)
            return []
        import pandas as pd
        try:
            df = pd.read_csv(csv)
        except Exception:
            _logger.exception("Could not read P17 CSV %s", csv)
            return []

        out: List[WatchlistEntry] = []
        for r in df.itertuples():
            tier = str(getattr(r, "tier", "") or "")
            explosive = _b(getattr(r, "explosive_candidate", False))
            if tier not in _P17_TIERS and not explosive:
                continue
            ticker = str(getattr(r, "ticker", "")).strip().upper()
            if not ticker or ticker == "NAN":
                continue
            catalyst = str(getattr(r, "catalyst_signals", "") or "")
            signals = [s for s in catalyst.split("|") if s]
            out.append(WatchlistEntry(
                ticker=ticker, source="p17", tier=tier, explosive=explosive,
                company_name=str(getattr(r, "company_name", "") or ""),
                prior_close=_f(getattr(r, "price", 0.0)),
                avg_volume_30d=_f(getattr(r, "avg_volume_30d", 0.0)),
                float_shares=_f(getattr(r, "float_shares", 0.0)),
                market_cap=_f(getattr(r, "market_cap", 0.0)),
                dilution_penalty=_f(getattr(r, "dilution_penalty", 0.0)),
                short_interest_pct_float=(
                    _f(getattr(r, "short_interest_pct_float", None))
                    if getattr(r, "short_interest_pct_float", None) is not None else None),
                has_catalyst=bool(signals) or _f(getattr(r, "catalyst_score", 0.0)) > 0,
                catalyst_signals=signals,
            ))
        _logger.info("P17 source: %d watchlist names from %s", len(out), os.path.basename(csv))
        return out

    def _latest_p17_csv(self) -> str:
        """Most recent ``{date}/{date}_candidates.csv`` with date ≤ target_date."""
        files = sorted(glob.glob(os.path.join(self.p17_results_dir, "*", "*_candidates.csv")))
        eligible = [f for f in files
                    if os.path.basename(os.path.dirname(f)) <= self.target_date]
        return eligible[-1] if eligible else ""

    def _from_gappers(self) -> List[WatchlistEntry]:
        """
        Pre-market gappers / most-active under the price cap via the IBKR scanner.

        Requires the Gateway; returns [] (logged) when unreachable so the build still
        succeeds from the P17 source alone.
        """
        feed = self.cfg.feed_config
        try:
            from ib_insync import IB, ScannerSubscription, TagValue
        except Exception:
            _logger.warning("ib_insync unavailable — gappers source skipped")
            return []

        ib = IB()
        try:
            ib.connect(feed.ibkr_host, feed.ibkr_port,
                       clientId=feed.ibkr_client_id + 1, timeout=10)
        except Exception as e:
            _logger.warning("IBKR Gateway unreachable (%s) — gappers source skipped", e)
            return []

        try:
            sub = ScannerSubscription(
                instrument="STK", locationCode="STK.US.MAJOR", scanCode="TOP_PERC_GAIN")
            tag_values = [
                TagValue("priceBelow", str(self.cfg.filter_config.max_price)),
                TagValue("volumeAbove", str(int(self.cfg.filter_config.min_daily_volume))),
            ]
            rows = ib.reqScannerData(sub, [], tag_values)
            out: List[WatchlistEntry] = []
            for row in rows:
                sym = getattr(getattr(row, "contractDetails", None), "contract", None)
                ticker = (sym.symbol if sym else "").strip().upper()
                if ticker:
                    out.append(WatchlistEntry(ticker=ticker, source="gapper"))
            _logger.info("Gappers source: %d names from IBKR scanner", len(out))
            return out
        except Exception:
            _logger.exception("IBKR scanner failed — gappers source skipped")
            return []
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    def _from_manual(self) -> List[WatchlistEntry]:
        return [WatchlistEntry(ticker=t.strip().upper(), source="manual")
                for t in self.cfg.manual_pins if t.strip()]

    # ── Filter / rank / merge ──────────────────────────────────────────────

    def _passes_filters(self, e: WatchlistEntry) -> bool:
        """Hard filters (§7). Unknown float/volume (gappers) pass on price alone."""
        f = self.cfg.filter_config
        if e.prior_close > 0 and e.prior_close > f.max_price:
            return False
        if e.float_shares > 0 and e.float_shares > f.max_float_shares:
            return False
        if e.avg_volume_30d > 0 and e.avg_volume_30d < f.min_daily_volume:
            return False
        return True

    def _priority(self, e: WatchlistEntry) -> float:
        """Higher = kept first when capping. Explosive/manual rank above plain tiers."""
        base = {"manual": 500.0, "gapper": 150.0}.get(e.source, 0.0)
        if e.source == "p17":
            base = {"A": 400.0, "B": 300.0, "C": 200.0}.get(e.tier, 100.0)
        if e.explosive:
            base += 250.0
        if e.has_catalyst:
            base += 50.0
        base -= e.dilution_penalty        # fade risk lowers priority
        return base

    @staticmethod
    def _merge(acc: Dict[str, WatchlistEntry], new: List[WatchlistEntry]) -> None:
        """Add entries, keeping the *first* seen (call order encodes precedence)."""
        for e in new:
            acc.setdefault(e.ticker, e)

    # ── Output ─────────────────────────────────────────────────────────────

    def write(self, entries: List[WatchlistEntry]) -> Path:
        out_dir = Path(self.output_dir) / self.target_date
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "watchlist.json"
        payload = {
            "date": self.target_date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(entries),
            "watchlist_cap": self.cfg.feed_config.watchlist_cap,
            "entries": [e.to_dict() for e in entries],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path
