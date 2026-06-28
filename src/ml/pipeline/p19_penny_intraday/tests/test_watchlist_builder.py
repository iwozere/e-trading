"""Tests for the P19 WatchlistBuilder (P17 source + filters + rank + output)."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p19_penny_intraday.config import P19Config
from src.ml.pipeline.p19_penny_intraday.watchlist_builder import WatchlistBuilder


_COLS = ["ticker", "company_name", "price", "market_cap", "avg_volume_30d",
         "float_shares", "dilution_penalty", "short_interest_pct_float",
         "catalyst_signals", "catalyst_score", "tier", "explosive_candidate"]


def _row(ticker, tier="C", price=3.0, explosive=False, float_shares=8_000_000,
         avg_vol=1_000_000, dilution=0.0, catalyst="", catalyst_score=0.0):
    return {"ticker": ticker, "company_name": f"{ticker} Inc", "price": price,
            "market_cap": 50_000_000, "avg_volume_30d": avg_vol,
            "float_shares": float_shares, "dilution_penalty": dilution,
            "short_interest_pct_float": 0.1, "catalyst_signals": catalyst,
            "catalyst_score": catalyst_score, "tier": tier,
            "explosive_candidate": explosive}


def _p17_dir(tmp_path, date, rows):
    d = tmp_path / "p17" / date
    d.mkdir(parents=True)
    pd.DataFrame(rows, columns=_COLS).to_csv(d / f"{date}_candidates.csv", index=False)
    return str(tmp_path / "p17")


def _builder(tmp_path, rows, date="2026-06-26", **cfg_over):
    p17 = _p17_dir(tmp_path, date, rows)
    cfg = P19Config.create_default()
    cfg.use_gappers = False                       # isolate P17 source in tests
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    return WatchlistBuilder(cfg, date, p17_results_dir=p17,
                            output_dir=str(tmp_path / "p19"))


# ── P17 source selection ────────────────────────────────────────────────────

def test_selects_tier_abc_and_explosive_only(tmp_path):
    rows = [_row("AAA", tier="B"), _row("BBB", tier="C"),
            _row("CCC", tier="W"),                       # excluded
            _row("DDD", tier="W", explosive=True)]       # included via explosive
    entries = _builder(tmp_path, rows).build()
    tickers = {e.ticker for e in entries}
    assert tickers == {"AAA", "BBB", "DDD"}


def test_baseline_context_mapped(tmp_path):
    rows = [_row("AAA", tier="B", float_shares=5_000_000, dilution=20.0,
                 catalyst="catalyst_tier1_news_2026-06-24", catalyst_score=90)]
    e = _builder(tmp_path, rows).build()[0]
    assert e.source == "p17" and e.tier == "B"
    assert e.float_shares == 5_000_000 and e.dilution_penalty == 20.0
    assert e.has_catalyst and e.catalyst_signals == ["catalyst_tier1_news_2026-06-24"]


# ── Hard filters ────────────────────────────────────────────────────────────

def test_filters_price_float_volume(tmp_path):
    rows = [
        _row("OK", price=3.0, float_shares=8_000_000, avg_vol=900_000),
        _row("PRICEY", price=9.0),                       # > $5
        _row("BIGFLOAT", float_shares=40_000_000),       # > 25M float
        _row("THIN", avg_vol=100_000),                   # < 500k avg vol
    ]
    tickers = {e.ticker for e in _builder(tmp_path, rows).build()}
    assert tickers == {"OK"}


# ── Ranking & cap ───────────────────────────────────────────────────────────

def test_ranking_prefers_explosive_then_tier(tmp_path):
    rows = [_row("C1", tier="C"), _row("B1", tier="B"),
            _row("EXP", tier="C", explosive=True)]
    ranked = _builder(tmp_path, rows).build()
    assert [e.ticker for e in ranked] == ["EXP", "B1", "C1"]


def test_cap_limits_count(tmp_path):
    rows = [_row(f"T{i}", tier="C") for i in range(10)]
    b = _builder(tmp_path, rows)
    b.cfg.feed_config.watchlist_cap = 3
    assert len(b.build()) == 3


# ── Output ──────────────────────────────────────────────────────────────────

def test_write_emits_watchlist_json(tmp_path):
    rows = [_row("AAA", tier="B")]
    summary = _builder(tmp_path, rows).run()
    assert summary["count"] == 1 and summary["sources"]["p17"] == 1
    payload = json.loads(Path(summary["path"]).read_text())
    assert payload["date"] == "2026-06-26"
    assert payload["entries"][0]["ticker"] == "AAA"
    assert payload["entries"][0]["tier"] == "B"


def test_baseline_enrichment_fills_non_p17(tmp_path):
    # manual pin (no baseline) gets enriched; P17 name keeps its CSV baseline
    rows = [_row("AAA", tier="B", avg_vol=2_000_000)]
    p17 = _p17_dir(tmp_path, "2026-06-26", rows)
    cfg = P19Config.create_default()
    cfg.use_gappers = False
    cfg.manual_pins = ["MAN"]
    fetched = {}

    def fake_fetcher(ticker):
        fetched[ticker] = True
        return {"avg_volume_30d": 750_000, "prior_close": 1.23}

    b = WatchlistBuilder(cfg, "2026-06-26", p17_results_dir=p17,
                         output_dir=str(tmp_path / "p19"), baseline_fetcher=fake_fetcher)
    by_ticker = {e.ticker: e for e in b.build()}
    assert by_ticker["MAN"].avg_volume_30d == 750_000      # enriched
    assert by_ticker["MAN"].prior_close == 1.23
    assert by_ticker["AAA"].avg_volume_30d == 2_000_000    # P17 baseline untouched
    assert "AAA" not in fetched                            # P17 not re-fetched


def test_no_p17_csv_yields_empty(tmp_path):
    cfg = P19Config.create_default()
    cfg.use_gappers = False
    b = WatchlistBuilder(cfg, "2026-06-26",
                         p17_results_dir=str(tmp_path / "empty"),
                         output_dir=str(tmp_path / "p19"))
    assert b.build() == []
