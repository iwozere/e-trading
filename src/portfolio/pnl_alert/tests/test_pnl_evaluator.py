"""Unit tests for `pnl_evaluator.evaluate`."""

import pytest

from src.portfolio.pnl_alert.pnl_evaluator import evaluate
from src.portfolio.pnl_alert.position_aggregator import Holding


def _h(symbol: str, avg: float, qty: float = 1.0, source: str = "ibkr") -> Holding:
    return Holding(symbol=symbol, avg_price=avg, quantity=qty, source=source)


def test_filters_below_threshold():
    """Rows with pnl_pct below the threshold are excluded."""
    holdings = [_h("AAA", 100.0), _h("BBB", 100.0)]
    prices = {"AAA": 109.0, "BBB": 115.0}

    rows = evaluate(holdings, prices, threshold_pct=0.10)

    assert [r.symbol for r in rows] == ["BBB"]


def test_exact_threshold_included():
    """An exactly-at-threshold row is included (>= semantics)."""
    holdings = [_h("AAA", 100.0)]
    prices = {"AAA": 110.0}

    rows = evaluate(holdings, prices, threshold_pct=0.10)

    assert len(rows) == 1
    assert rows[0].pnl_pct == pytest.approx(0.10)
    assert rows[0].pnl_abs == pytest.approx(10.0)


def test_sorted_by_pnl_pct_desc_with_tiebreaker():
    """Rows are sorted by pnl_pct desc, then pnl_abs desc, then symbol asc."""
    holdings = [
        _h("AAA", 100.0, qty=1),
        _h("BBB", 100.0, qty=10),
        _h("CCC", 100.0, qty=5),
    ]
    prices = {"AAA": 130.0, "BBB": 120.0, "CCC": 120.0}

    rows = evaluate(holdings, prices, threshold_pct=0.10)

    assert [r.symbol for r in rows] == ["AAA", "BBB", "CCC"]


def test_missing_price_excluded():
    """Symbols without a current price are dropped without failing the run."""
    holdings = [_h("AAA", 100.0), _h("BBB", 100.0)]
    prices = {"AAA": 150.0}

    rows = evaluate(holdings, prices, threshold_pct=0.10)

    assert [r.symbol for r in rows] == ["AAA"]


def test_non_positive_avg_price_is_skipped():
    """Holdings with avg_price <= 0 are defensively skipped."""
    holdings = [_h("AAA", 0.0)]
    prices = {"AAA": 100.0}

    rows = evaluate(holdings, prices, threshold_pct=0.10)

    assert rows == []


def test_negative_threshold_rejected():
    """The threshold must be strictly positive."""
    with pytest.raises(ValueError):
        evaluate([], {}, threshold_pct=0.0)


def test_quantity_affects_pnl_abs_not_pct():
    """pnl_pct is per-share; pnl_abs scales with quantity."""
    holdings = [_h("AAA", 100.0, qty=3)]
    prices = {"AAA": 120.0}

    rows = evaluate(holdings, prices, threshold_pct=0.10)

    assert rows[0].pnl_pct == pytest.approx(0.20)
    assert rows[0].pnl_abs == pytest.approx(60.0)
