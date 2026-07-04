"""Unit tests for `position_aggregator.merge_holdings`."""

from src.portfolio.pnl_alert.position_aggregator import (
    SOURCE_IBKR,
    SOURCE_WATCHLIST,
    RawIbkrPosition,
    merge_holdings,
)
from src.portfolio.pnl_alert.watchlist_loader import WatchlistEntry


def _pos(symbol: str, avg: float, qty: float = 10.0, sec_type: str = "STK") -> RawIbkrPosition:
    return RawIbkrPosition(symbol=symbol, avg_price=avg, quantity=qty, sec_type=sec_type)


def test_ibkr_wins_on_symbol_conflict():
    """If a symbol exists in both sources, the IBKR entry is kept."""
    holdings, conflicts = merge_holdings(
        [_pos("AAPL", 100.0, 5.0)],
        [WatchlistEntry(symbol="AAPL", avg_price=200.0)],
    )

    assert len(holdings) == 1
    assert holdings[0].source == SOURCE_IBKR
    assert holdings[0].avg_price == 100.0
    assert holdings[0].quantity == 5.0
    assert conflicts == ["AAPL"]


def test_non_stk_ibkr_filtered_when_stk_only():
    """Non-STK IBKR positions are dropped when stk_only=True."""
    holdings, _ = merge_holdings(
        [_pos("AAPL", 100.0, 5.0, sec_type="STK"), _pos("SPX", 0.5, 1.0, sec_type="OPT")],
        [],
        stk_only=True,
    )

    assert [h.symbol for h in holdings] == ["AAPL"]


def test_non_stk_kept_when_stk_only_false():
    """All sec-types survive when stk_only=False."""
    holdings, _ = merge_holdings(
        [_pos("SPX", 0.5, 1.0, sec_type="OPT")],
        [],
        stk_only=False,
    )

    assert [h.symbol for h in holdings] == ["SPX"]


def test_watchlist_adds_symbols_not_in_ibkr():
    """Watchlist entries that don't collide are added with source=watchlist."""
    holdings, conflicts = merge_holdings(
        [_pos("AAPL", 100.0)],
        [WatchlistEntry(symbol="NVDA", avg_price=500.0)],
    )

    symbols = sorted(h.symbol for h in holdings)
    assert symbols == ["AAPL", "NVDA"]

    nvda = next(h for h in holdings if h.symbol == "NVDA")
    assert nvda.source == SOURCE_WATCHLIST
    assert nvda.quantity == 1.0
    assert conflicts == []


def test_zero_or_negative_ibkr_position_dropped():
    """IBKR rows with non-positive quantity or avg_price are dropped."""
    holdings, _ = merge_holdings(
        [
            _pos("AAA", 100.0, qty=0.0),
            _pos("BBB", 0.0, qty=10.0),
            _pos("GOOD", 50.0, qty=2.0),
        ],
        [],
    )

    assert [h.symbol for h in holdings] == ["GOOD"]


def test_symbol_case_is_normalized_for_watchlist_conflict():
    """Conflict detection is case-insensitive."""
    holdings, conflicts = merge_holdings(
        [_pos("AAPL", 100.0)],
        [WatchlistEntry(symbol="aapl", avg_price=200.0)],
    )

    assert [h.symbol for h in holdings] == ["AAPL"]
    assert conflicts == ["AAPL"]
