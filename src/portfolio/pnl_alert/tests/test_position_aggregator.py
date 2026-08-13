"""Unit tests for `position_aggregator.merge_holdings`."""

from src.portfolio.pnl_alert.position_aggregator import (
    SOURCE_IBKR,
    RawIbkrPosition,
    merge_holdings,
)


def _pos(symbol: str, avg: float, qty: float = 10.0, sec_type: str = "STK") -> RawIbkrPosition:
    return RawIbkrPosition(symbol=symbol, avg_price=avg, quantity=qty, sec_type=sec_type)


def test_ibkr_positions_become_holdings():
    """Raw IBKR positions are converted 1:1 into holdings, source=ibkr."""
    holdings = merge_holdings([_pos("AAPL", 100.0, 5.0)])

    assert len(holdings) == 1
    assert holdings[0].symbol == "AAPL"
    assert holdings[0].source == SOURCE_IBKR
    assert holdings[0].avg_price == 100.0
    assert holdings[0].quantity == 5.0


def test_non_stk_ibkr_filtered_when_stk_only():
    """Non-STK IBKR positions are dropped when stk_only=True."""
    holdings = merge_holdings(
        [_pos("AAPL", 100.0, 5.0, sec_type="STK"), _pos("SPX", 0.5, 1.0, sec_type="OPT")],
        stk_only=True,
    )

    assert [h.symbol for h in holdings] == ["AAPL"]


def test_non_stk_kept_when_stk_only_false():
    """All sec-types survive when stk_only=False."""
    holdings = merge_holdings([_pos("SPX", 0.5, 1.0, sec_type="OPT")], stk_only=False)

    assert [h.symbol for h in holdings] == ["SPX"]


def test_zero_or_negative_ibkr_position_dropped():
    """IBKR rows with non-positive quantity or avg_price are dropped."""
    holdings = merge_holdings(
        [
            _pos("AAA", 100.0, qty=0.0),
            _pos("BBB", 0.0, qty=10.0),
            _pos("GOOD", 50.0, qty=2.0),
        ]
    )

    assert [h.symbol for h in holdings] == ["GOOD"]


def test_returns_sorted_by_symbol():
    holdings = merge_holdings([_pos("NVDA", 100.0), _pos("AAPL", 50.0)])

    assert [h.symbol for h in holdings] == ["AAPL", "NVDA"]
