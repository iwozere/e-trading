"""Unit tests for `IBKROpenOrdersFeed` / `fetch_protective_qty`."""

from types import SimpleNamespace

from src.portfolio.management.open_orders import IBKROpenOrdersFeed, fetch_protective_qty


def _trade(symbol: str, action: str, order_type: str, total_qty: float, remaining: float | None = None):
    """Build a minimal object shaped like ib_insync's `Trade`."""
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol),
        order=SimpleNamespace(action=action, orderType=order_type, totalQuantity=total_qty),
        orderStatus=SimpleNamespace(remaining=remaining if remaining is not None else total_qty),
    )


class FakeIB:
    def __init__(self, trades, connect_ok: bool = True):
        self._trades = trades
        self._connect_ok = connect_ok
        self.disconnected = False

    def connect(self, host, port, clientId, timeout, readonly):
        del host, port, clientId, timeout
        assert readonly is True, "open-orders feed must always connect read-only"
        if not self._connect_ok:
            raise ConnectionRefusedError("no gateway")

    def reqAllOpenOrders(self):
        return self._trades

    def disconnect(self):
        self.disconnected = True


def test_protective_order_qty_sums_matching_sell_stops():
    feed = IBKROpenOrdersFeed(host="h", port=4001, client_id=21)
    feed._ib = FakeIB(  # type: ignore[assignment]  # duck-typed test double, not a real ib_insync.IB
        [
            _trade("AAA", "SELL", "STP", 100),
            _trade("AAA", "SELL", "TRAIL", 50, remaining=30),  # partially filled -> use remaining
            _trade("BBB", "SELL", "STP", 10),
        ]
    )

    qty = feed.protective_order_qty(["AAA"])

    assert qty == {"AAA": 130.0}  # 100 + 30, BBB excluded (not requested)


def test_protective_order_qty_ignores_buy_side_and_non_protective_types():
    feed = IBKROpenOrdersFeed(host="h", port=4001, client_id=21)
    feed._ib = FakeIB(  # type: ignore[assignment]  # duck-typed test double, not a real ib_insync.IB
        [
            _trade("AAA", "BUY", "STP", 100),  # wrong side, e.g. protecting a short
            _trade("AAA", "SELL", "LMT", 100),  # not a protective type
        ]
    )

    assert feed.protective_order_qty(["AAA"]) == {}


def test_protective_order_qty_no_connection_returns_empty():
    feed = IBKROpenOrdersFeed(host="h", port=4001, client_id=21)
    assert feed.protective_order_qty(["AAA"]) == {}


def test_protective_order_qty_skips_malformed_trade():
    feed = IBKROpenOrdersFeed(host="h", port=4001, client_id=21)
    feed._ib = FakeIB(  # type: ignore[assignment]  # duck-typed test double, not a real ib_insync.IB
        [SimpleNamespace(contract=None, order=None, orderStatus=None), _trade("AAA", "SELL", "STP", 10)]
    )

    assert feed.protective_order_qty(["AAA"]) == {"AAA": 10.0}


def test_fetch_protective_qty_disconnects_even_on_failure():
    fake_ib = FakeIB([_trade("AAA", "SELL", "STP", 10)])

    class BrokenFeed(IBKROpenOrdersFeed):
        def connect(self, attempts=2, backoff_seconds=3.0):
            del attempts, backoff_seconds
            self._ib = fake_ib  # type: ignore[assignment]  # duck-typed test double, not a real ib_insync.IB
            return True

        def protective_order_qty(self, symbols):
            del symbols
            raise RuntimeError("boom")

    feed = BrokenFeed(host="h", port=4001, client_id=21)
    try:
        fetch_protective_qty(feed, ["AAA"])
        assert False, "expected RuntimeError to propagate"
    except RuntimeError:
        pass
    assert fake_ib.disconnected is True


def test_fetch_protective_qty_returns_false_when_connect_fails():
    class UnreachableFeed(IBKROpenOrdersFeed):
        def connect(self, attempts=2, backoff_seconds=3.0):
            del attempts, backoff_seconds
            return False

    feed = UnreachableFeed(host="h", port=4001, client_id=21)
    connected, qty = fetch_protective_qty(feed, ["AAA"])
    assert connected is False
    assert qty == {}
