"""
Backtrader broker bridge — composition over conditional inheritance.

``BaseBroker`` subclasses ``abc.ABC`` only. This module provides a thin Cerebro-facing
``bt.broker.BrokerBase`` that delegates to a core broker implementing
:class:`~src.trading.broker.backtrader_bridge_protocol.SupportsBacktraderBridge`.
"""

from __future__ import annotations

from typing import Any

from src.trading.broker.backtrader_availability import BACKTRADER_AVAILABLE
from src.trading.broker.backtrader_bridge_protocol import SupportsBacktraderBridge

if BACKTRADER_AVAILABLE:
    import backtrader as bt

    class BacktraderBrokerBridge(bt.broker.BrokerBase):  # type: ignore[name-defined]
        """Adapts a core broker to Backtrader's ``BrokerBase`` API."""

        def __init__(self, core: SupportsBacktraderBridge) -> None:
            super().__init__()
            self._core = core
            core.enable_backtrader_trading_mode()
            # Writers / analyzers expect ``startingcash`` and ``cash`` (see ``BackBroker``).
            _cash = float(self._core._bt_getcash())
            self.startingcash = _cash
            self.cash = _cash

        def getcash(self) -> float:
            return self._core._bt_getcash()

        def getvalue(self, datas=None) -> float:
            return self._core._bt_getvalue(datas)

        def getposition(self, data) -> Any:
            return self._core._bt_getposition(data)

        def buy(self, owner=None, data=None, size=None, price=None, plimit=None,
                exectype=None, valid=None, tradeid=0, oco=None, trailamount=None,
                trailpercent=None, parent=None, transmit=True, **kwargs):
            return self._core.buy(
                owner=owner, data=data, size=size, price=price, plimit=plimit,
                exectype=exectype, valid=valid, tradeid=tradeid, oco=oco,
                trailamount=trailamount, trailpercent=trailpercent, parent=parent,
                transmit=transmit, **kwargs,
            )

        def sell(self, owner=None, data=None, size=None, price=None, plimit=None,
                 exectype=None, valid=None, tradeid=0, oco=None, trailamount=None,
                 trailpercent=None, parent=None, transmit=True, **kwargs):
            return self._core.sell(
                owner=owner, data=data, size=size, price=price, plimit=plimit,
                exectype=exectype, valid=valid, tradeid=tradeid, oco=oco,
                trailamount=trailamount, trailpercent=trailpercent, parent=parent,
                transmit=transmit, **kwargs,
            )

        def cancel(self, order) -> bool:
            return self._core.cancel(order)

        def next(self) -> None:
            self.cash = float(self._core._bt_getcash())
            return self._core.next()

        def get_notification(self):
            return self._core.get_notification()

else:
    BacktraderBrokerBridge = None  # type: ignore[misc,assignment]


def wrap_broker_for_cerebro(core: SupportsBacktraderBridge) -> Any:
    """
    Return a ``bt.broker.BrokerBase`` wrapper around ``core`` for ``cerebro.setbroker``.

    ``get_broker`` returns this kind of core; always wrap before attaching to Cerebro.

    Raises:
        ImportError: if backtrader is not installed.
    """
    if not BACKTRADER_AVAILABLE or BacktraderBrokerBridge is None:
        raise ImportError(
            "backtrader is required for wrap_broker_for_cerebro; "
            "install backtrader or run without Cerebro integration."
        )
    return BacktraderBrokerBridge(core)
