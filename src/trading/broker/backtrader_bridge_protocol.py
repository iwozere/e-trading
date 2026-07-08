"""Static typing for objects that ``BacktraderBrokerBridge`` can wrap."""

from __future__ import annotations

from typing import Optional, Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsBacktraderBridge(Protocol):
    """
    Minimum surface the bridge delegates to. Implemented by ``BaseBroker``
    and all concrete brokers that do not override ``buy``/``sell`` with
    incompatible signatures.
    """

    def enable_backtrader_trading_mode(self) -> None: ...

    def _bt_getcash(self) -> float: ...

    def _bt_getvalue(self, datas: Optional[Any] = None) -> float: ...

    def _bt_getposition(self, data: Any) -> Any: ...

    def buy(self, *args: Any, **kwargs: Any) -> Any: ...

    def sell(self, *args: Any, **kwargs: Any) -> Any: ...

    def cancel(self, order: Any) -> bool: ...

    def next(self) -> None: ...

    def get_notification(self) -> Any: ...
