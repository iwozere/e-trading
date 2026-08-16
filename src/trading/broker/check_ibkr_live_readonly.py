from typing import TYPE_CHECKING

from src.common.asyncio_compat import ensure_event_loop

if TYPE_CHECKING:
    # Type-check against ib_insync's types -- see ibkr_utils.py for why.
    from ib_insync import IB, MarketOrder, Stock
else:
    ensure_event_loop()  # Py3.14: must run before import ib_async/ib_insync
    try:  # prefer maintained ib_async
        from ib_async import IB, MarketOrder, Stock  # type: ignore[import-not-found]
    except ImportError:
        from ib_insync import IB, MarketOrder, Stock

ib = IB()
try:
    ib.connect("raspberrypi", 4001, clientId=1)
    print("✅ Connected to LIVE")

    # Try to place a dummy order
    contract = Stock("AAPL", "SMART", "USD")
    order = MarketOrder("BUY", 1)
    trade = ib.placeOrder(contract, order)

    ib.sleep(1)
    if trade.orderStatus.status == "Cancelled" or "Read-Only" in str(trade.log):
        print("🛡️ Read-Only Protection is ACTIVE. Order was blocked.")
    else:
        print("⚠️ Warning: Order went through or has a different status!")

finally:
    ib.disconnect()
