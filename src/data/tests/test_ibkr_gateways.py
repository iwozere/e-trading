from src.common.asyncio_compat import ensure_event_loop

ensure_event_loop()  # Py3.14: must run before import ib_async/ib_insync

try:  # prefer maintained ib_async
    from ib_async import IB  # type: ignore[import-not-found]
except ImportError:
    from ib_insync import IB

ib = IB()

# paper trading
ib.connect("raspberrypi", 4002, clientId=2)
print(ib.isConnected())  # should be True
print(ib.accountSummary())  # should print account info
ib.disconnect()

# live trading
ib.connect("raspberrypi", 4001, clientId=1)
print(ib.isConnected())  # should be True
print(ib.accountSummary())  # should print account info
ib.disconnect()
