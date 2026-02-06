from ib_insync import *

ib = IB()
try:
    ib.connect('raspberrypi', 4001, clientId=1)
    print("✅ Connected to LIVE")

    # Try to place a dummy order
    contract = Stock('AAPL', 'SMART', 'USD')
    order = MarketOrder('BUY', 1)
    trade = ib.placeOrder(contract, order)

    ib.sleep(1)
    if trade.orderStatus.status == 'Cancelled' or "Read-Only" in str(trade.log):
        print("🛡️ Read-Only Protection is ACTIVE. Order was blocked.")
    else:
        print("⚠️ Warning: Order went through or has a different status!")

finally:
    ib.disconnect()