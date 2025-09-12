from ib_insync import IB

ib = IB()

# paper trading
ib.connect('raspberrypi', 4002, clientId=2)
print(ib.isConnected())        # should be True
print(ib.accountSummary())     # should print account info
ib.disconnect()

# live trading
ib.connect('raspberrypi', 4001, clientId=1)
print(ib.isConnected())        # should be True
print(ib.accountSummary())     # should print account info
ib.disconnect()