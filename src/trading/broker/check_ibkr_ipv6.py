from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import threading
import time

class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print(f"Error {errorCode}: {errorString}")

    def nextValidId(self, orderId):
        print(f"✓✓✓ SUCCESS! Connected to IB Gateway on Raspberry Pi!")
        print(f"Next valid order ID: {orderId}")
        self.connected = True
        self.disconnect()

print("Connecting to 10.0.0.4:4002 with clientId=2...")
app = TestApp()
app.connect("10.0.0.4", 4002, clientId=2)

api_thread = threading.Thread(target=app.run, daemon=True)
api_thread.start()

time.sleep(10)

if app.connected:
    print("\n🎉 YOUR BRIDGE IS WORKING!")
    print("You can now connect from any machine on your network.")
else:
    print("\n⚠ Connection timeout - check Windows firewall")