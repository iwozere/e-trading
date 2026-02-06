from ib_insync import IB, util, Stock
import logging

# Setup basic logging to see connection details
logging.basicConfig(level=logging.INFO)

def test_connection(name, port):
    ib = IB()
    print(f"\n--- Testing {name} Instance (Port {port}) ---")

    try:
        # Connect to the Pi's localhost (or use the Pi's IP if running remotely)
        ib.connect('raspberrypi', port, clientId=1, timeout=15)
        ib.reqMarketDataType(3) # delayed 15-20 minutes

        # Check Account & Trading Mode
        account = ib.managedAccounts()[0]
        print(f"✅ Connected to Account: {account}")

        # Verify Read-Only status (Checking API Settings)
        # Note: ib_insync doesn't have a direct 'is_read_only' property,
        # but we can check if it's Live or Paper mode.
        print(f"📊 Market Data Type: {'Live' if port == 4001 else 'Paper'}")

        # Fetch Positions
        positions = ib.positions()
        print(f"📦 Positions found: {len(positions)}")
        for pos in positions:
            print(f"   - {pos.contract.symbol}: {pos.position} shares @ {pos.avgCost}")

        # Basic Market Data Test (Apple Inc.)
        contract = Stock('MSFT', 'SMART', 'USD')
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract)
        ib.sleep(2) # Give it a moment to fetch
        print(f"🍏 AAPL Last Price: {ticker.last}")

    except Exception as e:
        print(f"❌ Failed to connect to {name}: {e}")
    finally:
        ib.disconnect()
        print(f"--- {name} Test Complete ---")

if __name__ == "__main__":
    # Test Live Instance
    test_connection("LIVE (Read-Only)", 4001)

    # Test Paper Instance
    test_connection("PAPER (Trading)", 4002)