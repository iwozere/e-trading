import socket
import asyncio
from ib_insync import IB
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import config.donotshare.donotshare as donotshare

async def check_connection():
    # Try hostname first, then IP if provided
    host = donotshare.IBKR_HOST or 'raspberrypi'

    # Try common local IPs if hostname is 'raspberrypi'
    test_hosts = [host]
    if host == 'raspberrypi':
        # Add common LAN IPs if known, or just stick to host for now
        pass

    ports = [
        ('Paper (Gateway)', 4002),
        ('Live (Gateway)', 4001),
        ('Custom Socat (Bridge)', 4004),
        ('Jts Local (from jts.ini)', 4000)
    ]

    print(f"--- IBKR Diagnostics ---")

    for name, port in ports:
        print(f"\nChecking {name} on port {port}...")

        # 1. Basic socket check
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"  [SUCCESS] Port {port} is OPEN and reachable.")
        except ConnectionRefusedError:
            print(f"  [FAILED] Port {port} is CLOSED (Connection Refused).")
            continue
        except socket.timeout:
            print(f"  [FAILED] Port {port} timed out (Firewall issue?).")
            continue
        except Exception as e:
            print(f"  [FAILED] Error checking port {port}: {e}")
            continue

        # 2. IB handshake check
        print(f"  Attempting IBKR handshake on port {port}...")
        ib = IB()
        try:
            # Use a random higher clientId to avoid any conflicts
            # and a longer timeout for the first connection
            await asyncio.wait_for(ib.connectAsync(host, port, clientId=12345), timeout=10)
            print(f"  [SUCCESS] IBKR Handshake successful!")
            await ib.disconnect()
        except asyncio.TimeoutError:
            print(f"  [FAILED] IBKR Handshake timed out (Connected to Pi, but Gateway didn't respond).")
        except Exception as e:
            print(f"  [FAILED] IBKR Handshake failed: {e}")
        finally:
            if ib.isConnected():
                ib.disconnect()

    print("\n--- Recommendations ---")
    print("If all ports are CLOSED:")
    print("1. Ensure IB Gateway or TWS is actually running on the Raspberry Pi.")
    print("2. Check if the port matches your IB Gateway/TWS settings.")
    print("3. Ensure 'Enable ActiveX and Socket Clients' is checked in IBKR settings.")
    print("4. UNCHECK 'Allow connections from localhost only' in IBKR settings.")
    print("5. Add this machine's IP to 'Trusted IPs' in IBKR settings.")
    print(f"6. If '{host}' cannot be resolved, try using the actual IP address of the Pi in .env.")
    print("7. Port 4004 appears to be a bridge to 4002 based on container logs.")

if __name__ == "__main__":
    asyncio.run(check_connection())
