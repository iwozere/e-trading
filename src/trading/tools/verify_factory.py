
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd()))

from src.config.configuration_factory import ConfigurationFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_factory")

def run_test():
    factory = ConfigurationFactory()

    # Create a dummy manifest in memory
    manifest = {
        "bot_id": "test_bot_verification",
        "description": "Test Bot for Factory Verification",
        "modules": {
            "broker": "instances/brokers/broker-binance-paper.json",
            "strategy": "instances/strategies/strategy-rsi-or-bb+advanced-atr-LTCUSDT-30m.json"
        },
        "overrides": {
            "symbol": "LTCUSDT",
            "initial_balance": 100000
        }
    }

    print("\n1. Loading Manifest...")
    try:
        config = factory.load_manifest(manifest)
        print("✅ Manifest Loaded and Resolved Successfully")

        # Verify resolution
        if "broker" in config and "type" in config["broker"]["broker"]:
            print(f"✅ Broker Resolved: {config['broker']['broker']['type']}")
        else:
            print("❌ Broker resolution failed")

        if "strategy" in config and "parameters" in config["strategy"]["strategy"]:
            print(f"✅ Strategy Resolved: {config['strategy']['strategy']['type']}")
        else:
            print("❌ Strategy resolution failed")

        print("\nResolved Config Structure:")
        print(json.dumps(config, indent=2)[:500] + "...")

    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
