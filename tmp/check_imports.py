
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Simulate uvicorn putting src at the front
sys.path.insert(0, str(PROJECT_ROOT / "src"))
print(f"Path[0] (Simulated bad): {sys.path[0]}")

# NOW APPLY THE FIX from main.py
SRC_ROOT = str(PROJECT_ROOT / "src")
if sys.path[0] != str(PROJECT_ROOT):
    if str(PROJECT_ROOT) in sys.path:
        sys.path.remove(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

# Remove src root from path
if SRC_ROOT in sys.path:
    sys.path.remove(SRC_ROOT)

print(f"Path[0] (Fixed): {sys.path[0]}")

# Test imports
import importlib

# Crucial: remove config from sys.modules to force re-evaluation
if 'config' in sys.modules:
    del sys.modules['config']
if 'config.donotshare' in sys.modules:
    del sys.modules['config.donotshare']

try:
    import config.donotshare
    print("SUCCESS: config.donotshare imported from ROOT")
    # print(f"Location: {config.donotshare.__file__}")
except ImportError as e:
    print(f"FAILURE on donotshare: {e}")

try:
    from src.config.config_manager import ConfigManager
    print("SUCCESS: src.config.config_manager imported")
except ImportError as e:
    print(f"FAILURE on src.config: {e}")
