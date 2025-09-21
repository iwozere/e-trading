#!/usr/bin/env python3
"""
Start Backend and Frontend Services
----------------------------------

This script starts the FastAPI backend server.
The frontend should be started separately using npm.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from config.donotshare.donotshare import TRADING_API_PORT

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting Trading Web UI Backend...")
    print(f"Backend API: http://localhost:{TRADING_API_PORT}")
    print("=" * 50)

    try:
        # Start backend server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.web_ui.backend.main:app",
            "--host", "0.0.0.0",
            "--port", str(TRADING_API_PORT),
            "--reload"
        ], check=True)

    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

if __name__ == "__main__":
    start_backend()