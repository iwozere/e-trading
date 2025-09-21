#!/usr/bin/env python3
"""
Development Web UI Startup Script
--------------------------------

This script starts both the FastAPI backend and React frontend
for development with the correct port configuration.

Usage:
    python start_web_ui_dev.py
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from config.donotshare.donotshare import TRADING_API_PORT, TRADING_WEBGUI_PORT

def main():
    """Start both backend and frontend for development."""

    # Set environment variables for frontend
    env = os.environ.copy()
    env['VITE_API_BASE_URL'] = f'http://localhost:{TRADING_API_PORT}'
    env['VITE_WS_URL'] = f'ws://localhost:{TRADING_API_PORT}'

    processes = []

    try:
        print("🚀 Starting Trading Web UI Development Environment")
        print("=" * 60)
        print(f"Backend API: http://localhost:{TRADING_API_PORT}")
        print(f"Frontend UI: http://localhost:{TRADING_WEBGUI_PORT}")
        print("=" * 60)

        # Start backend API server
        print("Starting backend API server...")
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.web_ui.backend.main:app",
            "--host", "0.0.0.0",
            "--port", str(TRADING_API_PORT),
            "--reload"
        ], env=env)
        processes.append(backend_process)

        # Give backend time to start
        time.sleep(3)

        # Start frontend development server
        print("Starting frontend development server...")
        frontend_dir = PROJECT_ROOT / "src/web_ui/frontend"

        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)

        frontend_process = subprocess.Popen([
            "npm", "run", "dev", "--", "--port", str(TRADING_WEBGUI_PORT)
        ], cwd=frontend_dir, env=env)
        processes.append(frontend_process)

        print("\n✅ Both services started successfully!")
        print(f"🌐 Open your browser to: http://localhost:{TRADING_WEBGUI_PORT}")
        print("🔑 Login credentials: admin/admin or trader/trader")
        print("\nPress Ctrl+C to stop both services...")

        # Wait for processes
        while True:
            time.sleep(1)
            # Check if any process died
            for process in processes:
                if process.poll() is not None:
                    print(f"Process {process.pid} died, shutting down...")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")

    finally:
        # Cleanup processes
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    print(f"Error stopping process: {e}")

        print("✅ All services stopped")

if __name__ == "__main__":
    main()