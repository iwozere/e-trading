#!/usr/bin/env python3
"""
Quick Start Script for Trading Web UI
------------------------------------

This script provides an easy way to start the trading web UI system.
It handles setup, dependency checks, and launches both backend and frontend.

Usage:
    python start_trading_webui.py [--dev] [--port PORT]

Examples:
    python start_trading_webui.py --dev     # Development mode
    python start_trading_webui.py           # Production mode
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
import subprocess
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from config.donotshare.donotshare import TELEGRAM_API_PORT, TELEGRAM_WEBGUI_PORT

_logger = setup_logger(__name__)


def print_banner():
    """Print startup banner."""
    print("🚀 Trading Web UI Quick Start")
    print("=" * 50)
    print()


def check_environment():
    """Check if environment is properly set up."""
    print("🔍 Checking environment...")

    # Check if .env file exists
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("📝 Creating .env file from template...")

        # Copy from example
        env_example = PROJECT_ROOT / ".env.example"
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print("✅ .env file created")
            print("💡 Please edit .env file with your API keys")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✅ .env file found")

    # Check if config directory exists
    config_dir = PROJECT_ROOT / "config/enhanced_trading"
    if not config_dir.exists():
        print("📁 Creating config directory...")
        config_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Config directory created")

    # Check if strategy config exists
    strategy_config = config_dir / "raspberry_pi_multi_strategy.json"
    if not strategy_config.exists():
        print("⚠️  Strategy configuration not found")
        print("💡 Run setup first: python setup_enhanced_trading.py")
        return False
    else:
        print("✅ Strategy configuration found")

    print()
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")

    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI and Uvicorn available")
    except ImportError:
        print("❌ FastAPI/Uvicorn not installed")
        print("💡 Install with: pip install fastapi uvicorn")
        return False

    try:
        import socketio
        print("✅ Socket.IO available")
    except ImportError:
        print("❌ Socket.IO not installed")
        print("💡 Install with: pip install python-socketio")
        return False

    print()
    return True


def start_backend(dev_mode=False, port=None):
    """Start the backend server."""
    # Use configured port or default
    if port is None:
        port = int(TELEGRAM_API_PORT) if TELEGRAM_API_PORT else 8000

    print(f"🚀 Starting backend server on port {port}...")

    try:
        if dev_mode:
            # Development mode with auto-reload
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.web_ui.backend.main:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--reload",
                "--reload-dir", "src"
            ]
        else:
            # Production mode
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.web_ui.backend.main:app",
                "--host", "0.0.0.0",
                "--port", str(port)
            ]

        # Set environment
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT)

        print(f"💻 Command: {' '.join(cmd)}")
        print(f"🌐 Backend will be available at: http://localhost:{port}")
        print(f"📚 API docs will be available at: http://localhost:{port}/docs")
        print()
        print("🔄 Starting server... (Press Ctrl+C to stop)")
        print("=" * 50)

        # Start the server
        subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return False

    return True


def start_frontend_dev(port=8000):
    """Start the frontend development server."""
    frontend_dir = PROJECT_ROOT / "src/web_ui/frontend"

    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False

    print("🎨 Starting frontend development server...")

    # Check if Node.js is available
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Node.js not found")
            print("💡 Install Node.js from: https://nodejs.org/")
            return False
        print(f"✅ Node.js available: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Node.js not found")
        print("💡 Install Node.js from: https://nodejs.org/")
        return False

    # Install dependencies if needed
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("📦 Installing frontend dependencies...")
        try:
            subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

    # Set environment variables
    env = os.environ.copy()
    env['VITE_API_BASE_URL'] = f'http://localhost:{port}'
    env['VITE_WS_URL'] = f'ws://localhost:{port}'

    try:
        print("🎨 Frontend will be available at: http://localhost:5173")
        print("🔄 Starting frontend... (This will open in a new terminal)")

        # Start frontend development server
        subprocess.run(['npm', 'run', 'dev'], cwd=frontend_dir, env=env)

    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return False

    return True


def show_usage_info(dev_mode=False, port=8000):
    """Show usage information."""
    print("🎯 Trading Web UI Started Successfully!")
    print("=" * 50)

    if dev_mode:
        print("🔧 Development Mode")
        print(f"📡 Backend API: http://localhost:{port}")
        print(f"📚 API Documentation: http://localhost:{port}/docs")
        print(f"🎨 Frontend: http://localhost:5173")
        print()
        print("💡 The frontend will auto-reload when you make changes")
        print("💡 The backend will auto-reload when you modify Python files")
    else:
        print("🚀 Production Mode")
        print(f"🌐 Web UI: http://localhost:{port}")
        print(f"📚 API Documentation: http://localhost:{port}/docs")

    print()
    print("🔐 Default Login Credentials:")
    print("   Username: admin")
    print("   Password: admin")
    print()
    print("📋 Quick Actions:")
    print("   • Create new strategies via the web interface")
    print("   • Monitor running strategies in real-time")
    print("   • Start/stop strategies with one click")
    print("   • View system metrics and performance")
    print()
    print("🛑 To stop the server, press Ctrl+C")
    print("=" * 50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Trading Web UI Quick Start')
    parser.add_argument('--dev', action='store_true',
                       help='Run in development mode with auto-reload')
    parser.add_argument('--port', type=int, default=None,
                       help=f'Port to run the server on (default: {TELEGRAM_API_PORT or 8000})')
    parser.add_argument('--setup', action='store_true',
                       help='Run setup first')

    args = parser.parse_args()

    print_banner()

    # Run setup if requested
    if args.setup:
        print("🔧 Running setup...")
        try:
            import setup_enhanced_trading
            setup_enhanced_trading.main()
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return
        print()

    # Check environment
    if not check_environment():
        print("❌ Environment check failed")
        print("💡 Try running: python start_trading_webui.py --setup")
        return

    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return

    # Show usage info
    show_usage_info(args.dev, args.port)

    if args.dev:
        print("🔧 Development mode selected")
        print("💡 You'll need to start frontend separately in another terminal:")
        print(f"   cd src/web_ui/frontend && npm run dev")
        print()

    # Start backend
    start_backend(args.dev, args.port)


if __name__ == "__main__":
    main()