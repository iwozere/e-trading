#!/usr/bin/env python3
"""
Web UI Runner Script
-------------------

This script runs both the FastAPI backend and serves the React frontend
for the trading web UI system. It can run in development or production mode.

Usage:
    python src/web_ui/run_web_ui.py [--dev] [--port PORT] [--host HOST]

Examples:
    python src/web_ui/run_web_ui.py --dev          # Development mode
    python src/web_ui/run_web_ui.py --port 8080   # Production mode on port 8080
"""

import asyncio
import argparse
import subprocess
import sys
import os
import signal
from pathlib import Path
from typing import Optional, List
import uvicorn
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class WebUIRunner:
    """Manages running the web UI backend and frontend."""

    def __init__(self, dev_mode: bool = False, host: str = "0.0.0.0", port: int = 8000):
        """Initialize the runner."""
        self.dev_mode = dev_mode
        self.host = host
        self.port = port
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.processes: List[subprocess.Popen] = []

    def setup_environment(self):
        """Setup environment variables and paths."""
        # Set environment variables
        os.environ['PYTHONPATH'] = str(PROJECT_ROOT)

        # Create necessary directories
        directories = [
            PROJECT_ROOT / "logs/web_ui",
            PROJECT_ROOT / "src/web_ui/frontend/dist",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        _logger.info("Environment setup complete")

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            # Check Python dependencies
            import fastapi
            import uvicorn
            _logger.info("✅ Python dependencies available")

            # Check if Node.js is available for development mode
            if self.dev_mode:
                result = subprocess.run(['node', '--version'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    _logger.info(f"✅ Node.js available: {result.stdout.strip()}")
                else:
                    _logger.error("❌ Node.js not found (required for development mode)")
                    return False

                # Check if npm is available
                result = subprocess.run(['npm', '--version'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    _logger.info(f"✅ npm available: {result.stdout.strip()}")
                else:
                    _logger.error("❌ npm not found (required for development mode)")
                    return False

            return True

        except ImportError as e:
            _logger.error(f"❌ Missing Python dependency: {e}")
            return False

    def install_frontend_dependencies(self) -> bool:
        """Install frontend dependencies if needed."""
        if not self.dev_mode:
            return True

        frontend_dir = PROJECT_ROOT / "src/web_ui/frontend"
        node_modules = frontend_dir / "node_modules"

        if not node_modules.exists():
            _logger.info("Installing frontend dependencies...")

            try:
                result = subprocess.run(
                    ['npm', 'install'],
                    cwd=frontend_dir,
                    check=True,
                    capture_output=True,
                    text=True
                )
                _logger.info("✅ Frontend dependencies installed")
                return True

            except subprocess.CalledProcessError as e:
                _logger.error(f"❌ Failed to install frontend dependencies: {e}")
                _logger.error(f"stdout: {e.stdout}")
                _logger.error(f"stderr: {e.stderr}")
                return False
        else:
            _logger.info("✅ Frontend dependencies already installed")
            return True

    def build_frontend(self) -> bool:
        """Build the frontend for production."""
        if self.dev_mode:
            return True

        frontend_dir = PROJECT_ROOT / "src/web_ui/frontend"
        dist_dir = frontend_dir / "dist"

        _logger.info("Building frontend for production...")

        try:
            # Install dependencies first
            if not self.install_frontend_dependencies():
                return False

            # Build the frontend
            result = subprocess.run(
                ['npm', 'run', 'build'],
                cwd=frontend_dir,
                check=True,
                capture_output=True,
                text=True
            )

            if dist_dir.exists():
                _logger.info("✅ Frontend built successfully")
                return True
            else:
                _logger.error("❌ Frontend build failed - dist directory not found")
                return False

        except subprocess.CalledProcessError as e:
            _logger.error(f"❌ Frontend build failed: {e}")
            _logger.error(f"stdout: {e.stdout}")
            _logger.error(f"stderr: {e.stderr}")
            return False

    def start_backend(self):
        """Start the FastAPI backend server."""
        _logger.info(f"Starting backend server on {self.host}:{self.port}")

        if self.dev_mode:
            # Development mode with auto-reload
            uvicorn.run(
                "src.api.main:app",
                host=self.host,
                port=self.port,
                reload=True,
                reload_dirs=[str(PROJECT_ROOT / "src")],
                log_level="info"
            )
        else:
            # Production mode
            uvicorn.run(
                "src.api.main:app",
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True
            )

    def start_frontend_dev(self):
        """Start the frontend development server."""
        if not self.dev_mode:
            return

        frontend_dir = PROJECT_ROOT / "src/web_ui/frontend"

        _logger.info("Starting frontend development server...")

        try:
            # Set environment variables for frontend
            env = os.environ.copy()
            env['VITE_API_BASE_URL'] = f'http://{self.host}:{self.port}'
            env['VITE_WS_URL'] = f'ws://{self.host}:{self.port}'

            self.frontend_process = subprocess.Popen(
                ['npm', 'run', 'dev'],
                cwd=frontend_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            self.processes.append(self.frontend_process)
            _logger.info("✅ Frontend development server started")

        except Exception as e:
            _logger.error(f"❌ Failed to start frontend development server: {e}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            _logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def shutdown(self):
        """Shutdown all processes."""
        _logger.info("Shutting down web UI...")

        # Stop heartbeat
        try:
            if hasattr(self, 'heartbeat_manager') and self.heartbeat_manager:
                self.heartbeat_manager.stop_heartbeat()
                _logger.info("Stopped web UI heartbeat")
        except Exception as e:
            _logger.error("Error stopping heartbeat: %s", e)

        for process in self.processes:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    _logger.error(f"Error terminating process: {e}")

        _logger.info("✅ Web UI shutdown complete")

    def run(self):
        """Run the web UI system."""
        try:
            _logger.info("🚀 Starting Trading Web UI System")
            _logger.info("=" * 60)

            # Setup
            self.setup_environment()
            self.setup_signal_handlers()

            # Check dependencies
            if not self.check_dependencies():
                _logger.error("❌ Dependency check failed")
                return False

            # Install frontend dependencies
            if not self.install_frontend_dependencies():
                _logger.error("❌ Frontend dependency installation failed")
                return False

            # Build frontend for production
            if not self.build_frontend():
                _logger.error("❌ Frontend build failed")
                return False

            if self.dev_mode:
                _logger.info("🔧 Running in DEVELOPMENT mode")
                _logger.info(f"Backend: http://{self.host}:{self.port}")
                _logger.info(f"Frontend: http://localhost:5173 (Vite dev server)")

                # Start frontend development server in background
                self.start_frontend_dev()

                # Give frontend time to start
                time.sleep(2)

            else:
                _logger.info("🚀 Running in PRODUCTION mode")
                _logger.info(f"Web UI: http://{self.host}:{self.port}")

            _logger.info("=" * 60)

            # Initialize heartbeat manager
            _logger.info("Initializing heartbeat manager...")
            try:
                from src.common.heartbeat_manager import HeartbeatManager

                def web_ui_health_check():
                    """Health check function for web UI."""
                    try:
                        # Check if processes are running
                        backend_running = self.backend_process is not None and self.backend_process.poll() is None
                        frontend_running = True  # In production, frontend is served by backend

                        if self.dev_mode:
                            frontend_running = self.frontend_process is not None and self.frontend_process.poll() is None

                        if backend_running and frontend_running:
                            return {
                                'status': 'HEALTHY',
                                'metadata': {
                                    'mode': 'development' if self.dev_mode else 'production',
                                    'host': self.host,
                                    'port': self.port,
                                    'backend_running': backend_running,
                                    'frontend_running': frontend_running
                                }
                            }
                        elif backend_running:
                            return {
                                'status': 'DEGRADED',
                                'error_message': 'Frontend not running' if self.dev_mode else 'Backend only',
                                'metadata': {
                                    'mode': 'development' if self.dev_mode else 'production',
                                    'backend_running': backend_running,
                                    'frontend_running': frontend_running
                                }
                            }
                        else:
                            return {
                                'status': 'DOWN',
                                'error_message': 'Backend not running',
                                'metadata': {
                                    'mode': 'development' if self.dev_mode else 'production',
                                    'backend_running': backend_running,
                                    'frontend_running': frontend_running
                                }
                            }
                    except Exception as e:
                        return {
                            'status': 'DOWN',
                            'error_message': f'Health check failed: {str(e)}'
                        }

                # Create and start heartbeat manager
                self.heartbeat_manager = HeartbeatManager(
                    system='web_ui',
                    interval_seconds=30
                )
                self.heartbeat_manager.set_health_check_function(web_ui_health_check)
                self.heartbeat_manager.start_heartbeat()

                _logger.info("Heartbeat manager started for web UI")

            except Exception as e:
                _logger.error("Failed to initialize heartbeat manager: %s", e)
                self.heartbeat_manager = None

            # Start backend (this blocks)
            self.start_backend()

        except KeyboardInterrupt:
            _logger.info("Received keyboard interrupt")
        except Exception as e:
            _logger.error(f"Error running web UI: {e}")
            return False
        finally:
            self.shutdown()

        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Trading Web UI Runner')
    parser.add_argument('--dev', action='store_true',
                       help='Run in development mode with auto-reload')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to (default: 8000)')

    args = parser.parse_args()

    # Create and run the web UI
    runner = WebUIRunner(
        dev_mode=args.dev,
        host=args.host,
        port=args.port
    )

    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()