#!/usr/bin/env python3
"""
Raspberry Pi Multi-Strategy Trading Service
------------------------------------------

This is the main service for running multiple trading strategies on Raspberry Pi.
It's designed to run as a system service (systemd) and provides:

- Multiple strategy instances in a single service
- Per-strategy broker configuration (paper/live)
- Health monitoring and auto-recovery
- Resource monitoring (CPU, memory, temperature)
- Graceful shutdown and restart
- Comprehensive logging and notifications

Usage:
    python raspberry_pi_trading_service.py [config_file]

Service Installation:
    sudo python raspberry_pi_trading_service.py --install-service
    sudo systemctl start trading-service
    sudo systemctl enable trading-service
"""

import asyncio
import argparse
import json
import signal
import sys
import os
import psutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.trading.strategy_manager import StrategyManager
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class RaspberryPiTradingService:
    """Main trading service for Raspberry Pi."""

    def __init__(self, config_file: Optional[str] = None,
                 use_db: bool = False,
                 db_user_id: Optional[int] = None,
                 db_poll_interval: int = 60):
        """Initialize the service."""
        self.config_file = config_file or "config/enhanced_trading/raspberry_pi_multi_strategy.json"
        self.config = None
        self.strategy_manager = StrategyManager()
        self.is_running = False
        self.start_time = None
        self.system_monitor_task = None
        # DB-backed mode options
        self.use_db = use_db
        self.db_user_id = db_user_id
        self.db_poll_interval = db_poll_interval

    async def load_config(self) -> bool:
        """Load service configuration."""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                _logger.error("Configuration file not found: %s", config_path)
                return False

            with open(config_path, 'r') as f:
                self.config = json.load(f)

            _logger.info("Loaded service configuration: %s", config_path)
            _logger.info("Service: %s v%s", self.config['system']['name'], self.config['system']['version'])

            return True

        except Exception:
            _logger.exception("Failed to load configuration:")
            return False

    async def start_service(self) -> bool:
        """Start the trading service."""
        try:
            _logger.info("🚀 Starting Raspberry Pi Multi-Strategy Trading Service")
            _logger.info("=" * 80)

            # Load configuration
            if not await self.load_config():
                return False

            # Load strategies either from DB or config file
            if self.use_db:
                _logger.info("Using database-backed strategy loading and polling (user_id=%s, interval=%ss)",
                             self.db_user_id, self.db_poll_interval)
                ok = await self.strategy_manager.load_strategies_from_db(self.db_user_id)
                if not ok:
                    _logger.warning("Initial DB load returned no strategies or failed; will continue and poll")
            else:
                if not await self.strategy_manager.load_strategies_from_config(self.config_file):
                    return False

            # Set service as running
            self.is_running = True
            self.start_time = datetime.now(timezone.utc)

            # Start system monitoring
            self.system_monitor_task = asyncio.create_task(self._monitor_system())

            # Start strategy monitoring
            await self.strategy_manager.start_monitoring()

            # Start strategies based on mode
            if self.use_db:
                await self.strategy_manager.start_db_polling(user_id=self.db_user_id,
                                                            interval_seconds=self.db_poll_interval)
                # In DB mode, strategies will be started by the poller; report current count
                started_count = sum(1 for s in self.strategy_manager.get_all_status() if s["status"] == "running")
            else:
                started_count = await self.strategy_manager.start_all_strategies()

            if started_count == 0:
                _logger.error("No strategies started successfully")
                return False

            _logger.info("🎯 Service running with %s strategy instances", started_count)
            _logger.info("Service is ready and monitoring strategies")
            _logger.info("=" * 80)

            return True

        except Exception:
            _logger.exception("Failed to start service:")
            return False

    async def stop_service(self):
        """Stop the trading service."""
        _logger.info("🛑 Stopping Raspberry Pi Multi-Strategy Trading Service")

        self.is_running = False

        # Stop system monitoring
        if self.system_monitor_task:
            self.system_monitor_task.cancel()
            try:
                await self.system_monitor_task
            except asyncio.CancelledError:
                pass

        # Stop strategy manager
        await self.strategy_manager.shutdown()

        _logger.info("✅ Service stopped successfully")

    async def restart_service(self):
        """Restart the trading service."""
        _logger.info("🔄 Restarting service...")
        await self.stop_service()
        await asyncio.sleep(5)  # Brief pause
        await self.start_service()

    def get_service_status(self) -> dict:
        """Get comprehensive service status."""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        # Get system info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Get temperature (Raspberry Pi specific)
        temperature = self._get_cpu_temperature()

        # Get strategy status
        strategy_status = self.strategy_manager.get_all_status()
        running_strategies = sum(1 for s in strategy_status if s['status'] == 'running')

        return {
            'service': {
                'name': self.config['system']['name'] if self.config else 'Unknown',
                'version': self.config['system']['version'] if self.config else 'Unknown',
                'status': 'running' if self.is_running else 'stopped',
                'uptime_seconds': uptime,
                'start_time': self.start_time.isoformat() if self.start_time else None
            },
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'temperature_c': temperature,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            'strategies': {
                'total': len(strategy_status),
                'running': running_strategies,
                'stopped': len(strategy_status) - running_strategies,
                'instances': strategy_status
            }
        }

    async def _monitor_system(self):
        """Monitor system resources and health."""
        while self.is_running:
            try:
                status = self.get_service_status()
                system = status['system']

                # Log system status periodically
                _logger.info("📊 System Status: CPU %.1f%% | Memory %.1f%% | Temp %.1f°C | Strategies %s/%s",
                           system['cpu_percent'],
                           system['memory_percent'],
                           system['temperature_c'],
                           status['strategies']['running'],
                           status['strategies']['total'])

                # Check for alerts
                alerts = self.config.get('global_settings', {}).get('monitoring', {}).get('alert_thresholds', {})

                if system['cpu_percent'] > alerts.get('cpu_usage', 80):
                    _logger.warning("⚠️  High CPU usage: %.1f%%", system["cpu_percent"])

                if system['memory_percent'] > alerts.get('memory_usage', 85):
                    _logger.warning("⚠️  High memory usage: %.1f%%", system['memory_percent'])

                if system['temperature_c'] > alerts.get('temperature', 70):
                    _logger.warning("⚠️  High temperature: %.1f%%", system['temperature_c'])

                await asyncio.sleep(300)  # Monitor every 5 minutes

            except Exception:
                _logger.exception("Error in system monitoring:")
                await asyncio.sleep(60)

    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (Raspberry Pi specific)."""
        try:
            # Try Raspberry Pi thermal zone
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000.0
                return temp
        except:
            try:
                # Try alternative method
                result = os.popen('vcgencmd measure_temp').readline()
                temp = float(result.replace("temp=", "").replace("'C\n", ""))
                return temp
            except:
                return 0.0

    async def run_forever(self):
        """Run the service indefinitely."""
        try:
            # Start the service
            if not await self.start_service():
                return False

            # Run until interrupted
            while self.is_running:
                await asyncio.sleep(1)

            return True

        except Exception:
            _logger.exception("Service error:")
            return False
        finally:
            await self.stop_service()


def setup_signal_handlers(service):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        _logger.info("Received signal %s, initiating shutdown...", signum)
        asyncio.create_task(service.stop_service())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def create_systemd_service():
    """Create systemd service file for the trading service."""
    service_content = f"""[Unit]
Description=Raspberry Pi Multi-Strategy Trading Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory={PROJECT_ROOT}
Environment=PYTHONPATH={PROJECT_ROOT}
ExecStart=/usr/bin/python3 {PROJECT_ROOT}/raspberry_pi_trading_service.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-service

# Resource limits
MemoryMax=1G
CPUQuota=80%

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={PROJECT_ROOT}

[Install]
WantedBy=multi-user.target
"""

    service_file = Path("/etc/systemd/system/trading-service.service")

    try:
        with open(service_file, 'w') as f:
            f.write(service_content)

        print(f"✅ Created systemd service file: {service_file}")
        print("\nTo enable and start the service:")
        print("sudo systemctl daemon-reload")
        print("sudo systemctl enable trading-service")
        print("sudo systemctl start trading-service")
        print("\nTo check status:")
        print("sudo systemctl status trading-service")
        print("sudo journalctl -u trading-service -f")

        return True

    except PermissionError:
        print("❌ Permission denied. Run with sudo to install service.")
        return False
    except Exception as e:
        print(f"❌ Failed to create service file: {e}")
        return False


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Raspberry Pi Multi-Strategy Trading Service')
    parser.add_argument('config', nargs='?',
                       default='config/enhanced_trading/raspberry_pi_multi_strategy.json',
                       help='Configuration file path')
    parser.add_argument('--install-service', action='store_true',
                       help='Install as systemd service')
    parser.add_argument('--status', action='store_true',
                       help='Show service status')
    # DB-backed mode
    parser.add_argument('--use-db', action='store_true',
                       help='Load bots from DB (trading_bots) and poll for changes')
    parser.add_argument('--db-user-id', type=int, default=None,
                       help='Optional user_id to filter bots in DB mode')
    parser.add_argument('--db-poll-interval', type=int, default=60,
                       help='DB polling interval in seconds (default: 60)')

    args = parser.parse_args()

    # Install service
    if args.install_service:
        create_systemd_service()
        return

    # Create service instance
    service = RaspberryPiTradingService(
        args.config,
        use_db=args.use_db,
        db_user_id=args.db_user_id,
        db_poll_interval=args.db_poll_interval,
    )
    setup_signal_handlers(service)

    # Show status
    if args.status:
        if service.is_running:
            status = service.get_service_status()
            print(json.dumps(status, indent=2))
        else:
            print("Service is not running")
        return

    # Run the service
    try:
        _logger.info("Starting Raspberry Pi Trading Service...")
        success = await service.run_forever()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt")
        await service.stop_service()
        sys.exit(0)
    except Exception:
        _logger.exception("Service failed:")
        await service.stop_service()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())