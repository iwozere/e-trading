#!/usr/bin/env python3
"""
Enhanced Strategy Manager
------------------------

This module manages multiple strategy instances within a single service.
Each strategy instance can run with different broker configurations (paper/live).

Features:
- Multiple strategy instances in one service
- Per-strategy broker configuration (paper/live)
- Integration with existing BaseTradingBot and CustomStrategy
- Health monitoring and auto-recovery
- Unified logging and notifications
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from src.data.db.services.trading_service import trading_service
from src.data.db.services.users_service import UsersService
from src.notification.logger import setup_logger
from src.notification.service.client import NotificationServiceClient
from src.trading.broker.broker_manager import BrokerManager
from src.trading.constants import PROJECT_ROOT
from src.trading.instance_service import InstanceService
from src.trading.strategy_handler import strategy_handler

_logger = setup_logger(__name__)
_users_service = UsersService()


def _coerce_db_bot_id(raw_id: Any) -> int | None:
    """Return BotInstance PK for ``trading_service`` calls, or None if not numeric (e.g. ephemeral UUID)."""
    if raw_id is None:
        return None
    s = str(raw_id).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _update_bot_status_db(
    raw_id: Any,
    status: str,
    error_message: str | None = None,
    started_at: datetime | None = None,
) -> None:
    """Persist bot status when ``raw_id`` maps to a database row; no-op for manifest-only UUID instances."""
    bid = _coerce_db_bot_id(raw_id)
    if bid is None:
        _logger.debug("Skipping update_bot_status (no DB id): %s", raw_id)
        return
    try:
        trading_service.update_bot_status(bid, status, error_message=error_message, started_at=started_at)
    except Exception:
        _logger.debug("update_bot_status failed for bot_id=%s", bid, exc_info=True)


class StrategyManager:
    """
    Bot manager and SOLE CONFIGURATION LOADER.

    This is the ONLY component that loads configurations from the database.
    All other components receive configs through this manager.

    Responsibilities:
    - Load ALL bot configurations from database (ONLY place this happens)
    - Create and manage StrategyInstance objects
    - Handle bot lifecycle (start/stop/restart)
    - Monitor bot health and performance
    - Update database with bot status and metrics
    - DB polling for hot-reload
    """

    def __init__(
        self,
        user_id: int | None = None,
        notification_client: NotificationServiceClient | None = None,
        trade_repository: Any = None,
    ):
        """Initialize the strategy manager."""
        self.user_id = user_id
        self.trade_repository = trade_repository

        # Use database-only notification client by default if none provided
        self.notification_client = notification_client or NotificationServiceClient(service_url="database://")
        if not notification_client:
            _logger.info("Initialized default notification client in database-only mode")

        # Lifecycle management delegated to InstanceService
        self.instance_service = InstanceService(
            notification_client=self.notification_client, trade_repository=trade_repository
        )

        self.broker_manager = BrokerManager()

        self.is_running = False
        self.monitoring_task = None
        self.db_poll_task = None
        self._db_poll_running = False
        self._db_poll_user_id = user_id
        self._db_poll_interval: int = 60

        # Crash recovery marker
        self._marker_path = PROJECT_ROOT / ".trading_service_running"

    def add_instance(self, config: Dict[str, Any]) -> str:
        """Register one hydrated strategy manifest (CLI / LiveTradingBot / Web UI)."""
        instance_id = str(config.get("id") or config.get("bot_id") or uuid.uuid4())
        if instance_id in self.strategy_instances:
            self.strategy_instances[instance_id].config = config
        else:
            self.instance_service.create_instance(instance_id, config)
        return instance_id

    async def start_instance(self, instance_id: str) -> bool:
        return await self.start_strategy(instance_id)

    async def stop_instance(self, instance_id: str) -> bool:
        return await self.stop_strategy(instance_id)

    async def restart_instance(self, instance_id: str) -> bool:
        return await self.restart_strategy(instance_id)

    def get_instance_status(self, instance_id: str) -> Dict[str, Any] | None:
        return self.get_strategy_status(instance_id)

    @property
    def strategy_instances(self):
        """Maintain backward compatibility for components accessing instances directly."""
        return self.instance_service.instances

    async def load_strategies_from_config(self, config_file: str) -> bool:
        """Load strategy configurations from JSON file."""
        try:
            from src.config.configuration_factory import config_factory

            config_path = Path(config_file)
            if not config_path.exists():
                _logger.error("Configuration file not found: %s", config_path)
                return False

            with open(config_path) as f:
                config = json.load(f)

            strategies = config.get("strategies", [])
            if not strategies:
                _logger.error("No strategies found in configuration")
                return False

            _logger.info("Loading %d strategy configurations", len(strategies))

            for strategy_config in strategies:
                # Use Factory to Hydrate/Validate
                try:
                    if "bot_id" not in strategy_config:
                        strategy_config["bot_id"] = strategy_config.get("id") or str(uuid.uuid4())
                    strategy_config = config_factory.load_manifest(strategy_config)
                except Exception as e:
                    _logger.error("Strategy factory hydration failed: %s", e)
                    continue

                instance_id = strategy_config.get("id") or str(uuid.uuid4())

                # Delegate creation to InstanceService
                self.instance_service.create_instance(instance_id, strategy_config)

            _logger.info("Successfully loaded %d strategies", len(self.strategy_instances))
            return True

        except Exception:
            _logger.exception("Failed to load strategies from config:")
            return False

    async def start_all_strategies(self) -> int:
        """Start all configured strategy instances."""
        _logger.info("Starting all strategy instances...")

        started_count = 0
        for instance_id in self.strategy_instances.keys():
            # Check if strategy is enabled
            instance = self.strategy_instances[instance_id]
            if not instance.config.get("enabled", True):
                _logger.info("Strategy %s is disabled, skipping", instance.name)
                continue

            success = await self.instance_service.start_instance(instance_id)
            if success:
                started_count += 1

            # Small delay between starts
            await asyncio.sleep(1)

        _logger.info("Started %d/%d strategy instances", started_count, len(self.strategy_instances))
        return started_count

    async def stop_all_strategies(self):
        """Stop all running strategy instances."""
        await self.instance_service.stop_all_instances()

    async def start_strategy(self, instance_id: str) -> bool:
        """Start a specific strategy instance."""
        return await self.instance_service.start_instance(instance_id)

    async def stop_strategy(self, instance_id: str) -> bool:
        """Stop a specific strategy instance."""
        return await self.instance_service.stop_instance(instance_id)

    async def restart_strategy(self, instance_id: str) -> bool:
        """Restart a specific strategy instance."""
        if instance_id not in self.strategy_instances:
            _logger.error("Strategy instance %s not found", instance_id)
            return False

        return await self.strategy_instances[instance_id].restart()

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get status of all strategy instances."""
        return [instance.get_status() for instance in self.strategy_instances.values()]

    def get_strategy_status(self, instance_id: str) -> Dict[str, Any] | None:
        """Get status of a specific strategy instance."""
        if instance_id not in self.strategy_instances:
            return None

        return self.strategy_instances[instance_id].get_status()

    async def start_monitoring(self):
        """Start monitoring all strategy instances."""
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitor_strategies())

    async def stop_monitoring(self):
        """Stop monitoring."""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop DB polling if running
        self._db_poll_running = False
        if self.db_poll_task:
            self.db_poll_task.cancel()
            try:
                await self.db_poll_task
            except asyncio.CancelledError:
                pass

    async def _monitor_strategies(self):
        """Monitor strategy instances and handle auto-recovery."""
        while self.is_running:
            try:
                current_time = datetime.now(UTC)
                running_count = 0
                unhealthy_count = 0

                # Check health of all running strategies
                for instance_id, instance in self.strategy_instances.items():
                    if instance.status == "running":
                        running_count += 1

                        # Check heartbeat health
                        if instance.last_heartbeat:
                            heartbeat_age = (current_time - instance.last_heartbeat).total_seconds()
                            max_heartbeat_age = instance.heartbeat_interval * 3  # 3x normal interval

                            if heartbeat_age > max_heartbeat_age:
                                unhealthy_count += 1
                                _logger.warning(
                                    "Bot %s heartbeat stale (%.1fs old, max %.1fs). Attempting recovery...",
                                    instance.name,
                                    heartbeat_age,
                                    max_heartbeat_age,
                                )
                                # Try to restart unhealthy bot
                                if instance.error_count < 3:
                                    await instance.restart()
                                else:
                                    _logger.error(
                                        "Bot %s exceeded max restart attempts (%d), marking as error",
                                        instance.name,
                                        instance.error_count,
                                    )
                                    instance.status = "error"
                                    _update_bot_status_db(
                                        instance_id,
                                        "error",
                                        error_message=f"Exceeded max restart attempts ({instance.error_count})",
                                    )

                    elif instance.status == "error" and instance.error_count < 3:
                        # Auto-recovery for failed strategies (max 3 attempts)
                        _logger.warning(
                            "Attempting auto-recovery for %s (attempt %d/3)", instance.name, instance.error_count + 1
                        )
                        await instance.restart()

                # Log periodic status with detailed metrics
                _logger.info(
                    "Strategy Monitor: %d/%d running, %d unhealthy, %d total",
                    running_count,
                    len(self.strategy_instances),
                    unhealthy_count,
                    len(self.strategy_instances),
                )

                await asyncio.sleep(60)  # Monitor every minute

            except Exception:
                _logger.exception("Error in strategy monitoring:")
                await asyncio.sleep(10)

    # -------------------- Crash Detection and Recovery --------------------
    def _detect_crash_recovery(self) -> bool:
        """
        Detect if this is a crash recovery (unclean shutdown).

        Primary signal: any bots in ``status='running'`` in the database when
        we start up indicates they were not stopped cleanly.  This is reliable
        across containerised deployments where filesystem markers are ephemeral.

        Secondary signal (fallback): the ``_marker_path`` file, kept for
        environments where the DB is unavailable during early startup or where
        single-process restart patterns rely on the file.

        Returns:
            True if previous shutdown was unclean (crash detected).
        """
        # DB-first: query for bots that were left in 'running' state.
        try:
            running_bots = trading_service.get_bots_by_status("running")
            if running_bots:
                _logger.warning(
                    "UNCLEAN SHUTDOWN detected: %d bot(s) left in 'running' state in DB",
                    len(running_bots),
                )
                return True
        except Exception as exc:
            _logger.debug("DB crash-detection query failed, falling back to marker file: %s", exc)

        # File fallback: legacy marker for non-containerised deployments.
        if self._marker_path.exists():
            _logger.warning("UNCLEAN SHUTDOWN detected: stale session marker found at %s", self._marker_path)
            _logger.warning("This indicates the service crashed or was forcefully terminated")
            self._marker_path.unlink()
            return True

        _logger.info("Clean startup detected — no crash signals found")
        return False

    def _mark_service_running(self) -> None:
        """Record that the service is running (creates marker file as fallback)."""
        try:
            self._marker_path.touch()
            _logger.debug("Created session marker file: %s", self._marker_path)
        except Exception as exc:
            _logger.warning("Failed to create session marker: %s", exc)

    def _mark_clean_shutdown(self) -> None:
        """Record a clean shutdown by removing the marker file."""
        try:
            if self._marker_path.exists():
                self._marker_path.unlink()
                _logger.info("Removed session marker — clean shutdown")
            else:
                _logger.debug("Session marker already removed")
        except Exception as exc:
            _logger.warning("Failed to remove session marker: %s", exc)

    def _recover_bot_state(self, bot_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recover bot state from database on restart.

        Queries open positions and pending trades to reconstruct state context
        that can be used by the strategy on restart.

        Args:
            bot_id: Bot ID to recover state for
            config: Bot configuration dictionary

        Returns:
            Enhanced config with recovered state in '_recovered_positions' and '_recovered_trades'
        """
        try:
            # Get open positions
            open_positions = trading_service.get_open_positions(bot_id=str(bot_id))

            if open_positions:
                _logger.info("Recovered %d open position(s) for bot %d", len(open_positions), bot_id)
                # Store positions in config for strategy to access
                config["_recovered_positions"] = open_positions

                # Log position details
                for pos in open_positions:
                    _logger.info(
                        "  Position: %s %s qty=%.8f avg_price=%.8f status=%s",
                        pos.get("symbol"),
                        pos.get("direction"),
                        pos.get("qty_open", 0),
                        pos.get("avg_price", 0),
                        pos.get("status"),
                    )

            # Get open trades
            open_trades = trading_service.get_open_trades()
            bot_trades = [t for t in open_trades if t["bot_id"] == bot_id]

            if bot_trades:
                _logger.info("Recovered %d open trade(s) for bot %d", len(bot_trades), bot_id)
                config["_recovered_trades"] = bot_trades

                # Log trade details
                for trade in bot_trades:
                    _logger.info(
                        "  Trade: %s entry=%.8f @ %s status=%s",
                        trade.get("symbol"),
                        trade.get("entry_price", 0),
                        trade.get("entry_time"),
                        trade.get("status"),
                    )

            return config

        except Exception:
            _logger.exception("Error recovering state for bot %d:", bot_id)
            return config

    # -------------------- DB-backed loading and polling --------------------
    async def load_strategies_from_db(self, user_id: int | None = None, resume_mode: bool = True) -> bool:
        """
        Load strategy configurations from the database - ONLY PLACE THIS HAPPENS.

        This is the SOLE configuration loader. No other component should load from DB.

        Behavior:
        - In resume_mode: Detects crash recovery and resumes only previously running bots
        - In normal mode: Loads all enabled bots
        - Validates each bot config using trading_service AND StrategyHandler
        - Recovers bot state (positions, trades) for crash recovery
        - Builds or updates StrategyInstance entries keyed by bot id
        - Invalid configs are skipped and marked as 'error' in DB

        Args:
            user_id: Optional user ID to filter bots
            resume_mode: If True, use smart resume logic (default: True)

        Returns:
            True if successfully loaded at least one bot, False otherwise
        """
        try:
            _logger.info("=" * 80)
            _logger.info("LOADING BOT CONFIGURATIONS FROM DATABASE (SOLE CONFIG LOADER)")
            _logger.info("=" * 80)

            # Check for crash recovery
            was_crashed = False
            if resume_mode:
                was_crashed = self._detect_crash_recovery()

                if was_crashed:
                    _logger.warning("CRASH RECOVERY MODE: Resuming previously running bots")
                    # Load only bots that were running before crash
                    bots = trading_service.get_bots_by_status("running", user_id)
                    _logger.info("Found %d bot(s) that were running before crash", len(bots))
                else:
                    _logger.info("NORMAL STARTUP: Loading all enabled bots")
                    bots = trading_service.get_enabled_bots(user_id)
                    _logger.info("Found %d enabled bot(s) in database", len(bots))
            else:
                _logger.info("NORMAL STARTUP (resume_mode=False): Loading all enabled bots")
                bots = trading_service.get_enabled_bots(user_id)
                _logger.info("Found %d enabled bot(s) in database", len(bots))

            # Create marker file to track this session
            self._mark_service_running()

            if not bots:
                _logger.warning("No bots to load%s", f" for user_id={user_id}" if user_id else "")
                return False

            loaded = 0
            for bot in bots:
                instance_id = str(bot["id"])
                bot_name = bot.get("description") or f"Bot {bot['id']}"

                _logger.info("Processing bot: %s (ID: %s)", bot_name, instance_id)

                # Validate database record/config
                is_valid, errors, warnings = trading_service.validate_bot_configuration(bot["id"])
                if not is_valid:
                    _logger.error("Bot %s config invalid, skipping. Errors: %s", bot["id"], errors)
                    _update_bot_status_db(bot["id"], "error", error_message="; ".join(errors))
                    continue

                # Log warnings
                for warning in warnings:
                    _logger.warning("Bot %s: %s", bot["id"], warning)

                # Map db record to StrategyInstance config
                si_config = self._db_bot_to_strategy_config(bot)

                # Use Factory to Hydrate/Validate
                try:
                    from src.config.configuration_factory import config_factory

                    si_config = config_factory.load_manifest(si_config)
                except Exception as e:
                    _logger.error("Bot %s: Factory validation/hydration failed: %s", bot["id"], e)
                    _update_bot_status_db(bot["id"], "error", error_message=f"Config Factory Error: {e}")
                    continue

                # Recover state if this is a crash recovery
                if was_crashed:
                    _logger.info("Recovering state for bot %s...", bot["id"])
                    si_config = self._recover_bot_state(bot["id"], si_config)

                # Additional validation using StrategyHandler
                strategy_type = si_config.get("strategy", {}).get("type", "CustomStrategy")
                strategy_config = si_config.get("strategy", {})

                is_strategy_valid, strategy_errors, strategy_warnings = strategy_handler.validate_strategy_config(
                    strategy_type, strategy_config
                )

                if not is_strategy_valid:
                    _logger.error("Bot %s strategy config invalid: %s", bot["id"], strategy_errors)
                    _update_bot_status_db(
                        bot["id"],
                        "error",
                        error_message="Strategy validation failed: " + "; ".join(strategy_errors),
                    )
                    continue

                # Log strategy warnings
                for warning in strategy_warnings:
                    _logger.warning("Bot %s strategy: %s", bot["id"], warning)

                # Create or update instance
                if instance_id in self.strategy_instances:
                    # Update existing config
                    _logger.info("Updating existing bot instance: %s", bot_name)
                    self.strategy_instances[instance_id].config = si_config
                else:
                    # Delegate to InstanceService
                    self.instance_service.create_instance(instance_id, si_config)

                loaded += 1
                _logger.info("✅ Successfully loaded bot: %s", bot_name)

            _logger.info("=" * 80)
            _logger.info("CONFIGURATION LOADING COMPLETE: %d/%d bots loaded", loaded, len(bots))
            _logger.info("=" * 80)

            return loaded > 0

        except Exception:
            _logger.exception("Failed to load strategies from DB:")
            return False

    def _db_bot_to_strategy_config(self, bot: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a trading_bots row dict into StrategyInstance config shape."""
        cfg = bot.get("config") or {}
        # Prefer fields from config; fallback to DB row
        name = cfg.get("name") or bot.get("description") or f"Bot {bot.get('id')}"
        symbol = cfg.get("symbol") or cfg.get("trading_pair") or cfg.get("pair") or cfg.get("ticker")
        broker_cfg = cfg.get("broker") or {}
        strategy_cfg = cfg.get("strategy") or {}

        # Enabled if DB status not 'disabled' and config doesn't explicitly disable
        enabled = bot.get("status") != "disabled" and cfg.get("enabled", True)

        # Config is already modular in DB after migration (bot_id and modules are inside cfg)
        return {
            "bot_id": str(bot.get("id")),
            "id": str(bot.get("id")),
            "name": name,
            "enabled": enabled,
            "symbol": symbol or "BTCUSDT",
            **cfg,
        }

    async def start_db_polling(self, user_id: int | None = None, interval_seconds: int = 60):
        """Continuously poll DB and sync strategy processes.

        Actions on each poll:
        - Start new StrategyInstances for bots present in DB with status 'enabled' that aren't running yet.
        - Stop and remove instances for bots that became 'disabled' or were deleted.
        - Refresh configs for bots that changed.
        - Update bot statuses in DB to 'running' when started and 'stopped' when stopped.
        """
        self._db_poll_user_id = user_id
        self._db_poll_interval = max(5, interval_seconds)
        self._db_poll_running = True

        # Initial load
        await self.load_strategies_from_db(user_id)

        async def _poll_loop():
            while self._db_poll_running:
                try:
                    # Fetch enabled bots (status != disabled)
                    enabled_bots = {str(b["id"]): b for b in trading_service.get_enabled_bots(user_id)}

                    # Start or update instances for enabled bots
                    for bot_id, bot in enabled_bots.items():
                        db_bid = _coerce_db_bot_id(bot_id)
                        if db_bid is None:
                            _logger.warning("Skipping poll for bot with non-numeric id: %s", bot_id)
                            continue
                        is_valid, errors, _ = trading_service.validate_bot_configuration(db_bid)
                        if not is_valid:
                            _logger.error("Bot %s invalid during polling, skipping start. Errors: %s", bot_id, errors)
                            _update_bot_status_db(bot_id, "error", error_message="; ".join(errors))
                            continue

                        desired_status = bot.get("status", "enabled")
                        if bot_id not in self.strategy_instances:
                            si_config = self._db_bot_to_strategy_config(bot)
                            self.instance_service.create_instance(bot_id, si_config)

                        instance = self.strategy_instances.get(bot_id)
                        if instance is None:
                            _logger.error("Missing strategy instance for bot %s after create", bot_id)
                            continue
                        if desired_status in ("enabled", "starting") and instance.status != "running":
                            _update_bot_status_db(bot_id, "starting")
                            ok = await instance.start()
                            _update_bot_status_db(
                                bot_id,
                                "running" if ok else "error",
                                error_message=None if ok else instance.last_error,
                            )
                        else:
                            instance.config = self._db_bot_to_strategy_config(bot)

                    # Stop instances for bots no longer enabled
                    to_stop = [iid for iid in list(self.strategy_instances.keys()) if iid not in enabled_bots]
                    for iid in to_stop:
                        instance = self.strategy_instances.get(iid)
                        if instance and instance.status == "running":
                            await instance.stop()
                            _update_bot_status_db(iid, "stopped")
                        # Remove from manager
                        self.strategy_instances.pop(iid, None)

                    await asyncio.sleep(self._db_poll_interval)

                except asyncio.CancelledError:
                    break
                except Exception:
                    _logger.exception("Error during DB polling loop:")
                    await asyncio.sleep(self._db_poll_interval)

        # Launch poll loop
        self.db_poll_task = asyncio.create_task(_poll_loop())

    async def shutdown(self):
        """
        Gracefully shutdown the strategy manager with state persistence.

        This method ensures:
        - All running bots are stopped cleanly
        - Bot statuses are persisted to database
        - Resources are properly released
        - Clean shutdown is marked (no crash marker on restart)
        """
        _logger.info("🛑 Shutting down Enhanced Strategy Manager...")

        try:
            # Stop monitoring first
            await self.stop_monitoring()

            # Stop all strategy instances and persist their status
            _logger.info("Stopping all strategy instances and persisting statuses...")
            for instance_id, instance in self.strategy_instances.items():
                if instance.status == "running":
                    _logger.info("Stopping bot %s (%s)", instance.name, instance_id)

                    # Stop the bot
                    await instance.stop()

                    # Persist stopped status to database
                    _update_bot_status_db(instance_id, "stopped", error_message=None)
                    _logger.debug("Persisted stopped status for bot %s", instance_id)

            # Close broker manager
            _logger.info("Closing broker manager...")
            await self.broker_manager.shutdown()

            # Close notification client
            if self.notification_client is not None:
                _logger.info("Closing notification client...")
                await self.notification_client.close()

            # Mark clean shutdown (remove crash marker)
            self._mark_clean_shutdown()

            _logger.info("Enhanced Strategy Manager shutdown complete")

        except Exception:
            _logger.exception("Error during shutdown:")
            # Don't mark clean shutdown if errors occurred
            _logger.warning("Shutdown completed with errors — crash marker not removed")
