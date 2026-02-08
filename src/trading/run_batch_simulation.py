import os
import sys
import json
import glob
import logging
from pathlib import Path
from datetime import datetime
import shutil
import uuid
import types

# Add src to path if running directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Real imports
from src.trading.tools.plot_simulation import plot_result_file

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("batch_simulation")

# -----------------------------------------------------------------------------
# MOCKING INFRASTRUCTURE BEFORE IMPORTS
# -----------------------------------------------------------------------------
class MockRepository:
    """Mock repository to bypass database calls."""
    def get_bot_instance(self, bot_id): return None
    def create_bot_instance(self, data): return None
    def update_bot_instance(self, bot_id, data): return None
    def get_open_trades(self, bot_id, symbol): return []
    def create_trade(self, data):
        # Return a dummy object with an id
        class DummyTrade:
            id = str(uuid.uuid4())
        return DummyTrade()
    def update_trade(self, trade_id, data): return None
    def ensure_open_position(self, *args, **kwargs): return None
    def get_open_positions(self, *args, **kwargs): return []
    def heartbeat(self, *args, **kwargs): return None

# Create mock module for trading_bot_service
mock_service_module = types.ModuleType("src.trading.services.trading_bot_service")
mock_service_module.trading_bot_service = MockRepository()
sys.modules["src.trading.services.trading_bot_service"] = mock_service_module

# Create mock module for db services to be safe
mock_db_module = types.ModuleType("src.data.db.services")
mock_db_module.trading_service = MockRepository()
sys.modules["src.data.db.services"] = mock_db_module

# -----------------------------------------------------------------------------
# REAL IMPORTS
# -----------------------------------------------------------------------------
from src.trading.live_trading_bot import LiveTradingBot
from src.config.configuration_factory import config_factory

class ConfigWrapper:
    """
    Wrapper for configuration dict to mimic TradingBotConfig and provide legacy config methods.
    Bypasses the broken TradingBotConfig model in the codebase.
    """
    def __init__(self, config_dict):
        self._config = config_dict

        # Attribute access for common fields
        self.bot_id = config_dict.get("bot_id", "sim_bot")
        self.symbol = config_dict.get("symbol", "UNKNOWN")
        self.initial_balance = config_dict.get("initial_balance", 10000.0)
        self.paper_trading = True # Force paper trading for simulation

        # Factory flattens modules into the root config
        self._modules = config_dict

        # Try to determine strategy type/name
        self.strategy_type = type("StrategyType", (), {"value": "CustomStrategy"}) # Default object with value attr

        # Try to find strategy name from components
        if "strategy" in self._modules:
            strat = self._modules["strategy"]
            if isinstance(strat, dict) and "name" in strat:
                pass

        # Parse position sizing
        self.position_size = 0.1 # Default
        if "portfolio_manager" in self._modules:
            pm = self._modules["portfolio_manager"]
            if isinstance(pm, dict) and "order_size_pct" in pm:
                 self.position_size = pm["order_size_pct"]

        # Risk parameters
        self.max_drawdown_pct = 20.0
        self.max_exposure = 1.0

        # Commission
        self.commission = 0.001 # Default
        broker_cfg = self.get_broker_config()
        if "paper_trading_config" in broker_cfg:
            self.commission = broker_cfg["paper_trading_config"].get("commission_rate", 0.001)

        # Broker Type (Added Fix)
        self.broker_type = broker_cfg.get("type", "file_broker")

        # Strategy Params
        self.strategy_params = {}
        # Try to extract entry/exit params from components if they exist
        strategy_logic = self._modules.get("strategy", {})
        if isinstance(strategy_logic, dict):
             pass

    def get_broker_config(self):
        # Factory puts "broker" key at root
        return self._modules.get("broker", {})

    def get_trading_config(self):
        return {} # Defaults

    def get_data_config(self):
        # Construct data config from broker data source or overrides
        broker_cfg = self.get_broker_config()
        data_source = broker_cfg.get("data_source", {})

        return {
            "data_source": "file",
            "symbol": self.symbol,
            "file_path": data_source.get("file_path"),
            "simulate_realtime": False,
        }

    def get_strategy_config(self):
        return self._modules.get("strategy", {})

    def get_risk_management_config(self):
         return self._modules.get("global_risk", {})

    def get_logging_config(self):
        return {"level": "INFO"}

    def get_notifications_config(self):
        return self._modules.get("notifications", {})


class BatchSimulationBot(LiveTradingBot):
    """
    Subclass of LiveTradingBot for batch simulation.
    Overrides configuration loading and execution loop.
    """
    def __init__(self, config_file):
        # We need to bypass LiveTradingBot.__init__ because it loads config using the broken model
        # So we copy the essential parts of __init__ here

        self.config_file = config_file
        config_wrapper = self._load_configuration()
        self.config = config_wrapper

        # Extract components for BaseTradingBot (copied from LiveTradingBot)
        # Note: _create_broker uses self.config.get_broker_config()
        broker = self._create_broker()

        # Strategy Class
        from src.strategy.custom_strategy import CustomStrategy
        strategy_class = CustomStrategy

        parameters = self._create_strategy_parameters()

        # Initialize BaseTradingBot
        legacy_config = self._convert_to_legacy_format() # Uses self.config methods

        # Initialize paths manually as BaseTradingBot expects state_file but doesn't seem to set it (Added Fix)
        data_dir = os.path.join(PROJECT_ROOT, "data", "bots", self.config.bot_id)
        os.makedirs(data_dir, exist_ok=True)
        self.state_file = os.path.join(data_dir, "state.json")

        # Call BaseTradingBot init directly (skip LiveTradingBot init)
        from src.trading.base_trading_bot import BaseTradingBot
        BaseTradingBot.__init__(
            self,
            config=legacy_config,
            strategy_class=strategy_class,
            parameters=parameters,
            broker=broker,
            paper_trading=True,
            bot_id=self.config.bot_id
        )

        # Restore ConfigWrapper because BaseTradingBot overwrites self.config with the dict (Added Fix)
        # We must restore it for LiveTradingBot methods that expect Pydantic/Wrapper interface
        self.config = config_wrapper

        # LiveTradingBot specific attributes
        self.data_feed = None
        self.cerebro = None
        self.should_stop = False
        self.error_count = 0
        self.monitor_thread = None
        self.trading_pair = self.config.symbol
        self.active_positions = {}
        self.strategy_instance = None

    def _load_configuration(self):
        """Override to return our ConfigWrapper."""
        try:
            # We assume config_file is a full path to the temp manifest
            hydrated_config = config_factory.load_manifest(self.config_file)
            return ConfigWrapper(hydrated_config)
        except Exception as e:
            logger.exception("Failed to load configuration wrapper")
            raise

    def start(self):
        """Override start to run synchronously without monitoring thread."""
        logger.info(f"Starting simulation for {self.config.bot_id}")

        if not self._create_data_feed():
            raise RuntimeError("Failed to create data feed")

        if not self._setup_backtrader():
            raise RuntimeError("Failed to setup Backtrader")

        # Run Backtrader directly
        self._run_backtrader()

    def _run_backtrader(self):
        """Run Backtrader and capture the strategy instance."""
        try:
            results = self.cerebro.run()
            if results and len(results) > 0:
                self.strategy_instance = results[0]
            return True
        except Exception as e:
            logger.exception(f"Error running Backtrader: {e}")
            return False

    def _setup_backtrader(self):
        """
        Setup Backtrader engine for simulation.
        Override to use default Backtrader broker instead of our custom MockBroker,
        as MockBroker is not compatible with Backtrader's engine.
        """
        try:
            import backtrader as bt
            self.cerebro = bt.Cerebro()

            # Add data feed
            self.cerebro.adddata(self.data_feed)

            # Add strategy
            self.cerebro.addstrategy(self.strategy_class, **self.parameters)

            # Setup initial cash on DEFAULT broker
            initial_balance = self.config.initial_balance
            self.cerebro.broker.setcash(initial_balance)

            # Setup commission on DEFAULT broker
            commission = self.config.commission
            self.cerebro.broker.setcommission(commission=commission)

            logger.info(f"Setup Backtrader with initial balance: {initial_balance}")
            return True

        except Exception as e:
            logger.exception(f"Error setting up Backtrader: {e}")
            return False

    def _create_strategy_parameters(self):
        """Override to handle strategy parameters more flexibly."""
        # Directly pass the loaded strategy configuration to the strategy
        strat_config = self.config.get_strategy_config()

        # Unwrap parameters if nested
        if isinstance(strat_config, dict) and "parameters" in strat_config:
            strat_config = strat_config["parameters"]

        return {
            "strategy_config": strat_config
        }

    # Override notification methods to do nothing
    def notify_trade_event(self, *args, **kwargs): pass
    def notify_bot_event(self, *args, **kwargs): pass
    def notify_error(self, *args, **kwargs): pass
    def _initialize_bot_instance(self): pass # Skip DB init


    # Helper to get results
    def get_simulation_results(self):
        if self.strategy_instance is not None:
            # Extract results from strategy instance
            stats = self.strategy_instance.get_performance_summary()
            trades = getattr(self.strategy_instance, 'trades', [])

            # Format results
            return {
                "bot_id": self.bot_id,
                "symbol": self.trading_pair,
                "initial_balance": self.initial_balance,
                "final_balance": stats.get("current_equity", self.current_balance),
                "total_pnl": stats.get("total_pnl", 0.0),
                "total_trades": stats.get("total_trades", len(trades)),
                "trades": trades,
                "win_rate": stats.get("win_rate", 0.0),
                "max_drawdown": stats.get("max_drawdown", 0.0),
                "strategy_config": self.config.get_strategy_config(),
                "data_file": self.config.get_data_config().get("file_path", "")
            }

        # Fallback to local tracking (mostly empty for simulation)
        return {
            "bot_id": self.bot_id,
            "symbol": self.trading_pair,
            "initial_balance": self.initial_balance,
            "final_balance": self.current_balance,
            "total_pnl": self.total_pnl,
            "total_trades": len(self.trade_history),
            "trades": self.trade_history,
            "strategy_config": self.config.get_strategy_config(),
            "data_file": self.config.get_data_config().get("file_path", "")
        }


def run_batch_simulation():
    # Paths
    data_dir = PROJECT_ROOT / "data"
    strategy_dir = PROJECT_ROOT / "config" / "contracts" / "instances" / "strategies"
    template_path = PROJECT_ROOT / "config" / "contracts" / "instances" / "bots" / "manifest-template.json"
    results_dir = PROJECT_ROOT / "results" / "simulation"
    broker_template_path = PROJECT_ROOT / "config" / "contracts" / "instances" / "brokers" / "broker-file-sim.json"

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = results_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Load templates
    try:
        with open(template_path, 'r') as f:
            manifest_template = json.load(f)

        with open(broker_template_path, 'r') as f:
            broker_template = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Template not found: {e}")
        return

    # Find files
    csv_files = glob.glob(str(data_dir / "*.csv"))
    strategy_files = glob.glob(str(strategy_dir / "*.json"))

    logger.info(f"Found {len(csv_files)} data files and {len(strategy_files)} strategies. Total combinations: {len(csv_files) * len(strategy_files)}")

    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return
    if not strategy_files:
        logger.warning(f"No strategy files found in {strategy_dir}")
        return

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        filename_parts = csv_path.name.split('_')
        symbol = filename_parts[0] if filename_parts else "UNKNOWN"
        data_name = csv_path.stem

        for strategy_file in strategy_files:
            strategy_path = Path(strategy_file)
            strategy_name = strategy_path.stem

            # Check if result already exists to allow resuming (Added Fix)
            report_path = results_dir / f"{data_name}-{strategy_name}.json"
            if report_path.exists():
                logger.info(f"Skipping already completed simulation: {report_path.name}")
                continue

            logger.info(f"Running simulation: {data_name} + {strategy_name}")

            # Prepare Broker Config
            current_broker = broker_template.copy()
            if "broker" in current_broker:
                 current_broker["broker"]["data_source"]["file_path"] = str(csv_path)

            # Write temp broker
            temp_broker_path = temp_dir / f"{data_name}-broker.json"
            temp_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
            with open(temp_broker_path, 'w') as f:
                json.dump(current_broker, f, indent=2)

            # Prepare Manifest
            manifest = manifest_template.copy()
            manifest["bot_id"] = f"{data_name}-{strategy_name}"
            manifest["name"] = f"Sim {symbol} {strategy_name}"
            manifest["symbol"] = symbol

            # Use modules structure
            if "modules" not in manifest:
                 manifest["modules"] = {}

            if "components" in manifest:
                comps = manifest.pop("components")
                # Map strategy_logic -> strategy
                if "strategy_logic" in comps:
                     manifest["modules"]["strategy"] = comps["strategy_logic"]

                # Copy other components
                for k, v in comps.items():
                    if k != "strategy_logic" and k != "broker":
                         manifest["modules"][k] = v

            # Verify and fix specific modules
            manifest["modules"]["broker"] = str(temp_broker_path)
            manifest["modules"]["strategy"] = str(strategy_path)

            # Fix incorrect paths from template
            manifest["modules"]["strategy_risk"] = "config/contracts/instances/strategy-risks/risk-strategy-standard.json"
            manifest["modules"]["global_risk"] = "config/contracts/instances/risk/risk-global-aggressive.json"
            manifest["modules"]["portfolio_manager"] = "config/contracts/instances/portfolio/portfolio-standard.json"
            manifest["modules"]["notifications"] = "config/contracts/instances/notifications/notifications-file.json"

            # Write temp manifest
            temp_manifest_path = temp_dir / f"{manifest['bot_id']}.json"
            temp_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
            with open(temp_manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            try:
                # Run Bot
                bot = BatchSimulationBot(str(temp_manifest_path))
                bot.start()

                # Get Results
                results = bot.get_simulation_results()

                # Save Report
                with open(report_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                logger.info(f"Report saved to {report_path}. PnL: {results['total_pnl']:.2f}%")

                # Generate Plot
                try:
                    plot_path = plot_result_file(str(report_path))
                    if plot_path:
                        logger.info(f"Plot generated: {plot_path}")
                except Exception as pe:
                    logger.error(f"Failed to generate plot for {report_path}: {pe}")

            except BaseException as e:
                logger.exception(f"Error running simulation for {data_name} - {strategy_name}")
                import traceback
                traceback.print_exc()

    # Cleanup temp
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up temp dir: {e}")

if __name__ == "__main__":
    run_batch_simulation()
    print("Done. Now you can run src/trading/tools/analyze_simulation_results.py")
