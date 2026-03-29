import os
import sys
import json
import glob
import logging
from pathlib import Path
from datetime import datetime
import shutil
import uuid

# Add src to path if running directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Real imports
from src.trading.tools.plot_simulation import plot_result_file
from src.model.config_models import TradingBotConfig

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("batch_simulation")

# -----------------------------------------------------------------------------
# MOCKING INFRASTRUCTURE
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

# Create instance for DI
mock_repo = MockRepository()

# -----------------------------------------------------------------------------
# REAL IMPORTS
# -----------------------------------------------------------------------------
from src.trading.base_trading_bot import BaseTradingBot
from src.config.configuration_factory import config_factory

# -----------------------------------------------------------------------------
# BATCH SIMULATION BOT
# -----------------------------------------------------------------------------


class BatchSimulationBot(BaseTradingBot):
    """
    Subclass of BaseTradingBot for batch simulation.
    Overrides configuration loading and execution loop.
    """
    def __init__(self, config_file):
        """Initialize simulation bot."""
        self.config_file = config_file
        self.config = self._load_configuration()
 
        # Create components
        broker = self._create_broker()
        from src.strategy.custom_strategy import CustomStrategy
        strategy_class = CustomStrategy
        parameters = self._create_strategy_parameters()
        
        # BaseTradingBot expects a dict for config
        config_dict = self.config.model_dump()

        # Initialize paths manually
        data_dir = os.path.join(PROJECT_ROOT, "data", "bots", self.config.bot_id)
        os.makedirs(data_dir, exist_ok=True)
        self.state_file = os.path.join(data_dir, "state.json")

        # Call BaseTradingBot init
        BaseTradingBot.__init__(
            self,
            config=config_dict,
            strategy_class=strategy_class,
            parameters=parameters,
            broker=broker,
            paper_trading=True,
            bot_id=self.config.bot_id,
            trade_repository=mock_repo
        )

        # Simulation attributes
        self.data_feed = None
        self.cerebro = None
        self.should_stop = False
        self.error_count = 0
        self.monitor_thread = None
        self.trading_pair = self.config.symbol
        self.active_positions = {}
        self.strategy_instance = None

    def _create_broker(self):
        """Create a file-based broker for simulation."""
        from src.trading.broker.broker_factory import get_broker
        broker_config = self.config.broker.model_dump()
        return get_broker(broker_config)

    def _create_data_feed(self):
        """Create a CSV data feed for simulation."""
        from src.data.feed.data_feed_factory import DataFeedFactory
        broker_cfg = self.config.broker.model_dump()
        data_source = broker_cfg.get("data_source", {})
        
        data_config = {
            "data_source": "file",
            "symbol": self.config.symbol,
            "file_path": data_source.get("file_path"),
            "simulate_realtime": False,
        }
        self.data_feed = DataFeedFactory.create_data_feed(data_config)
        return self.data_feed is not None

    def _convert_to_legacy_format(self):
        """Convert config wrapper to dict format expected by BaseTradingBot."""
        return self.config._config

    def _load_configuration(self) -> TradingBotConfig:
        """Load and validate configuration using TradingBotConfig model."""
        try:
            hydrated_config = config_factory.load_manifest(self.config_file)
            return TradingBotConfig.model_validate(hydrated_config)
        except Exception as e:
            logger.exception("Failed to load/validate configuration")
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
        """Setup Backtrader engine for simulation."""
        try:
            import backtrader as bt
            self.cerebro = bt.Cerebro()

            # Add data feed
            self.cerebro.adddata(self.data_feed)

            # Add strategy
            self.cerebro.addstrategy(self.strategy_class, **self.parameters)

            # Setup initial cash on DEFAULT broker
            initial_balance = self.config.broker.cash
            self.cerebro.broker.setcash(initial_balance)
 
            # Setup commission on DEFAULT broker
            # Try to get from paper_trading_config, otherwise default
            pt_config = self.config.broker.paper_trading_config
            commission = pt_config.get("commission_rate", 0.001)
            self.cerebro.broker.setcommission(commission=commission)

            logger.info(f"Setup Backtrader with initial balance: {initial_balance}")
            return True

        except Exception as e:
            logger.exception(f"Error setting up Backtrader: {e}")
            return False

    def _create_strategy_parameters(self):
        """Override to handle strategy parameters more flexibly."""
        strat_params = self.config.strategy.parameters
        return {
            "strategy_config": strat_params
        }

    # Override notification methods to do nothing
    def notify_trade_event(self, *args, **kwargs): pass
    def notify_bot_event(self, *args, **kwargs): pass
    def notify_error(self, *args, **kwargs): pass
    def _initialize_bot_instance(self): pass # Skip DB init

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
                "strategy_config": self.config.strategy.model_dump(),
                "data_file": self.config.broker.model_dump().get("data_source", {}).get("file_path", "")
            }

        return {
            "bot_id": self.bot_id,
            "symbol": self.trading_pair,
            "initial_balance": self.initial_balance,
            "final_balance": self.current_balance,
            "total_pnl": self.total_pnl,
            "total_trades": len(self.trade_history),
            "trades": self.trade_history,
            "strategy_config": self.config.strategy.model_dump(),
            "data_file": self.config.broker.model_dump().get("data_source", {}).get("file_path", "")
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

            # Check if result already exists to allow resuming
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
            temp_dir.mkdir(parents=True, exist_ok=True)
            with open(temp_broker_path, 'w') as f:
                json.dump(current_broker, f, indent=2)

            # Prepare Manifest
            manifest = manifest_template.copy()
            manifest["bot_id"] = f"{data_name}-{strategy_name}"
            manifest["name"] = f"Sim {symbol} {strategy_name}"
            manifest["symbol"] = symbol

            if "modules" not in manifest:
                 manifest["modules"] = {}

            if "components" in manifest:
                comps = manifest.pop("components")
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
            temp_dir.mkdir(parents=True, exist_ok=True)
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

    # Cleanup temp
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up temp dir: {e}")

if __name__ == "__main__":
    run_batch_simulation()
    print("Done. Now you can run src/trading/tools/analyze_simulation_results.py")
