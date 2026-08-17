"""
Tests for ConfigValidator / validate_config_file
-------------------------------------------------

Guards against config/trading/*.json drifting out of sync with the runtime
schema (config/schemas/bot_config.yaml). trading.service loads its config
straight off disk at startup (see trading_bot.py / live_trading_bot.py) with
no format-conversion step, so a config file written in an outdated shape
fails validation and the service refuses to start.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.config_validator import validate_config_file

# The config file trading.service is started with in production
# (see docs/HLA/modules/trading-engine.md).
PRODUCTION_BOT_CONFIG = "config/trading/0001.json"


class TestProductionBotConfig:
    """The live trading bot must always be able to load and validate its own config."""

    def test_production_config_is_schema_valid(self):
        """config/trading/0001.json must pass schema validation with no errors or warnings."""
        is_valid, errors, warnings = validate_config_file(PRODUCTION_BOT_CONFIG)

        assert is_valid, f"Production bot config {PRODUCTION_BOT_CONFIG} is invalid: {errors}"
        assert errors == []
        assert warnings == [], f"Production bot config {PRODUCTION_BOT_CONFIG} has warnings: {warnings}"

    def test_production_config_hydrates_and_builds_live_trading_bot(self):
        """The full startup path (config_factory + LiveTradingBot) must succeed, not just the schema check."""
        from src.trading.live_trading_bot import LiveTradingBot

        bot = LiveTradingBot("0001.json")

        assert bot.instance_id
