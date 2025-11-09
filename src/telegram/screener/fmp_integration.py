"""
FMP Integration Module

This module provides shared FMP (Financial Modeling Prep) integration functionality
for all screener types. It handles FMP-based initial screening and criteria management.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from config.donotshare.donotshare import FMP_API_KEY
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class FMPIntegration:
    """
    FMP Integration class that provides FMP screening functionality
    for all screener types in the system.
    """

    def __init__(self):
        """Initialize FMP integration with configuration loading."""
        self.config_path = Path(__file__).resolve().parents[4] / "config" / "screener" / "fmp_screener_criteria.json"
        self.fmp_config = self._load_fmp_config()
        self.fmp_downloader = None

    def _load_fmp_config(self) -> Dict[str, Any]:
        """Load FMP configuration from JSON file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                _logger.info("FMP configuration loaded successfully")
                return config
            else:
                _logger.warning("FMP configuration file not found, using defaults")
                return self._get_default_config()
        except Exception:
            _logger.exception("Error loading FMP configuration:")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default FMP configuration if file is not available."""
        return {
            "default_configs": {
                "fundamental": {
                    "description": "Default FMP criteria for fundamental screening",
                    "fmp_criteria": {
                        "marketCapMoreThan": 1000000000,
                        "peRatioLessThan": 20,
                        "returnOnEquityMoreThan": 0.10,
                        "limit": 100
                    }
                },
                "hybrid": {
                    "description": "Default FMP criteria for hybrid screening",
                    "fmp_criteria": {
                        "marketCapMoreThan": 2000000000,
                        "peRatioLessThan": 25,
                        "returnOnEquityMoreThan": 0.12,
                        "limit": 80
                    }
                },
                "technical": {
                    "description": "Default FMP criteria for technical screening",
                    "fmp_criteria": {
                        "marketCapMoreThan": 500000000,
                        "limit": 150
                    }
                }
            },
            "predefined_strategies": {}
        }

    def _get_fmp_downloader(self) -> Optional[FMPDataDownloader]:
        """Get FMP downloader instance, initializing if needed."""
        if self.fmp_downloader is None:
            try:
                api_key = FMP_API_KEY
                if not api_key:
                    _logger.warning("FMP_API_KEY not set, FMP screening will be skipped")
                    return None

                self.fmp_downloader = FMPDataDownloader(api_key=api_key)
                _logger.info("FMP downloader initialized successfully")
            except Exception:
                _logger.exception("Error initializing FMP downloader:")
                return None

        return self.fmp_downloader

    def get_fmp_criteria(self, screener_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get FMP criteria for screening based on configuration.

        Args:
            screener_config: Screener configuration containing FMP criteria or screener type

        Returns:
            Dictionary with FMP criteria to use for screening
        """
        # Check if user provided explicit FMP criteria
        if "fmp_criteria" in screener_config:
            _logger.info("Using user-provided FMP criteria")
            return screener_config["fmp_criteria"]

        # Check for predefined strategy
        if "fmp_strategy" in screener_config:
            strategy_name = screener_config["fmp_strategy"]
            if strategy_name in self.fmp_config.get("predefined_strategies", {}):
                strategy_config = self.fmp_config["predefined_strategies"][strategy_name]
                _logger.info("Using predefined FMP strategy: %s", strategy_name)
                return strategy_config.get("fmp_criteria", {})
            else:
                _logger.warning("Predefined FMP strategy not found: %s", strategy_name)

        # Check for screener type defaults
        screener_type = screener_config.get("screener_type")
        if screener_type and screener_type in self.fmp_config.get("default_configs", {}):
            default_config = self.fmp_config["default_configs"][screener_type]
            _logger.info("Using default FMP criteria for screener type: %s", screener_type)
            return default_config.get("fmp_criteria", {})

        # Fallback to hybrid default
        _logger.info("Using fallback hybrid FMP criteria")
        return self.fmp_config.get("default_configs", {}).get("hybrid", {}).get("fmp_criteria", {"limit": 100})

    def run_fmp_screening(self, screener_config: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Run FMP screening based on configuration.

        Args:
            screener_config: Screener configuration containing FMP criteria or strategy

        Returns:
            Tuple of (ticker_list, fmp_results)
        """
        try:
            # Get FMP criteria
            fmp_criteria = self.get_fmp_criteria(screener_config)

            if not fmp_criteria:
                _logger.warning("No FMP criteria found, skipping FMP screening")
                return [], {}

            # Get FMP downloader
            fmp_downloader = self._get_fmp_downloader()
            if not fmp_downloader:
                _logger.warning("FMP downloader not available, skipping FMP screening")
                return [], {}

            # Run FMP screening
            _logger.info("Running FMP screening with criteria: %s", fmp_criteria)
            fmp_results = fmp_downloader.get_stock_screener(fmp_criteria)

            if fmp_results:
                ticker_list = [stock['symbol'] for stock in fmp_results]
                _logger.info("FMP screening completed: %d tickers found", len(ticker_list))
                return ticker_list, {"fmp_results": fmp_results, "fmp_criteria": fmp_criteria}
            else:
                _logger.warning("FMP screening returned no results")
                return [], {}

        except Exception:
            _logger.exception("Error in FMP screening")
            return [], {}

    def get_fmp_fundamentals(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Get fundamental data for a list of tickers using FMP.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker symbols to fundamental data
        """
        try:
            fmp_downloader = self._get_fmp_downloader()
            if not fmp_downloader:
                _logger.warning("FMP downloader not available, cannot get fundamentals")
                return {}

            _logger.info("Getting FMP fundamentals for %d tickers", len(tickers))
            fundamentals = fmp_downloader.get_fundamentals_batch(tickers)
            _logger.info("FMP fundamentals retrieved for %d tickers", len(fundamentals))

            return fundamentals

        except Exception:
            _logger.exception("Error getting FMP fundamentals")
            return {}

    def validate_fmp_criteria(self, fmp_criteria: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate FMP criteria to ensure they are supported.

        Args:
            fmp_criteria: FMP criteria to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check if criteria is a dictionary
        if not isinstance(fmp_criteria, dict):
            errors.append("FMP criteria must be a dictionary")
            return False, errors

        # Get supported criteria from FMP downloader
        try:
            downloader = self._get_fmp_downloader()
            if downloader:
                # Use the validation method from FMP downloader
                downloader._validate_screener_criteria(fmp_criteria)
            else:
                # Basic validation without FMP downloader
                supported_criteria = {
                    "marketCapMoreThan", "marketCapLowerThan",
                    "peRatioLessThan", "peRatioMoreThan",
                    "priceToBookRatioLessThan", "priceToBookRatioMoreThan",
                    "priceToSalesRatioLessThan", "priceToSalesRatioMoreThan",
                    "debtToEquityLessThan", "debtToEquityMoreThan",
                    "currentRatioMoreThan", "currentRatioLessThan",
                    "quickRatioMoreThan", "quickRatioLessThan",
                    "returnOnEquityMoreThan", "returnOnEquityLessThan",
                    "returnOnAssetsMoreThan", "returnOnAssetsLessThan",
                    "returnOnCapitalEmployedMoreThan", "returnOnCapitalEmployedLessThan",
                    "dividendYieldMoreThan", "dividendYieldLessThan",
                    "payoutRatioLessThan", "payoutRatioMoreThan",
                    "betaLessThan", "betaMoreThan",
                    "exchange", "Country", "isETF", "isFund", "isActivelyTrading", "limit"
                }

                invalid_criteria = set(fmp_criteria.keys()) - supported_criteria
                if invalid_criteria:
                    errors.append(f"Unsupported FMP criteria: {', '.join(invalid_criteria)}")

        except Exception as e:
            errors.append(f"Error validating FMP criteria: {str(e)}")

        return len(errors) == 0, errors

    def get_available_fmp_strategies(self) -> Dict[str, str]:
        """
        Get list of available predefined FMP strategies.

        Returns:
            Dictionary mapping strategy names to descriptions
        """
        strategies = {}
        predefined = self.fmp_config.get("predefined_strategies", {})

        for name, config in predefined.items():
            strategies[name] = config.get("description", f"Strategy: {name}")

        return strategies

    def get_default_fmp_configs(self) -> Dict[str, str]:
        """
        Get list of available default FMP configurations.

        Returns:
            Dictionary mapping config names to descriptions
        """
        configs = {}
        defaults = self.fmp_config.get("default_configs", {})

        for name, config in defaults.items():
            configs[name] = config.get("description", f"Default config: {name}")

        return configs


# Global FMP integration instance
_fmp_integration = None

def get_fmp_integration() -> FMPIntegration:
    """Get global FMP integration instance."""
    global _fmp_integration
    if _fmp_integration is None:
        _fmp_integration = FMPIntegration()
    return _fmp_integration


def run_fmp_screening(screener_config: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Convenience function to run FMP screening.

    Args:
        screener_config: Screener configuration

    Returns:
        Tuple of (ticker_list, fmp_results)
    """
    fmp_integration = get_fmp_integration()
    return fmp_integration.run_fmp_screening(screener_config)


def get_fmp_criteria(screener_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to get FMP criteria.

    Args:
        screener_config: Screener configuration

    Returns:
        FMP criteria dictionary
    """
    fmp_integration = get_fmp_integration()
    return fmp_integration.get_fmp_criteria(screener_config)


def validate_fmp_criteria(fmp_criteria: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate FMP criteria.

    Args:
        fmp_criteria: FMP criteria to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    fmp_integration = get_fmp_integration()
    return fmp_integration.validate_fmp_criteria(fmp_criteria)
