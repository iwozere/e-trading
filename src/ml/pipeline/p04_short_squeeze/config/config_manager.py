"""
Configuration manager for the Short Squeeze Detection Pipeline.

This module provides YAML configuration loading, validation, and type-safe access
to all pipeline configuration parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from src.notification.logger import setup_logger
from src.ml.pipeline.p04_short_squeeze.config.data_classes import (
    PipelineConfig, SchedulingConfig, ScreenerConfig, DeepScanConfig,
    AlertConfig, AdHocConfig, ReportConfig, PerformanceConfig, ScoringConfig,
    UniverseConfig, ScreenerFilters, ScreenerWeights, DeepScanMetrics,
    DeepScanWeights, AlertThresholds, AlertThreshold, AlertCooldown,
    AlertChannels, WeeklyReportConfig, DailyReportConfig, ApiRateLimits,
    DatabaseConfig, ErrorHandlingConfig
)

# Set up project root path
PROJECT_ROOT = Path(__file__).resolve().parents[5]

_logger = setup_logger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigManager:
    """
    Manages configuration loading, validation, and access for the pipeline.

    Supports loading from YAML files with environment variable substitution
    and provides type-safe access to configuration parameters.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file. If None, uses default location.
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[PipelineConfig] = None
        self._raw_config: Optional[Dict[str, Any]] = None

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        return str(PROJECT_ROOT / "config" / "pipeline" / "p04_short_squeeze.yaml")

    def load_config(self, config_path: Optional[str] = None) -> PipelineConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Optional path to configuration file. Uses instance path if None.

        Returns:
            Loaded and validated PipelineConfig instance.

        Raises:
            ConfigValidationError: If configuration is invalid.
            FileNotFoundError: If configuration file doesn't exist.
        """
        if config_path:
            self.config_path = config_path

        _logger.info("Loading configuration from %s", self.config_path)

        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                raw_config = yaml.safe_load(file)

            # Substitute environment variables
            raw_config = self._substitute_env_vars(raw_config)
            self._raw_config = raw_config

            # Convert to typed configuration
            self._config = self._build_config_from_dict(raw_config)

            # Validate configuration
            self._validate_config(self._config)

            _logger.info("Configuration loaded successfully with run_id: %s", self._config.run_id)
            return self._config

        except FileNotFoundError:
            _logger.error("Configuration file not found: %s", self.config_path)
            raise
        except yaml.YAMLError as e:
            _logger.exception("Failed to parse YAML configuration:")
            raise ConfigValidationError(f"Invalid YAML syntax: {e}")
        except Exception as e:
            _logger.exception("Failed to load configuration:")
            raise ConfigValidationError(f"Configuration loading failed: {e}")

    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively substitute environment variables in configuration.

        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        """
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_string_env_vars(config)
        else:
            return config

    def _substitute_string_env_vars(self, value: str) -> str:
        """Substitute environment variables in a string value."""
        import re

        def replace_env_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                env_value = os.getenv(var_expr)
                if env_value is None:
                    raise ConfigValidationError(f"Environment variable {var_expr} not found")
                return env_value

        return re.sub(r'\$\{([^}]+)\}', replace_env_var, value)

    def _build_config_from_dict(self, config_dict: Dict[str, Any]) -> PipelineConfig:
        """Build typed configuration from dictionary."""
        try:
            # Build nested configurations
            scheduling = self._build_scheduling_config(config_dict.get('scheduling', {}))
            screener = self._build_screener_config(config_dict.get('screener', {}))
            deep_scan = self._build_deep_scan_config(config_dict.get('deep_scan', {}))
            alerting = self._build_alert_config(config_dict.get('alerting', {}))
            adhoc = self._build_adhoc_config(config_dict.get('adhoc', {}))
            reporting = self._build_report_config(config_dict.get('reporting', {}))
            performance = self._build_performance_config(config_dict.get('performance', {}))
            scoring = self._build_scoring_config(config_dict.get('scoring', {}))

            return PipelineConfig(
                scheduling=scheduling,
                screener=screener,
                deep_scan=deep_scan,
                alerting=alerting,
                adhoc=adhoc,
                reporting=reporting,
                performance=performance,
                scoring=scoring
            )

        except Exception as e:
            raise ConfigValidationError(f"Failed to build configuration: {e}")

    def _build_scheduling_config(self, config: Dict[str, Any]) -> SchedulingConfig:
        """Build scheduling configuration."""
        screener_config = config.get('screener', {})
        deep_scan_config = config.get('deep_scan', {})

        return SchedulingConfig(
            screener_frequency=screener_config.get('frequency', 'weekly'),
            screener_day=screener_config.get('day', 'monday'),
            screener_time=screener_config.get('time', '08:00'),
            deep_scan_frequency=deep_scan_config.get('frequency', 'daily'),
            deep_scan_time=deep_scan_config.get('time', '10:00'),
            timezone=config.get('timezone', 'Europe/Zurich')
        )

    def _build_screener_config(self, config: Dict[str, Any]) -> ScreenerConfig:
        """Build screener configuration."""
        universe_config = config.get('universe', {})
        filters_config = config.get('filters', {})
        scoring_config = config.get('scoring', {}).get('weights', {})

        universe = UniverseConfig(
            min_market_cap=universe_config.get('min_market_cap', 100_000_000),
            max_market_cap=universe_config.get('max_market_cap', 10_000_000_000),
            min_avg_volume=universe_config.get('min_avg_volume', 200_000),
            exchanges=universe_config.get('exchanges', ['NYSE', 'NASDAQ'])
        )

        filters = ScreenerFilters(
            si_percent_min=filters_config.get('si_percent_min', 0.15),
            days_to_cover_min=filters_config.get('days_to_cover_min', 5.0),
            float_max=filters_config.get('float_max', 100_000_000),
            top_k_candidates=filters_config.get('top_k_candidates', 50)
        )

        scoring = ScreenerWeights(
            short_interest_pct=scoring_config.get('short_interest_pct', 0.4),
            days_to_cover=scoring_config.get('days_to_cover', 0.3),
            float_ratio=scoring_config.get('float_ratio', 0.2),
            volume_consistency=scoring_config.get('volume_consistency', 0.1)
        )

        return ScreenerConfig(universe=universe, filters=filters, scoring=scoring)

    def _build_deep_scan_config(self, config: Dict[str, Any]) -> DeepScanConfig:
        """Build deep scan configuration."""
        metrics_config = config.get('metrics', {})
        scoring_config = config.get('scoring', {}).get('weights', {})

        metrics = DeepScanMetrics(
            volume_lookback_days=metrics_config.get('volume_lookback_days', 14),
            sentiment_lookback_hours=metrics_config.get('sentiment_lookback_hours', 24),
            options_min_volume=metrics_config.get('options_min_volume', 100)
        )

        scoring = DeepScanWeights(
            volume_spike=scoring_config.get('volume_spike', 0.35),
            sentiment_24h=scoring_config.get('sentiment_24h', 0.25),
            call_put_ratio=scoring_config.get('call_put_ratio', 0.20),
            borrow_fee=scoring_config.get('borrow_fee', 0.20)
        )

        return DeepScanConfig(
            batch_size=config.get('batch_size', 10),
            api_delay_seconds=config.get('api_delay_seconds', 0.2),
            metrics=metrics,
            scoring=scoring
        )

    def _build_alert_config(self, config: Dict[str, Any]) -> AlertConfig:
        """Build alert configuration."""
        thresholds_config = config.get('thresholds', {})
        cooldown_config = config.get('cooldown', {})
        channels_config = config.get('channels', {})

        # Build threshold configurations
        high_config = thresholds_config.get('high', {})
        medium_config = thresholds_config.get('medium', {})
        low_config = thresholds_config.get('low', {})

        thresholds = AlertThresholds(
            high=AlertThreshold(
                squeeze_score=high_config.get('squeeze_score', 0.8),
                min_si_percent=high_config.get('min_si_percent', 0.25),
                min_volume_spike=high_config.get('min_volume_spike', 4.0),
                min_sentiment=high_config.get('min_sentiment', 0.6)
            ),
            medium=AlertThreshold(
                squeeze_score=medium_config.get('squeeze_score', 0.6),
                min_si_percent=medium_config.get('min_si_percent', 0.20),
                min_volume_spike=medium_config.get('min_volume_spike', 3.0),
                min_sentiment=medium_config.get('min_sentiment', 0.5)
            ),
            low=AlertThreshold(
                squeeze_score=low_config.get('squeeze_score', 0.4),
                min_si_percent=low_config.get('min_si_percent', 0.15),
                min_volume_spike=low_config.get('min_volume_spike', 2.0),
                min_sentiment=low_config.get('min_sentiment', 0.4)
            )
        )

        cooldown = AlertCooldown(
            high_alert_days=cooldown_config.get('high_alert_days', 7),
            medium_alert_days=cooldown_config.get('medium_alert_days', 5),
            low_alert_days=cooldown_config.get('low_alert_days', 3)
        )

        telegram_config = channels_config.get('telegram', {})
        email_config = channels_config.get('email', {})

        channels = AlertChannels(
            telegram_enabled=telegram_config.get('enabled', True),
            telegram_chat_ids=telegram_config.get('chat_ids', ['@trading_alerts']),
            email_enabled=email_config.get('enabled', True),
            email_recipients=email_config.get('recipients', ['trader@example.com'])
        )

        return AlertConfig(thresholds=thresholds, cooldown=cooldown, channels=channels)

    def _build_adhoc_config(self, config: Dict[str, Any]) -> AdHocConfig:
        """Build ad-hoc configuration."""
        return AdHocConfig(
            default_ttl_days=config.get('default_ttl_days', 7),
            max_active_candidates=config.get('max_active_candidates', 20),
            auto_promote_threshold=config.get('auto_promote_threshold', 0.7)
        )

    def _build_report_config(self, config: Dict[str, Any]) -> ReportConfig:
        """Build report configuration."""
        weekly_config = config.get('weekly_summary', {})
        daily_config = config.get('daily_report', {})

        weekly = WeeklyReportConfig(
            top_candidates=weekly_config.get('top_candidates', 20),
            include_charts=weekly_config.get('include_charts', True),
            formats=weekly_config.get('formats', ['html', 'csv'])
        )

        daily = DailyReportConfig(
            top_scores=daily_config.get('top_scores', 10),
            include_trends=daily_config.get('include_trends', True),
            formats=daily_config.get('formats', ['html'])
        )

        return ReportConfig(weekly_summary=weekly, daily_report=daily)

    def _build_performance_config(self, config: Dict[str, Any]) -> PerformanceConfig:
        """Build performance configuration."""
        api_config = config.get('api_rate_limits', {})
        db_config = config.get('database', {})
        error_config = config.get('error_handling', {})

        api_limits = ApiRateLimits(
            fmp_calls_per_minute=api_config.get('fmp_calls_per_minute', 250),
            finnhub_calls_per_minute=api_config.get('finnhub_calls_per_minute', 50)
        )

        database = DatabaseConfig(
            batch_size=db_config.get('batch_size', 100),
            connection_timeout=db_config.get('connection_timeout', 30),
            query_timeout=db_config.get('query_timeout', 60)
        )

        error_handling = ErrorHandlingConfig(
            max_retries=error_config.get('max_retries', 3),
            backoff_factor=error_config.get('backoff_factor', 2.0),
            circuit_breaker_threshold=error_config.get('circuit_breaker_threshold', 0.5)
        )

        return PerformanceConfig(
            api_rate_limits=api_limits,
            database=database,
            error_handling=error_handling
        )

    def _build_scoring_config(self, config: Dict[str, Any]) -> ScoringConfig:
        """Build scoring configuration."""
        return ScoringConfig(
            normalization_method=config.get('normalization_method', 'minmax'),
            score_bounds=tuple(config.get('score_bounds', [0.0, 1.0])),
            weight_validation=config.get('weight_validation', True)
        )

    def _validate_config(self, config: PipelineConfig) -> None:
        """
        Validate the loaded configuration.

        Args:
            config: Configuration to validate.

        Raises:
            ConfigValidationError: If validation fails.
        """
        errors = []

        # Validate scoring weights sum to 1.0
        screener_weights = config.screener.scoring
        screener_total = (screener_weights.short_interest_pct +
                         screener_weights.days_to_cover +
                         screener_weights.float_ratio +
                         screener_weights.volume_consistency)
        if abs(screener_total - 1.0) > 0.01:
            errors.append(f"Screener weights sum to {screener_total:.3f}, expected 1.0")

        deep_scan_weights = config.deep_scan.scoring
        deep_scan_total = (deep_scan_weights.volume_spike +
                          deep_scan_weights.sentiment_24h +
                          deep_scan_weights.call_put_ratio +
                          deep_scan_weights.borrow_fee)
        if abs(deep_scan_total - 1.0) > 0.01:
            errors.append(f"Deep scan weights sum to {deep_scan_total:.3f}, expected 1.0")

        # Validate positive values
        if config.screener.universe.min_market_cap <= 0:
            errors.append("Minimum market cap must be positive")

        if config.screener.universe.min_avg_volume <= 0:
            errors.append("Minimum average volume must be positive")

        if config.screener.filters.top_k_candidates <= 0:
            errors.append("Top K candidates must be positive")

        # Validate alert thresholds are in ascending order
        thresholds = config.alerting.thresholds
        if not (thresholds.low.squeeze_score <= thresholds.medium.squeeze_score <= thresholds.high.squeeze_score):
            errors.append("Alert thresholds must be in ascending order (low <= medium <= high)")

        # Validate API rate limits
        if config.performance.api_rate_limits.fmp_calls_per_minute > 300:
            errors.append("FMP rate limit exceeds API maximum of 300 calls/minute")

        if config.performance.api_rate_limits.finnhub_calls_per_minute > 60:
            errors.append("Finnhub rate limit exceeds API maximum of 60 calls/minute")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ConfigValidationError(error_msg)

    def get_config(self) -> PipelineConfig:
        """
        Get the loaded configuration.

        Returns:
            The loaded PipelineConfig instance.

        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config

    def get_screener_config(self) -> ScreenerConfig:
        """Get screener configuration."""
        return self.get_config().screener

    def get_deep_scan_config(self) -> DeepScanConfig:
        """Get deep scan configuration."""
        return self.get_config().deep_scan

    def get_alert_config(self) -> AlertConfig:
        """Get alert configuration."""
        return self.get_config().alerting

    def get_scheduling_config(self) -> SchedulingConfig:
        """Get scheduling configuration."""
        return self.get_config().scheduling

    def export_config_to_dict(self) -> Dict[str, Any]:
        """
        Export current configuration to dictionary format.

        Returns:
            Configuration as dictionary.
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return asdict(self._config)

    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to save the configuration file.
        """
        config_dict = self.export_config_to_dict()

        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)

        _logger.info("Configuration saved to %s", output_path)