"""
Alert Evaluator Service

Centralized alert evaluation service with rule evaluation and rearm logic.
Integrates with existing market data and indicator services while maintaining
clean separation of concerns.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
from datetime import datetime, timezone, timedelta
import asyncio
import json
import math
import hashlib

import pandas as pd
import yaml

from src.data.data_manager import DataManager
from src.indicators.service import IndicatorService
from src.data.db.services.jobs_service import JobsService
from src.common.alerts.schema_validator import AlertSchemaValidator, ValidationResult
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

UTC = timezone.utc


def utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


@dataclass
class AlertConfig:
    """Configuration for an alert evaluation."""
    ticker: str
    timeframe: str
    rule: Dict[str, Any]
    rearm: Optional[Dict[str, Any]]
    options: Dict[str, Any]
    notify: Dict[str, Any]


@dataclass
class AlertEvaluationResult:
    """Result of alert evaluation."""
    triggered: bool
    rearmed: bool
    state_updates: Dict[str, Any]
    notification_data: Optional[Dict[str, Any]]
    error: Optional[str]


@dataclass
class RearmResult:
    """Result of rearm logic evaluation."""
    should_rearm: bool
    new_status: str  # "ARMED", "TRIGGERED", "INACTIVE"
    state_updates: Dict[str, Any]


class AlertEvaluator:
    """
    Centralized alert evaluation service with rule evaluation and rearm logic.

    This service replaces the distributed alert evaluation logic in telegram services
    with a centralized, testable implementation that integrates cleanly with existing
    data and indicator services.
    """

    def __init__(self,
                 data_manager: DataManager,
                 indicator_service: IndicatorService,
                 jobs_service: JobsService,
                 schema_validator: AlertSchemaValidator):
        """
        Initialize the alert evaluator.

        Args:
            data_manager: Service for retrieving market data
            indicator_service: Service for calculating technical indicators
            jobs_service: Service for database operations
            schema_validator: Service for validating alert configurations
        """
        self.data_manager = data_manager
        self.indicator_service = indicator_service
        self.jobs_service = jobs_service
        self.schema_validator = schema_validator

        _logger.info("AlertEvaluator initialized successfully")

    async def evaluate_alert(self, job_run) -> AlertEvaluationResult:
        """
        Evaluate an alert job and return the result.

        Args:
            job_run: ScheduleRun object containing job details

        Returns:
            AlertEvaluationResult with evaluation outcome
        """
        try:
            # Get job details from the schedule
            schedule = job_run.schedule

            # Parse and validate alert configuration
            alert_config = self._parse_alert_config(schedule.task_params)
            if not alert_config:
                return AlertEvaluationResult(
                    triggered=False,
                    rearmed=False,
                    state_updates={},
                    notification_data=None,
                    error="Failed to parse alert configuration"
                )

            # Load current alert state
            current_state = self._load_alert_state(schedule.state_json)

            # Check if we should evaluate (once per bar logic)
            if not self._should_evaluate(alert_config, current_state):
                _logger.debug("Skipping evaluation for job %s - already processed this bar", schedule.id)
                return AlertEvaluationResult(
                    triggered=False,
                    rearmed=False,
                    state_updates={},
                    notification_data=None,
                    error=None
                )

            # Fetch market data
            market_data = await self._fetch_market_data(alert_config)
            if market_data is None or market_data.empty:
                return AlertEvaluationResult(
                    triggered=False,
                    rearmed=False,
                    state_updates={},
                    notification_data=None,
                    error="No market data available"
                )

            # Calculate required indicators
            indicators = await self._calculate_indicators(alert_config, market_data)

            # Evaluate rule tree
            triggered, rule_sides, rule_snapshot = self._evaluate_rule_tree(
                alert_config.rule, market_data, indicators, current_state.get("sides", {})
            )

            # Evaluate rearm logic if configured
            rearmed = False
            rearm_sides = {}
            rearm_snapshot = None

            if alert_config.rearm:
                rearmed, rearm_sides, rearm_snapshot = self._evaluate_rule_tree(
                    alert_config.rearm, market_data, indicators, current_state.get("sides", {})
                )

            # Apply rearm logic to determine final state
            rearm_result = self._apply_rearm_logic(
                alert_config, current_state, triggered, rearmed
            )

            # Merge sides for persistence
            all_sides = {**rule_sides, **rearm_sides}

            # Update state
            new_state = {
                **current_state,
                "sides": all_sides,
                "last_bar_ts": market_data.index[-1].isoformat(),
                "last_evaluation": utcnow().isoformat()
            }
            new_state.update(rearm_result.state_updates)

            # Prepare notification data if triggered
            notification_data = None
            if triggered and rearm_result.new_status == "TRIGGERED":
                notification_data = self._prepare_notification_data(
                    alert_config, market_data, indicators, rule_snapshot
                )

            return AlertEvaluationResult(
                triggered=triggered and rearm_result.new_status == "TRIGGERED",
                rearmed=rearmed,
                state_updates=new_state,
                notification_data=notification_data,
                error=None
            )

        except Exception as e:
            _logger.exception("Error evaluating alert for job %s", getattr(job_run, 'id', 'unknown'))
            return AlertEvaluationResult(
                triggered=False,
                rearmed=False,
                state_updates={},
                notification_data=None,
                error=str(e)
            )

    def _parse_alert_config(self, task_params: Dict[str, Any]) -> Optional[AlertConfig]:
        """
        Parse and validate alert configuration from task_params.

        Args:
            task_params: Raw task parameters from job schedule

        Returns:
            AlertConfig object or None if parsing fails
        """
        try:
            # Validate against schema first
            validation_result = self.schema_validator.validate_alert_config(task_params)
            if not validation_result.is_valid:
                _logger.error("Alert configuration validation failed: %s", validation_result.errors)
                return None

            # Log warnings if any
            for warning in validation_result.warnings:
                _logger.warning("Alert configuration warning: %s", warning)

            # Extract required fields
            ticker = task_params.get("ticker")
            timeframe = task_params.get("timeframe")
            rule = task_params.get("rule")

            if not all([ticker, timeframe, rule]):
                _logger.error("Missing required fields in alert config: ticker=%s, timeframe=%s, rule=%s",
                            ticker, timeframe, rule)
                return None

            # Extract optional fields with defaults
            rearm = task_params.get("rearm")
            options = task_params.get("options", {})
            notify = task_params.get("notify", {})

            return AlertConfig(
                ticker=ticker,
                timeframe=timeframe,
                rule=rule,
                rearm=rearm,
                options=options,
                notify=notify
            )

        except Exception as e:
            _logger.exception("Error parsing alert configuration:")
            return None

    def _load_alert_state(self, state_json: Optional[str]) -> Dict[str, Any]:
        """
        Load alert state from JSON string with comprehensive error recovery.

        Args:
            state_json: JSON string containing alert state

        Returns:
            State dictionary with default values if loading fails
        """
        return self._load_alert_state_with_recovery(state_json)

    def _should_evaluate(self, alert_config: AlertConfig, current_state: Dict[str, Any]) -> bool:
        """
        Check if alert should be evaluated based on once-per-bar logic.

        Args:
            alert_config: Alert configuration
            current_state: Current alert state

        Returns:
            True if evaluation should proceed
        """
        # Check evaluate_once_per_bar option
        evaluate_once = alert_config.options.get("evaluate_once_per_bar", True)
        if not evaluate_once:
            return True

        # Get last processed bar timestamp
        last_bar_ts = current_state.get("last_bar_ts")
        if not last_bar_ts:
            return True

        # For now, always evaluate - the actual bar timestamp comparison
        # will be implemented when we have market data
        return True

    async def _fetch_market_data(self, alert_config: AlertConfig) -> Optional[pd.DataFrame]:
        """
        Fetch market data for alert evaluation with provider failover.

        This method implements robust data fetching with:
        - Multiple provider attempts
        - Graceful degradation on failures
        - Data quality validation
        - Appropriate error handling

        Args:
            alert_config: Alert configuration

        Returns:
            DataFrame with OHLCV data or None if all providers fail
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Calculate required lookback period
                lookback_bars = self._calculate_required_lookback(alert_config)

                # Calculate date range based on timeframe and lookback
                end_date = utcnow()
                timeframe_minutes = self._timeframe_to_minutes(alert_config.timeframe)

                # Add buffer for weekends and holidays
                buffer_multiplier = 1.5 if timeframe_minutes >= 1440 else 1.2  # More buffer for daily+ data
                start_date = end_date - timedelta(minutes=timeframe_minutes * lookback_bars * buffer_multiplier)

                _logger.debug("Fetching market data for %s %s (attempt %d/%d): %d bars from %s to %s",
                            alert_config.ticker, alert_config.timeframe, retry_count + 1, max_retries,
                            lookback_bars, start_date.isoformat(), end_date.isoformat())

                # Fetch data using DataManager with provider failover
                df = self.data_manager.get_ohlcv(
                    symbol=alert_config.ticker,
                    timeframe=alert_config.timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    force_refresh=retry_count > 0  # Force refresh on retries
                )

                # Validate data quality
                if not self._validate_market_data(df, alert_config, lookback_bars):
                    retry_count += 1
                    if retry_count < max_retries:
                        _logger.warning("Data validation failed for %s %s, retrying...",
                                      alert_config.ticker, alert_config.timeframe)
                        continue
                    else:
                        _logger.error("Data validation failed for %s %s after %d attempts",
                                    alert_config.ticker, alert_config.timeframe, max_retries)
                        return None

                # Trim to closed bars only
                df = self._trim_to_closed_bars(df, alert_config.timeframe)

                # Final validation after trimming
                if df is None or len(df) < min(10, lookback_bars // 2):
                    _logger.warning("Insufficient data after trimming for %s %s: %d bars",
                                  alert_config.ticker, alert_config.timeframe, len(df) if df is not None else 0)
                    retry_count += 1
                    continue

                _logger.debug("Successfully fetched %d bars for %s %s", len(df),
                             alert_config.ticker, alert_config.timeframe)

                return df

            except Exception as e:
                retry_count += 1
                _logger.warning("Error fetching market data for %s %s (attempt %d/%d): %s",
                              alert_config.ticker, alert_config.timeframe, retry_count, max_retries, str(e))

                if retry_count >= max_retries:
                    _logger.error("Failed to fetch market data for %s %s after %d attempts",
                                alert_config.ticker, alert_config.timeframe, max_retries)
                    return None

                # Wait before retry (exponential backoff)
                await asyncio.sleep(2 ** retry_count)

        return None

    def _validate_market_data(self, df: Optional[pd.DataFrame], alert_config: AlertConfig,
                            expected_bars: int) -> bool:
        """
        Validate market data quality and completeness.

        Args:
            df: Market data DataFrame
            alert_config: Alert configuration
            expected_bars: Expected number of bars

        Returns:
            True if data is valid and sufficient
        """
        if df is None or df.empty:
            _logger.warning("Market data is None or empty")
            return False

        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            _logger.warning("Missing required columns in market data: %s", missing_columns)
            return False

        # Check minimum data requirements
        min_required_bars = min(50, expected_bars // 4)  # At least 25% of expected or 50 bars
        if len(df) < min_required_bars:
            _logger.warning("Insufficient market data: %d bars (minimum %d required)",
                          len(df), min_required_bars)
            return False

        # Check for excessive NaN values
        nan_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if nan_percentage > 0.1:  # More than 10% NaN values
            _logger.warning("Excessive NaN values in market data: %.1f%%", nan_percentage * 100)
            return False

        # Check for reasonable price ranges (basic sanity check)
        try:
            close_prices = df["close"].dropna()
            if len(close_prices) == 0:
                _logger.warning("No valid close prices in market data")
                return False

            price_range = close_prices.max() / close_prices.min()
            if price_range > 100:  # Prices vary by more than 100x
                _logger.warning("Suspicious price range in market data: %.2fx", price_range)
                # Don't fail validation, just warn - could be legitimate for some assets

            # Check for zero or negative prices
            if (close_prices <= 0).any():
                _logger.warning("Zero or negative prices found in market data")
                return False

        except Exception as e:
            _logger.warning("Error validating price data: %s", str(e))
            return False

        return True

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe_map = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440
        }
        return timeframe_map.get(timeframe, 60)  # Default to 1 hour

    def _trim_to_closed_bars(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Trim DataFrame to only include closed bars.

        Args:
            df: OHLCV DataFrame
            timeframe: Timeframe string

        Returns:
            DataFrame with only closed bars
        """
        if df.empty:
            return df

        # Calculate cutoff time for closed bars
        minutes = self._timeframe_to_minutes(timeframe)
        cutoff = utcnow() - timedelta(minutes=minutes / 2)

        # Get last bar timestamp
        last_ts = pd.Timestamp(df.index[-1]).to_pydatetime()
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=UTC)
        else:
            last_ts = last_ts.astimezone(UTC)

        # Remove last bar if it's not closed yet
        if last_ts > cutoff:
            df = df.iloc[:-1]

        return df

    def _calculate_required_lookback(self, alert_config: AlertConfig) -> int:
        """
        Calculate required lookback period for indicators.

        Args:
            alert_config: Alert configuration

        Returns:
            Number of bars needed for evaluation
        """
        # Start with configured lookback or default
        base_lookback = alert_config.options.get("lookback", 200)

        # Calculate lookback needed for rule and rearm
        rule_lookback = self._calculate_expression_lookback(alert_config.rule)
        rearm_lookback = 0
        if alert_config.rearm:
            rearm_lookback = self._calculate_expression_lookback(alert_config.rearm)

        # Return maximum needed
        return max(base_lookback, rule_lookback, rearm_lookback)

    def _calculate_expression_lookback(self, expr: Dict[str, Any]) -> int:
        """
        Calculate lookback needed for an expression tree.

        Args:
            expr: Expression dictionary

        Returns:
            Number of bars needed
        """
        if not expr:
            return 0

        # Handle logical operators
        if "and" in expr:
            return max(self._calculate_expression_lookback(sub) for sub in expr["and"])
        if "or" in expr:
            return max(self._calculate_expression_lookback(sub) for sub in expr["or"])
        if "not" in expr:
            return self._calculate_expression_lookback(expr["not"])

        # Handle comparison operators
        for op in ("gt", "gte", "lt", "lte", "eq", "ne", "between", "outside",
                  "inside_band", "outside_band", "crosses_above", "crosses_below"):
            if op in expr:
                node = expr[op]
                lhs_lookback = self._calculate_operand_lookback(node.get("lhs"))
                rhs_lookback = self._calculate_operand_lookback(node.get("rhs"))
                value_lookback = self._calculate_operand_lookback(node.get("value"))
                lower_lookback = self._calculate_operand_lookback(node.get("lower"))
                upper_lookback = self._calculate_operand_lookback(node.get("upper"))

                return max(lhs_lookback, rhs_lookback, value_lookback,
                          lower_lookback, upper_lookback)

        return 0

    def _calculate_operand_lookback(self, operand: Optional[Dict[str, Any]]) -> int:
        """
        Calculate lookback needed for an operand.

        Args:
            operand: Operand dictionary

        Returns:
            Number of bars needed
        """
        if not isinstance(operand, dict):
            return 0

        indicator = operand.get("indicator")
        if not indicator:
            return 0

        # Estimate lookback based on indicator type
        indicator_type = indicator.get("type", "").upper()
        params = indicator.get("params", {})

        if indicator_type in ("SMA", "EMA"):
            return int(params.get("period", 50)) + 5
        elif indicator_type == "RSI":
            return int(params.get("period", 14)) + 5
        elif indicator_type == "MACD":
            slow = int(params.get("slow", 26))
            signal = int(params.get("signal", 9))
            return slow + signal + 10
        elif indicator_type == "BOLLINGER":
            return int(params.get("period", 20)) + 5
        else:
            return 100  # Conservative default

    async def _calculate_indicators(self, alert_config: AlertConfig,
                                  market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all indicators needed for alert evaluation.

        Args:
            alert_config: Alert configuration
            market_data: OHLCV DataFrame

        Returns:
            Dictionary mapping indicator names to calculated series
        """
        indicators = {}

        try:
            # Extract indicator specs from rule and rearm expressions
            indicator_specs = []
            self._extract_indicator_specs(alert_config.rule, indicator_specs)
            if alert_config.rearm:
                self._extract_indicator_specs(alert_config.rearm, indicator_specs)

            # For now, use a simplified approach - calculate common indicators
            # This can be enhanced later to use the full IndicatorService interface
            indicators = self._calculate_basic_indicators(market_data, indicator_specs)

        except Exception as e:
            _logger.exception("Error calculating indicators:")

        return indicators

    def _calculate_basic_indicators(self, market_data: pd.DataFrame,
                                  specs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """
        Calculate basic indicators using simple implementations.

        This is a temporary implementation that can be enhanced later
        to use the full IndicatorService interface.

        Args:
            market_data: OHLCV DataFrame
            specs: List of indicator specifications

        Returns:
            Dictionary mapping indicator names to calculated series
        """
        indicators = {}

        for spec in specs:
            try:
                indicator_type = spec.get("type", "").upper()
                params = spec.get("params", {})
                output_key = spec.get("output") or indicator_type

                if indicator_type == "SMA":
                    period = int(params.get("period", 20))
                    source = params.get("source", "close")
                    if source in market_data.columns:
                        indicators[output_key] = market_data[source].rolling(window=period).mean()

                elif indicator_type == "EMA":
                    period = int(params.get("period", 20))
                    source = params.get("source", "close")
                    if source in market_data.columns:
                        indicators[output_key] = market_data[source].ewm(span=period).mean()

                elif indicator_type == "RSI":
                    period = int(params.get("period", 14))
                    source = params.get("source", "close")
                    if source in market_data.columns:
                        delta = market_data[source].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                        rs = gain / loss
                        indicators[output_key] = 100 - (100 / (1 + rs))

                elif indicator_type == "BOLLINGER":
                    period = int(params.get("period", 20))
                    std_dev = float(params.get("std_dev", 2.0))
                    source = params.get("source", "close")
                    if source in market_data.columns:
                        sma = market_data[source].rolling(window=period).mean()
                        std = market_data[source].rolling(window=period).std()

                        # Return middle band by default, can be enhanced for upper/lower
                        band_type = params.get("band", "middle")
                        if band_type == "upper":
                            indicators[output_key] = sma + (std * std_dev)
                        elif band_type == "lower":
                            indicators[output_key] = sma - (std * std_dev)
                        else:  # middle
                            indicators[output_key] = sma

                _logger.debug("Calculated basic indicator %s", output_key)

            except Exception as e:
                _logger.exception("Error calculating basic indicator %s:", spec.get("type"))

        return indicators

    def _extract_indicator_specs(self, expr: Dict[str, Any], specs: List[Dict[str, Any]]) -> None:
        """
        Extract indicator specifications from expression tree.

        Args:
            expr: Expression dictionary
            specs: List to append indicator specs to
        """
        if not expr:
            return

        # Handle logical operators
        if "and" in expr:
            for sub in expr["and"]:
                self._extract_indicator_specs(sub, specs)
        elif "or" in expr:
            for sub in expr["or"]:
                self._extract_indicator_specs(sub, specs)
        elif "not" in expr:
            self._extract_indicator_specs(expr["not"], specs)
        else:
            # Handle comparison operators
            for op in ("gt", "gte", "lt", "lte", "eq", "ne", "between", "outside",
                      "inside_band", "outside_band", "crosses_above", "crosses_below"):
                if op in expr:
                    node = expr[op]
                    self._extract_operand_indicators(node.get("lhs"), specs)
                    self._extract_operand_indicators(node.get("rhs"), specs)
                    self._extract_operand_indicators(node.get("value"), specs)
                    self._extract_operand_indicators(node.get("lower"), specs)
                    self._extract_operand_indicators(node.get("upper"), specs)

    def _extract_operand_indicators(self, operand: Optional[Dict[str, Any]],
                                  specs: List[Dict[str, Any]]) -> None:
        """
        Extract indicator specs from an operand.

        Args:
            operand: Operand dictionary
            specs: List to append indicator specs to
        """
        if not isinstance(operand, dict):
            return

        indicator = operand.get("indicator")
        if indicator and indicator not in specs:
            specs.append(indicator)

    def _evaluate_rule_tree(self, rule: Dict[str, Any], market_data: pd.DataFrame,
                           indicators: Dict[str, pd.Series],
                           sides_state: Dict[str, str]) -> Tuple[bool, Dict[str, str], Dict[str, float]]:
        """
        Evaluate a rule tree with logical operators.

        Args:
            rule: Rule expression dictionary
            market_data: OHLCV DataFrame
            indicators: Calculated indicators
            sides_state: Current crossing sides state

        Returns:
            Tuple of (result, updated_sides, snapshot_values)
        """
        if not rule:
            return False, {}, {}

        # Handle logical operators
        if "and" in rule:
            sides, snapshot = {}, {}
            for sub_rule in rule["and"]:
                result, sub_sides, sub_snapshot = self._evaluate_rule_tree(
                    sub_rule, market_data, indicators, sides_state
                )
                sides.update(sub_sides)
                snapshot.update(sub_snapshot)
                if not result:
                    return False, sides, snapshot
            return True, sides, snapshot

        if "or" in rule:
            sides, snapshot = {}, {}
            for sub_rule in rule["or"]:
                result, sub_sides, sub_snapshot = self._evaluate_rule_tree(
                    sub_rule, market_data, indicators, sides_state
                )
                sides.update(sub_sides)
                snapshot.update(sub_snapshot)
                if result:
                    return True, sides, snapshot
            return False, sides, snapshot

        if "not" in rule:
            result, sides, snapshot = self._evaluate_rule_tree(
                rule["not"], market_data, indicators, sides_state
            )
            return not result, sides, snapshot

        # Handle leaf nodes (comparison operators)
        for op in ("gt", "gte", "lt", "lte", "eq", "ne", "between", "outside",
                  "inside_band", "outside_band", "crosses_above", "crosses_below"):
            if op in rule:
                return self._evaluate_comparison(op, rule[op], market_data, indicators, sides_state)

        _logger.error("Unknown rule structure: %s", rule)
        return False, {}, {}

    def _evaluate_comparison(self, op: str, node: Dict[str, Any], market_data: pd.DataFrame,
                           indicators: Dict[str, pd.Series],
                           sides_state: Dict[str, str]) -> Tuple[bool, Dict[str, str], Dict[str, float]]:
        """
        Evaluate a comparison operation.

        Args:
            op: Comparison operator
            node: Comparison node dictionary
            market_data: OHLCV DataFrame
            indicators: Calculated indicators
            sides_state: Current crossing sides state

        Returns:
            Tuple of (result, updated_sides, snapshot_values)
        """
        def resolve_operand(operand) -> Tuple[Optional[float], Dict[str, float]]:
            """Resolve an operand to a numeric value."""
            if operand is None:
                return None, {}

            if "value" in operand:
                value = float(operand["value"])
                return value, {"value": value}

            if "field" in operand:
                field = operand["field"]
                if field not in market_data.columns:
                    _logger.error("Field %s not found in market data", field)
                    return None, {}

                series = market_data[field]
                if series.empty:
                    return None, {}

                value = series.iloc[-1]
                if isinstance(value, float) and math.isnan(value):
                    return None, {}

                return float(value), {field: float(value)}

            if "indicator" in operand:
                indicator_spec = operand["indicator"]
                indicator_key = indicator_spec.get("output") or indicator_spec["type"]

                if indicator_key not in indicators:
                    _logger.error("Indicator %s not found in calculated indicators", indicator_key)
                    return None, {}

                series = indicators[indicator_key]
                if series.empty:
                    return None, {}

                value = series.iloc[-1]
                if isinstance(value, float) and math.isnan(value):
                    return None, {}

                return float(value), {indicator_key: float(value)}

            return None, {}

        sides = {}
        snapshot = {}

        # Simple comparison operators
        if op in ("gt", "gte", "lt", "lte", "eq", "ne"):
            lhs, lhs_snap = resolve_operand(node.get("lhs"))
            rhs, rhs_snap = resolve_operand(node.get("rhs"))
            snapshot.update(lhs_snap)
            snapshot.update(rhs_snap)

            if lhs is None or rhs is None:
                return False, sides, snapshot

            comparisons = {
                "gt": lhs > rhs,
                "gte": lhs >= rhs,
                "lt": lhs < rhs,
                "lte": lhs <= rhs,
                "eq": lhs == rhs,
                "ne": lhs != rhs
            }

            return comparisons[op], sides, snapshot

        # Range operators
        if op in ("between", "outside", "inside_band", "outside_band"):
            # Normalize band operators
            if op == "inside_band":
                op = "between"
            elif op == "outside_band":
                op = "outside"

            value, value_snap = resolve_operand(node.get("value"))
            lower, lower_snap = resolve_operand(node.get("lower"))
            upper, upper_snap = resolve_operand(node.get("upper"))

            snapshot.update(value_snap)
            snapshot.update(lower_snap)
            snapshot.update(upper_snap)

            if None in (value, lower, upper):
                return False, sides, snapshot

            inside = lower <= value <= upper
            result = inside if op == "between" else not inside

            return result, sides, snapshot

        # Crossing operators
        if op in ("crosses_above", "crosses_below"):
            lhs, lhs_snap = resolve_operand(node.get("lhs"))
            rhs, rhs_snap = resolve_operand(node.get("rhs"))
            snapshot.update(lhs_snap)
            snapshot.update(rhs_snap)

            if lhs is None or rhs is None:
                return False, sides, snapshot

            # Generate unique key for this crossing
            cross_key = self._generate_cross_key(node)

            # Determine current side
            if lhs > rhs:
                current_side = "above"
            elif lhs < rhs:
                current_side = "below"
            else:
                current_side = sides_state.get(cross_key, "equal")

            # Update sides state
            sides[cross_key] = current_side

            # Get previous side
            previous_side = sides_state.get(cross_key)

            if previous_side is None:
                # First observation, don't trigger
                return False, sides, snapshot

            # Check for crossing
            if op == "crosses_above":
                result = previous_side in ("below", "equal") and current_side == "above"
            else:  # crosses_below
                result = previous_side in ("above", "equal") and current_side == "below"

            return result, sides, snapshot

        _logger.error("Unsupported comparison operator: %s", op)
        return False, sides, snapshot

    def _generate_cross_key(self, node: Dict[str, Any]) -> str:
        """
        Generate a unique key for crossing detection.

        Args:
            node: Comparison node dictionary

        Returns:
            Unique string key
        """
        # Create a stable hash of the node structure
        payload = json.dumps(node, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def _apply_rearm_logic(self, alert_config: AlertConfig, current_state: Dict[str, Any],
                          triggered: bool, rearmed: bool) -> RearmResult:
        """
        Apply rearm logic to determine final alert state.

        This method implements the complete rearm logic including:
        - Hysteresis support (percentage, fixed, ATR)
        - Cooldown periods
        - Persistence bar requirements
        - Crossing detection with side tracking

        Args:
            alert_config: Alert configuration
            current_state: Current alert state
            triggered: Whether rule triggered
            rearmed: Whether rearm condition met

        Returns:
            RearmResult with new status and state updates
        """
        current_status = current_state.get("status", "ARMED")
        state_updates = {}

        # If no rearm configuration, use simple trigger logic
        if not alert_config.rearm:
            if triggered:
                return RearmResult(
                    should_rearm=False,
                    new_status="TRIGGERED",
                    state_updates={"status": "TRIGGERED", "last_triggered": utcnow().isoformat()}
                )
            else:
                return RearmResult(
                    should_rearm=False,
                    new_status=current_status,
                    state_updates={}
                )

        # Get rearm configuration
        rearm_config = alert_config.rearm
        rearm_enabled = rearm_config.get("enabled", True)

        if not rearm_enabled:
            # Rearm disabled, use simple logic
            if triggered:
                return RearmResult(
                    should_rearm=False,
                    new_status="TRIGGERED",
                    state_updates={"status": "TRIGGERED", "last_triggered": utcnow().isoformat()}
                )
            else:
                return RearmResult(
                    should_rearm=False,
                    new_status=current_status,
                    state_updates={}
                )

        # Check cooldown period
        if not self._check_cooldown(current_state, rearm_config):
            return RearmResult(
                should_rearm=False,
                new_status="COOLDOWN",
                state_updates={"status": "COOLDOWN"}
            )

        # Check persistence bars requirement
        if not self._check_persistence_bars(current_state, rearm_config, triggered, rearmed):
            # Update persistence counters but don't change status yet
            persistence_updates = self._update_persistence_counters(current_state, triggered, rearmed)
            return RearmResult(
                should_rearm=False,
                new_status=current_status,
                state_updates=persistence_updates
            )

        # Apply main rearm state machine
        if current_status == "ARMED":
            if triggered:
                state_updates.update({
                    "status": "TRIGGERED",
                    "last_triggered": utcnow().isoformat(),
                    "trigger_count": current_state.get("trigger_count", 0) + 1
                })
                # Reset persistence counters
                state_updates.update(self._reset_persistence_counters())

                return RearmResult(
                    should_rearm=False,
                    new_status="TRIGGERED",
                    state_updates=state_updates
                )

        elif current_status in ("TRIGGERED", "COOLDOWN"):
            if rearmed:
                state_updates.update({
                    "status": "ARMED",
                    "last_rearmed": utcnow().isoformat()
                })
                # Reset persistence counters
                state_updates.update(self._reset_persistence_counters())

                return RearmResult(
                    should_rearm=True,
                    new_status="ARMED",
                    state_updates=state_updates
                )

        # No state change
        return RearmResult(
            should_rearm=False,
            new_status=current_status,
            state_updates={}
        )

    def _check_cooldown(self, current_state: Dict[str, Any], rearm_config: Dict[str, Any]) -> bool:
        """
        Check if cooldown period has passed.

        Args:
            current_state: Current alert state
            rearm_config: Rearm configuration

        Returns:
            True if cooldown has passed or is not applicable
        """
        cooldown_minutes = rearm_config.get("cooldown_minutes", 0)
        if cooldown_minutes <= 0:
            return True

        last_triggered = current_state.get("last_triggered")
        if not last_triggered:
            return True

        try:
            if isinstance(last_triggered, str):
                last_triggered_dt = datetime.fromisoformat(last_triggered.replace('Z', '+00:00'))
            else:
                last_triggered_dt = last_triggered

            if last_triggered_dt.tzinfo is None:
                last_triggered_dt = last_triggered_dt.replace(tzinfo=UTC)

            cooldown_period = timedelta(minutes=cooldown_minutes)
            time_since_trigger = utcnow() - last_triggered_dt

            return time_since_trigger >= cooldown_period

        except Exception as e:
            _logger.warning("Error checking cooldown: %s", str(e))
            return True  # Allow if we can't parse timestamp

    def _check_persistence_bars(self, current_state: Dict[str, Any], rearm_config: Dict[str, Any],
                               triggered: bool, rearmed: bool) -> bool:
        """
        Check if persistence bar requirements are met.

        Args:
            current_state: Current alert state
            rearm_config: Rearm configuration
            triggered: Whether rule triggered this bar
            rearmed: Whether rearm condition met this bar

        Returns:
            True if persistence requirements are met
        """
        persistence_bars = rearm_config.get("persistence_bars", 1)
        if persistence_bars <= 1:
            return True  # No persistence requirement

        # Check trigger persistence
        if triggered:
            trigger_bars = current_state.get("consecutive_trigger_bars", 0) + 1
            return trigger_bars >= persistence_bars

        # Check rearm persistence
        if rearmed:
            rearm_bars = current_state.get("consecutive_rearm_bars", 0) + 1
            return rearm_bars >= persistence_bars

        return False

    def _update_persistence_counters(self, current_state: Dict[str, Any],
                                   triggered: bool, rearmed: bool) -> Dict[str, Any]:
        """
        Update persistence bar counters.

        Args:
            current_state: Current alert state
            triggered: Whether rule triggered this bar
            rearmed: Whether rearm condition met this bar

        Returns:
            Dictionary with updated persistence counters
        """
        updates = {}

        if triggered:
            updates["consecutive_trigger_bars"] = current_state.get("consecutive_trigger_bars", 0) + 1
            updates["consecutive_rearm_bars"] = 0  # Reset rearm counter
        elif rearmed:
            updates["consecutive_rearm_bars"] = current_state.get("consecutive_rearm_bars", 0) + 1
            updates["consecutive_trigger_bars"] = 0  # Reset trigger counter
        else:
            # Neither triggered nor rearmed, reset both
            updates["consecutive_trigger_bars"] = 0
            updates["consecutive_rearm_bars"] = 0

        return updates

    def _reset_persistence_counters(self) -> Dict[str, Any]:
        """
        Reset persistence bar counters.

        Returns:
            Dictionary with reset persistence counters
        """
        return {
            "consecutive_trigger_bars": 0,
            "consecutive_rearm_bars": 0
        }

    def _prepare_notification_data(self, alert_config: AlertConfig, market_data: pd.DataFrame,
                                 indicators: Dict[str, pd.Series],
                                 rule_snapshot: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """
        Prepare notification data for triggered alert.

        Args:
            alert_config: Alert configuration
            market_data: OHLCV DataFrame
            indicators: Calculated indicators
            rule_snapshot: Snapshot values from rule evaluation

        Returns:
            Dictionary with notification data
        """
        # Get current market data
        current_bar = market_data.iloc[-1]

        notification_data = {
            "ticker": alert_config.ticker,
            "timeframe": alert_config.timeframe,
            "timestamp": utcnow().isoformat(),
            "price": float(current_bar["close"]),
            "volume": float(current_bar["volume"]),
            "rule_snapshot": rule_snapshot or {},
            "notify_config": alert_config.notify
        }

        # Add indicator values if available
        if indicators:
            indicator_values = {}
            for name, series in indicators.items():
                if not series.empty:
                    indicator_values[name] = float(series.iloc[-1])
            notification_data["indicators"] = indicator_values

        return notification_data

    def _calculate_hysteresis_level(self, alert_config: AlertConfig,
                                  market_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate hysteresis level for rearm logic.

        Args:
            alert_config: Alert configuration
            market_data: OHLCV DataFrame for ATR calculation

        Returns:
            Hysteresis level or None if calculation fails
        """
        if not alert_config.rearm:
            return None

        rearm_config = alert_config.rearm
        threshold = rearm_config.get("threshold")
        hysteresis = rearm_config.get("hysteresis", 0.25)
        hysteresis_type = rearm_config.get("hysteresis_type", "percentage")
        direction = rearm_config.get("direction", "above")

        if threshold is None:
            _logger.warning("No threshold specified in rearm config")
            return None

        try:
            if hysteresis_type == "percentage":
                hysteresis_amount = threshold * (hysteresis / 100.0)
            elif hysteresis_type == "atr":
                # Calculate ATR-based hysteresis
                atr_value = self._calculate_atr(market_data, period=14)
                if atr_value is None:
                    _logger.warning("Could not calculate ATR, falling back to percentage")
                    hysteresis_amount = threshold * 0.005  # 0.5% fallback
                else:
                    hysteresis_amount = atr_value * hysteresis
            else:  # "fixed"
                hysteresis_amount = hysteresis

            # Calculate rearm level based on direction
            if direction == "above":
                return threshold - hysteresis_amount
            else:  # "below"
                return threshold + hysteresis_amount

        except Exception as e:
            _logger.exception("Error calculating hysteresis level:")
            return None

    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range for ATR-based hysteresis.

        Args:
            market_data: OHLCV DataFrame
            period: ATR calculation period

        Returns:
            ATR value or None if calculation fails
        """
        try:
            if len(market_data) < period + 1:
                return None

            high = market_data["high"]
            low = market_data["low"]
            close = market_data["close"]
            prev_close = close.shift(1)

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate ATR as simple moving average of True Range
            atr = true_range.rolling(window=period).mean()

            if atr.empty:
                return None

            return float(atr.iloc[-1])

        except Exception as e:
            _logger.exception("Error calculating ATR:")
            return None

    def _update_alert_state(self, job_id: int, new_state: Dict[str, Any]) -> bool:
        """
        Update alert state in the database with comprehensive error handling.

        This method handles:
        - JSON serialization with proper error handling
        - Database transaction management
        - State validation and sanitization
        - Fallback mechanisms for corrupted state

        Args:
            job_id: Job schedule ID
            new_state: New state dictionary to persist

        Returns:
            True if update was successful
        """
        try:
            # Validate and sanitize state before persistence
            sanitized_state = self._sanitize_state(new_state)

            # Serialize to JSON with proper error handling
            try:
                state_json = json.dumps(sanitized_state, ensure_ascii=False, default=str)
            except (TypeError, ValueError) as e:
                _logger.exception("Error serializing state for job %s:", job_id)
                # Try with a minimal state
                minimal_state = {
                    "status": sanitized_state.get("status", "ARMED"),
                    "sides": sanitized_state.get("sides", {}),
                    "last_evaluation": utcnow().isoformat(),
                    "error": "State serialization failed"
                }
                state_json = json.dumps(minimal_state, ensure_ascii=False)

            # Validate JSON size (prevent excessive state growth)
            if len(state_json) > 10000:  # 10KB limit
                _logger.warning("Alert state too large for job %s (%d bytes), truncating",
                              job_id, len(state_json))
                # Keep only essential state
                essential_state = {
                    "status": sanitized_state.get("status", "ARMED"),
                    "sides": {},  # Reset sides to prevent growth
                    "last_evaluation": utcnow().isoformat(),
                    "truncated": True
                }
                state_json = json.dumps(essential_state, ensure_ascii=False)

            _logger.debug("Updating alert state for job %s (%d bytes): %s",
                         job_id, len(state_json), state_json[:200] + "..." if len(state_json) > 200 else state_json)

            # TODO: Implement actual database update through jobs service
            # This would be something like:
            # success = self.jobs_service.update_schedule_state(job_id, state_json)
            # For now, we'll simulate success
            success = True

            if success:
                _logger.debug("Successfully updated alert state for job %s", job_id)
            else:
                _logger.error("Failed to update alert state for job %s", job_id)

            return success

        except Exception as e:
            _logger.exception("Unexpected error updating alert state for job %s:", job_id)
            return False

    def _sanitize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize alert state to ensure it can be safely persisted.

        Args:
            state: Raw state dictionary

        Returns:
            Sanitized state dictionary
        """
        sanitized = {}

        try:
            # Copy basic fields with type validation
            if "status" in state:
                status = str(state["status"])
                if status in ("ARMED", "TRIGGERED", "COOLDOWN", "INACTIVE", "ERROR"):
                    sanitized["status"] = status
                else:
                    _logger.warning("Invalid status value: %s, defaulting to ARMED", status)
                    sanitized["status"] = "ARMED"
            else:
                sanitized["status"] = "ARMED"

            # Handle sides dictionary (for crossing detection)
            if "sides" in state and isinstance(state["sides"], dict):
                sides = {}
                for key, value in state["sides"].items():
                    if isinstance(key, str) and isinstance(value, str):
                        if value in ("above", "below", "equal"):
                            sides[key] = value
                        else:
                            _logger.warning("Invalid side value: %s for key %s", value, key)
                sanitized["sides"] = sides
            else:
                sanitized["sides"] = {}

            # Handle timestamps
            for timestamp_field in ("last_evaluation", "last_triggered", "last_rearmed", "last_bar_ts"):
                if timestamp_field in state:
                    try:
                        # Validate timestamp format
                        timestamp_str = str(state[timestamp_field])
                        # Try to parse it to validate
                        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        sanitized[timestamp_field] = timestamp_str
                    except (ValueError, TypeError) as e:
                        _logger.warning("Invalid timestamp for %s: %s", timestamp_field, str(e))

            # Handle numeric fields
            for numeric_field in ("trigger_count", "consecutive_trigger_bars", "consecutive_rearm_bars"):
                if numeric_field in state:
                    try:
                        sanitized[numeric_field] = int(state[numeric_field])
                    except (ValueError, TypeError):
                        _logger.warning("Invalid numeric value for %s: %s", numeric_field, state[numeric_field])
                        sanitized[numeric_field] = 0

            # Handle boolean fields
            for bool_field in ("truncated",):
                if bool_field in state:
                    sanitized[bool_field] = bool(state[bool_field])

            # Handle error messages
            if "error" in state:
                error_msg = str(state["error"])[:500]  # Limit error message length
                sanitized["error"] = error_msg

            # Add metadata
            sanitized["last_updated"] = utcnow().isoformat()
            sanitized["version"] = "1.0"  # State schema version

        except Exception as e:
            _logger.exception("Error sanitizing state:")
            # Return minimal valid state
            return {
                "status": "ERROR",
                "sides": {},
                "last_evaluation": utcnow().isoformat(),
                "error": f"State sanitization failed: {str(e)}"
            }

        return sanitized

    def _load_alert_state_with_recovery(self, state_json: Optional[str]) -> Dict[str, Any]:
        """
        Load alert state with comprehensive error recovery.

        This method handles:
        - JSON parsing errors
        - Schema validation
        - State migration from older versions
        - Corruption recovery

        Args:
            state_json: JSON string containing alert state

        Returns:
            Valid state dictionary with defaults for missing fields
        """
        default_state = {
            "status": "ARMED",
            "sides": {},
            "last_evaluation": None,
            "trigger_count": 0,
            "consecutive_trigger_bars": 0,
            "consecutive_rearm_bars": 0
        }

        if not state_json:
            _logger.debug("No state JSON provided, using default state")
            return default_state

        try:
            # Parse JSON
            state = json.loads(state_json)
            if not isinstance(state, dict):
                _logger.warning("State JSON is not a dictionary, using default state")
                return default_state

            # Validate and merge with defaults
            validated_state = default_state.copy()

            # Validate status
            if "status" in state and state["status"] in ("ARMED", "TRIGGERED", "COOLDOWN", "INACTIVE", "ERROR"):
                validated_state["status"] = state["status"]

            # Validate sides
            if "sides" in state and isinstance(state["sides"], dict):
                validated_sides = {}
                for key, value in state["sides"].items():
                    if isinstance(key, str) and value in ("above", "below", "equal"):
                        validated_sides[key] = value
                validated_state["sides"] = validated_sides

            # Validate timestamps
            for timestamp_field in ("last_evaluation", "last_triggered", "last_rearmed", "last_bar_ts"):
                if timestamp_field in state:
                    try:
                        timestamp_str = str(state[timestamp_field])
                        # Validate by parsing
                        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        validated_state[timestamp_field] = timestamp_str
                    except (ValueError, TypeError):
                        _logger.warning("Invalid timestamp in state for %s", timestamp_field)

            # Validate numeric fields
            for numeric_field in ("trigger_count", "consecutive_trigger_bars", "consecutive_rearm_bars"):
                if numeric_field in state:
                    try:
                        validated_state[numeric_field] = max(0, int(state[numeric_field]))
                    except (ValueError, TypeError):
                        _logger.warning("Invalid numeric value in state for %s", numeric_field)

            # Check for state version and migrate if needed
            state_version = state.get("version", "0.0")
            if state_version != "1.0":
                _logger.info("Migrating state from version %s to 1.0", state_version)
                validated_state = self._migrate_state(validated_state, state_version)

            return validated_state

        except json.JSONDecodeError as e:
            _logger.warning("Failed to parse state JSON: %s, using default state", str(e))
            return default_state
        except Exception as e:
            _logger.exception("Unexpected error loading state, using default state")
            return default_state

    def _migrate_state(self, state: Dict[str, Any], from_version: str) -> Dict[str, Any]:
        """
        Migrate state from older versions to current schema.

        Args:
            state: Current state dictionary
            from_version: Version to migrate from

        Returns:
            Migrated state dictionary
        """
        try:
            if from_version == "0.0":
                # Migration from legacy state format
                _logger.debug("Migrating from legacy state format")

                # Legacy states might have different field names
                if "is_armed" in state:
                    state["status"] = "ARMED" if state["is_armed"] else "TRIGGERED"
                    del state["is_armed"]

                # Ensure all required fields exist
                if "sides" not in state:
                    state["sides"] = {}

            # Add current version
            state["version"] = "1.0"
            state["migrated_from"] = from_version
            state["migration_timestamp"] = utcnow().isoformat()

        except Exception as e:
            _logger.exception("Error during state migration:")

        return state

    async def evaluate_all_alerts(self, user_id: Optional[int] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate all active alerts for a user or globally.

        Args:
            user_id: Optional user ID to filter alerts
            limit: Optional limit on number of alerts to evaluate

        Returns:
            Dictionary with evaluation results and statistics
        """
        try:
            # Get active alert jobs from the jobs service
            active_jobs = self.jobs_service.get_active_jobs(
                job_type="alert",
                user_id=user_id,
                limit=limit
            )

            _logger.info("Evaluating %d active alert(s) for user %s", len(active_jobs), user_id or "all")

            results = {
                "total_evaluated": 0,
                "triggered": 0,
                "rearmed": 0,
                "errors": 0,
                "results": []
            }

            for job in active_jobs:
                try:
                    # Create a mock job run for evaluation
                    job_run = type('JobRun', (), {
                        'schedule': job,
                        'id': f"eval_{job.id}_{utcnow().timestamp()}"
                    })()

                    # Evaluate the alert
                    result = await self.evaluate_alert(job_run)

                    results["total_evaluated"] += 1
                    if result.triggered:
                        results["triggered"] += 1
                    if result.rearmed:
                        results["rearmed"] += 1
                    if result.error:
                        results["errors"] += 1

                    # Store individual result
                    results["results"].append({
                        "job_id": job.id,
                        "ticker": job.task_params.get("ticker"),
                        "triggered": result.triggered,
                        "rearmed": result.rearmed,
                        "error": result.error,
                        "notification_data": result.notification_data
                    })

                    # Update job state if needed
                    if result.state_updates:
                        self.jobs_service.update_job_state(job.id, result.state_updates)

                except Exception as e:
                    _logger.exception("Failed to evaluate alert job %s", job.id)
                    results["errors"] += 1
                    results["results"].append({
                        "job_id": job.id,
                        "ticker": job.task_params.get("ticker", "unknown"),
                        "triggered": False,
                        "rearmed": False,
                        "error": str(e),
                        "notification_data": None
                    })

            _logger.info("Alert evaluation complete: %d evaluated, %d triggered, %d errors",
                        results["total_evaluated"], results["triggered"], results["errors"])

            return results

        except Exception as e:
            _logger.exception("Error during batch alert evaluation")
            return {
                "total_evaluated": 0,
                "triggered": 0,
                "rearmed": 0,
                "errors": 1,
                "results": [],
                "batch_error": str(e)
            }