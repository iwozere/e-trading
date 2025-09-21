"""
Re-Arm Alert System

Enhanced alert system with crossing detection and automatic re-arming
to prevent notification spam while maintaining useful alert behavior.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class ReArmConfig:
    """Configuration for re-arm alert behavior."""
    enabled: bool = True
    hysteresis: float = 0.25
    hysteresis_type: str = "percentage"  # "fixed", "percentage", "atr"
    cooldown_minutes: int = 15
    persistence_bars: int = 1
    close_only: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReArmConfig':
        """Create ReArmConfig from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class NotificationConfig:
    """Configuration for alert notifications."""
    channels: List[str] = None
    template: str = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = ["telegram"]
        if self.template is None:
            self.template = "{ticker} crossed {direction} {threshold} at {current_price}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NotificationConfig':
        """Create NotificationConfig from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EnhancedAlertConfig:
    """Complete enhanced alert configuration."""
    alert_type: str
    ticker: str
    threshold: float
    direction: str  # "above" or "below"
    re_arm_config: ReArmConfig
    notification_config: NotificationConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedAlertConfig':
        """Create EnhancedAlertConfig from dictionary."""
        re_arm_data = data.get("re_arm_config", {})
        notification_data = data.get("notification_config", {})

        return cls(
            alert_type=data.get("alert_type", "price"),
            ticker=data["ticker"],
            threshold=float(data["threshold"]),
            direction=data["direction"],
            re_arm_config=ReArmConfig.from_dict(re_arm_data),
            notification_config=NotificationConfig.from_dict(notification_data)
        )

    @classmethod
    def from_simple_params(cls, ticker: str, threshold: float, direction: str,
                          email: bool = False, rearm_enabled: bool = True) -> 'EnhancedAlertConfig':
        """Create config from simple parameters (backward compatibility)."""
        channels = ["telegram"]
        if email:
            channels.append("email")

        return cls(
            alert_type="price",
            ticker=ticker.upper(),
            threshold=threshold,
            direction=direction.lower(),
            re_arm_config=ReArmConfig(enabled=rearm_enabled),
            notification_config=NotificationConfig(channels=channels)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_type": self.alert_type,
            "ticker": self.ticker,
            "threshold": self.threshold,
            "direction": self.direction,
            "re_arm_config": self.re_arm_config.to_dict(),
            "notification_config": self.notification_config.to_dict()
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class ReArmAlertEvaluator:
    """Evaluates re-arm alerts with crossing detection and state management."""

    def __init__(self):
        self._logger = _logger

    def evaluate_alert(self, alert_data: Dict[str, Any], current_price: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate alert with re-arm logic.

        Args:
            alert_data: Alert data from database
            current_price: Current market price

        Returns:
            Tuple of (should_trigger, evaluation_details)
        """
        try:
            # Parse alert configuration
            config = self._parse_alert_config(alert_data)
            if not config:
                return False, {"error": "Invalid alert configuration"}

            # Get alert state
            is_armed = alert_data.get("is_armed", True)
            last_price = alert_data.get("last_price")
            last_triggered_at = alert_data.get("last_triggered_at")

            # Check cooldown
            if not self._check_cooldown(last_triggered_at, config.re_arm_config.cooldown_minutes):
                return False, {
                    "reason": "cooldown_active",
                    "current_price": current_price,
                    "last_triggered_at": last_triggered_at
                }

            # Handle re-arming logic
            if not is_armed:
                should_rearm = self._should_rearm(config, current_price)
                if should_rearm:
                    return False, {
                        "reason": "rearmed",
                        "current_price": current_price,
                        "rearm_level": self._calculate_rearm_level(config),
                        "action_required": "update_armed_state"
                    }
                else:
                    return False, {
                        "reason": "waiting_for_rearm",
                        "current_price": current_price,
                        "rearm_level": self._calculate_rearm_level(config)
                    }

            # Check if alert should trigger
            should_trigger = self._should_trigger(config, last_price, current_price)

            evaluation_details = {
                "current_price": current_price,
                "last_price": last_price,
                "threshold": config.threshold,
                "direction": config.direction,
                "rearm_level": self._calculate_rearm_level(config),
                "config": config.to_dict()
            }

            if should_trigger:
                evaluation_details.update({
                    "reason": "threshold_crossed",
                    "action_required": "trigger_and_disarm"
                })

            return should_trigger, evaluation_details

        except Exception as e:
            self._logger.exception("Error evaluating re-arm alert: ")
            return False, {"error": str(e)}

    def _parse_alert_config(self, alert_data: Dict[str, Any]) -> Optional[EnhancedAlertConfig]:
        """Parse alert configuration from database data."""
        try:
            # Check if we have enhanced config in re_arm_config field
            re_arm_config_json = alert_data.get("re_arm_config")
            if re_arm_config_json:
                try:
                    config_dict = json.loads(re_arm_config_json)
                    return EnhancedAlertConfig.from_dict(config_dict)
                except json.JSONDecodeError:
                    self._logger.warning("Invalid JSON in re_arm_config, using legacy format")

            # Fallback to legacy format
            ticker = alert_data.get("ticker")
            price = alert_data.get("price")
            condition = alert_data.get("condition")
            email = alert_data.get("email", False)

            if not all([ticker, price, condition]):
                return None

            return EnhancedAlertConfig.from_simple_params(
                ticker=ticker,
                threshold=float(price),
                direction=condition,
                email=email,
                rearm_enabled=True  # Default to re-arm enabled
            )

        except Exception as e:
            self._logger.exception("Error parsing alert config: ")
            return None

    def _check_cooldown(self, last_triggered_at: Optional[str], cooldown_minutes: int) -> bool:
        """Check if cooldown period has passed."""
        if not last_triggered_at or cooldown_minutes <= 0:
            return True

        try:
            if isinstance(last_triggered_at, str):
                last_triggered = datetime.fromisoformat(last_triggered_at.replace('Z', '+00:00'))
            else:
                last_triggered = last_triggered_at

            cooldown_period = timedelta(minutes=cooldown_minutes)
            return datetime.now() - last_triggered >= cooldown_period

        except Exception as e:
            self._logger.warning("Error checking cooldown: %s", e)
            return True  # Allow trigger if we can't parse timestamp

    def _should_rearm(self, config: EnhancedAlertConfig, current_price: float) -> bool:
        """Check if alert should be re-armed."""
        if not config.re_arm_config.enabled:
            return False

        rearm_level = self._calculate_rearm_level(config)

        if config.direction == "above":
            return current_price <= rearm_level
        else:  # "below"
            return current_price >= rearm_level

    def _should_trigger(self, config: EnhancedAlertConfig, last_price: Optional[float],
                       current_price: float) -> bool:
        """Check if alert should trigger based on crossing logic."""
        if last_price is None:
            # First evaluation - don't trigger, just store price
            return False

        threshold = config.threshold

        if config.direction == "above":
            # Trigger when price crosses from <= threshold to > threshold
            return last_price <= threshold and current_price > threshold
        else:  # "below"
            # Trigger when price crosses from >= threshold to < threshold
            return last_price >= threshold and current_price < threshold

    def _calculate_rearm_level(self, config: EnhancedAlertConfig) -> float:
        """Calculate the re-arm level based on hysteresis configuration."""
        threshold = config.threshold
        hysteresis = config.re_arm_config.hysteresis
        hysteresis_type = config.re_arm_config.hysteresis_type

        if hysteresis_type == "percentage":
            hysteresis_amount = threshold * (hysteresis / 100.0)
        elif hysteresis_type == "atr":
            # TODO: Implement ATR-based hysteresis
            # For now, fallback to percentage
            hysteresis_amount = threshold * 0.005  # 0.5% default
        else:  # "fixed"
            hysteresis_amount = hysteresis

        if config.direction == "above":
            return threshold - hysteresis_amount
        else:  # "below"
            return threshold + hysteresis_amount

    def format_notification_message(self, config: EnhancedAlertConfig,
                                  evaluation_details: Dict[str, Any]) -> str:
        """Format notification message using template."""
        try:
            template = config.notification_config.template
            current_price = evaluation_details.get("current_price", 0)
            rearm_level = evaluation_details.get("rearm_level", 0)

            message = template.format(
                ticker=config.ticker,
                threshold=config.threshold,
                direction=config.direction,
                current_price=f"{current_price:.2f}",
                rearm_level=f"{rearm_level:.2f}"
            )

            # Add re-arm information
            if config.re_arm_config.enabled:
                rearm_direction = "below" if config.direction == "above" else "above"
                message += f" Alert will re-arm {rearm_direction} {rearm_level:.2f}."

            return message

        except Exception as e:
            self._logger.warning("Error formatting message, using default: %s", e)
            return (f"{config.ticker} crossed {config.direction} "
                   f"{config.threshold} at {evaluation_details.get('current_price', 0):.2f}")

    def update_alert_state(self, alert_id: int, evaluation_details: Dict[str, Any],
                          current_price: float) -> Dict[str, Any]:
        """
        Generate database update parameters based on evaluation results.

        Returns:
            Dictionary of fields to update in database
        """
        updates = {
            "last_price": current_price
        }

        action_required = evaluation_details.get("action_required")

        if action_required == "trigger_and_disarm":
            updates.update({
                "is_armed": False,
                "last_triggered_at": datetime.now().isoformat()
            })
        elif action_required == "update_armed_state":
            updates["is_armed"] = True

        return updates


def create_default_rearm_config() -> ReArmConfig:
    """Create default re-arm configuration."""
    return ReArmConfig(
        enabled=True,
        hysteresis=0.25,
        hysteresis_type="percentage",
        cooldown_minutes=15,
        persistence_bars=1,
        close_only=False
    )


def migrate_legacy_alert_to_rearm(alert_data: Dict[str, Any]) -> str:
    """
    Convert legacy alert to enhanced re-arm format.

    Returns:
        JSON string for re_arm_config field
    """
    try:
        ticker = alert_data.get("ticker", "")
        price = float(alert_data.get("price", 0))
        condition = alert_data.get("condition", "above")
        email = alert_data.get("email", False)

        config = EnhancedAlertConfig.from_simple_params(
            ticker=ticker,
            threshold=price,
            direction=condition,
            email=email,
            rearm_enabled=True
        )

        return config.to_json()

    except Exception as e:
        _logger.exception("Error migrating legacy alert: ")
        # Return minimal config
        return json.dumps({
            "alert_type": "price",
            "ticker": alert_data.get("ticker", ""),
            "threshold": float(alert_data.get("price", 0)),
            "direction": alert_data.get("condition", "above"),
            "re_arm_config": create_default_rearm_config().to_dict(),
            "notification_config": {"channels": ["telegram"]}
        })