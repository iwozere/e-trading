"""Tests for built-in channel plugin discovery in src.notification.channels.loader."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.channels.loader import PluginLoader  # noqa: E402


def test_discover_builtin_plugins_skips_non_plugin_modules() -> None:
    """Helper modules (base, loader, config) are not offered as channel plugins."""
    plugins = PluginLoader().discover_builtin_plugins()

    for helper in ("base", "loader", "config"):
        assert f"src.notification.channels.{helper}" not in plugins


def test_discover_builtin_plugins_finds_real_channels() -> None:
    """The real channel modules are still discovered."""
    plugins = PluginLoader().discover_builtin_plugins()

    for channel in ("email_channel", "sms_channel", "telegram_channel"):
        assert f"src.notification.channels.{channel}" in plugins
