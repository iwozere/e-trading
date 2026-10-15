"""
``<data_root>/probe-paid/probe_results.json``: what ``fmp probe-paid`` and ``fmp calibrate-usage`` measured, and
the settings derived from it (spec §4.4: ``fmp plan`` must produce ``budget_plan.md`` from the probe results).

``apply_probe_results`` overlays ``settings`` on the configuration loaded from ``config/fmp/fmp.yaml``: measured
values win over the file's assumptions. Set ``fmp.use_probe_results: false`` to ignore them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.data.fmp_pipeline.fsutil import atomic_write_bytes
from src.data.fmp_pipeline.planner.config import PlanConfig, merge_settings
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

PROBE_DIR = "probe-paid"
RESULTS_FILE = "probe_results.json"


def results_path(data_root: Path) -> Path:
    """``<data_root>/probe-paid/probe_results.json``."""
    return data_root / PROBE_DIR / RESULTS_FILE


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def read_results(data_root: Path) -> Optional[dict[str, Any]]:
    """Return the results file, or ``None`` when no paid probe has run."""
    path = results_path(data_root)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_results(data_root: Path, findings: Any, settings: dict[str, Any], calls_used: int) -> Path:
    """
    Write the probe's findings and derived settings, keeping an earlier calibration.

    Returns:
        The file path.
    """
    previous = read_results(data_root) or {}
    merged_settings = dict(settings)
    if "bandwidth" in previous.get("settings", {}):
        merged_settings["bandwidth"] = previous["settings"]["bandwidth"]
    payload = {"probed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "calls_used": calls_used,
               "settings": merged_settings, "findings": _plain(findings),
               "calibration": previous.get("calibration")}
    path = results_path(data_root)
    atomic_write_bytes(path, json.dumps(payload, indent=1, sort_keys=True).encode("utf-8"))
    return path


def write_calibration(data_root: Path, calibration: dict[str, Any], basis: Optional[str]) -> Path:
    """Record a ``calibrate-usage`` result; a decided ``basis`` becomes ``settings.bandwidth.basis``."""
    payload = read_results(data_root) or {"settings": {}}
    payload["calibration"] = _plain(calibration)
    if basis is not None:
        payload.setdefault("settings", {})["bandwidth"] = {"basis": basis}
    path = results_path(data_root)
    atomic_write_bytes(path, json.dumps(payload, indent=1, sort_keys=True).encode("utf-8"))
    return path


def apply_probe_results(config: PlanConfig, data_root: Path) -> PlanConfig:
    """
    Overlay the measured settings on ``config`` (unless ``config.use_probe_results`` is false).

    Raises:
        ValueError: The results file holds a setting the configuration does not know or with a wrong type.
    """
    if not config.use_probe_results:
        return config
    results = read_results(data_root)
    if not results or not results.get("settings"):
        return config
    updated = merge_settings(config, results["settings"], "probe_results.")
    _logger.info("Applied paid-probe settings from %s: %s", results_path(data_root),
                 json.dumps(results["settings"], sort_keys=True))
    return updated
