"""
IBKR Flex Query XML position loader.

Parses Open Positions Flex Query exports and converts them to
RawIbkrPosition objects compatible with the position aggregator.

Positions from multiple accounts are merged by symbol using a weighted
average cost basis derived from ``costBasisMoney`` and ``position``.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from src.notification.logger import setup_logger
from src.portfolio.pnl_alert.position_aggregator import RawIbkrPosition

_logger = setup_logger(__name__)

# IBKR reports non-US ETFs without the exchange suffix that data providers
# (Tiingo, FMP, Yahoo) require. Map IBKR ticker → provider ticker here.
_IBKR_SYMBOL_MAP: dict[str, str] = {
    "VUSD": "VUSD.L",  # Vanguard FTSE All-World UCITS ETF (LSE, USD share class)
}


def resolve_xml_path(path_or_glob: str) -> Path:
    """
    Resolve a path string to a single XML file.

    When the path contains ``*``, all matches are collected and the
    lexicographically last one is returned — ISO-date filenames
    (``Open_Positions-2026-06-10.xml``) sort correctly this way.

    Args:
        path_or_glob: Exact file path or a glob such as
            ``src/portfolio/pnl_alert/config/Open_Positions-*.xml``.

    Returns:
        Resolved :class:`pathlib.Path` to the XML file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    if "*" not in path_or_glob:
        p = Path(path_or_glob)
        if not p.exists():
            raise FileNotFoundError(f"IBKR XML file not found: {p}")
        return p

    path_obj = Path(path_or_glob)
    matches = sorted(path_obj.parent.glob(path_obj.name))
    if not matches:
        raise FileNotFoundError(f"No IBKR XML files matched: {path_or_glob}")

    latest = matches[-1]
    _logger.info(
        "Resolved XML glob '%s' → %s (%d candidate(s))",
        path_or_glob,
        latest.name,
        len(matches),
    )
    return latest


def load_ibkr_xml(path: str) -> List[RawIbkrPosition]:
    """
    Load positions from an IBKR Flex Query Open Positions XML export.

    Positions spread across multiple ``<FlexStatement>`` blocks (accounts)
    are aggregated per symbol: quantities are summed and the weighted average
    cost basis is computed from ``costBasisMoney / position``.

    Positions with ``position <= 0`` are silently skipped.

    Args:
        path: Exact file path **or** a glob pattern
            (e.g. ``config/Open_Positions-*.xml``).  When a glob is given,
            the lexicographically-last match is used.

    Returns:
        Sorted list of :class:`~position_aggregator.RawIbkrPosition` objects,
        one per unique symbol.

    Raises:
        FileNotFoundError: If no matching file is found.
        ValueError: If the XML cannot be parsed.
    """
    xml_path = resolve_xml_path(path)

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse IBKR XML {xml_path}: {exc}") from exc

    root = tree.getroot()

    # symbol → (total_cost_basis_money, total_quantity)
    accumulated: dict[str, tuple[float, float]] = {}
    raw_count = 0

    for pos_el in root.iter("OpenPosition"):
        symbol_raw = pos_el.get("symbol", "").strip().upper()
        if not symbol_raw:
            continue
        symbol_raw = _IBKR_SYMBOL_MAP.get(symbol_raw, symbol_raw)

        try:
            quantity = float(pos_el.get("position", "0"))
            cost_basis_money = float(pos_el.get("costBasisMoney", "0"))
        except (ValueError, TypeError):
            _logger.warning("Skipping malformed OpenPosition element for %s", symbol_raw)
            continue

        if quantity <= 0:
            _logger.debug("Skipping non-positive position for %s (qty=%s)", symbol_raw, quantity)
            continue

        raw_count += 1

        if symbol_raw in accumulated:
            prev_cost, prev_qty = accumulated[symbol_raw]
            accumulated[symbol_raw] = (prev_cost + cost_basis_money, prev_qty + quantity)
        else:
            accumulated[symbol_raw] = (cost_basis_money, quantity)

    _logger.info("Parsed %d raw position records from %s", raw_count, xml_path.name)

    positions: List[RawIbkrPosition] = []
    for symbol, (total_cost, total_qty) in sorted(accumulated.items()):
        if total_qty <= 0:
            continue

        avg_price = total_cost / total_qty
        if avg_price <= 0:
            _logger.warning("Skipping %s: computed avg_price is non-positive (%.6f)", symbol, avg_price)
            continue

        positions.append(
            RawIbkrPosition(
                symbol=symbol,
                avg_price=avg_price,
                quantity=total_qty,
                sec_type="STK",
            )
        )

    _logger.info("Loaded %d merged positions from %s", len(positions), xml_path.name)
    return positions
