"""
IBKR Flex Query XML position loader.

Parses Open Positions Flex Query exports and converts them to
RawIbkrPosition objects compatible with the position aggregator.

Positions from multiple accounts are merged by symbol using a weighted
average cost basis derived from ``costBasisMoney`` and ``position``.

Symbol disambiguation
----------------------
IBKR tickers are not globally unique — e.g. ``GOLD`` is both a London-listed
gold ETC (``listingExchange="LSEETF"``, price ~$4) and Barrick Gold Corp on
NYSE (price ~$40). Left unresolved, a bare ``GOLD`` handed to the data
provider (Yahoo/Tiingo/FMP) resolves to whichever instrument that provider
considers canonical for the symbol — not necessarily the one IBKR meant —
producing a silently wrong price and a false PnL alert.

Resolution requires the "Listing Exchange" column to be enabled on the
"Open Positions" Flex Query template (IBKR Client Portal > Performance &
Reports > Flex Queries > edit > Open Positions section > check "Listing
Exchange" and "Currency"). Without it, ``listingExchange`` is empty for
every position and this module falls back to the bare symbol — the same
behavior as before this was added.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from src.notification.logger import setup_logger
from src.portfolio.pnl_alert.position_aggregator import RawIbkrPosition

_logger = setup_logger(__name__)

# Verified one-off overrides: IBKR ticker -> provider ticker. Takes priority
# over the automatic listing-exchange resolution below. Use this when the
# listing-exchange suffix alone is not enough to pick the right instrument.
_IBKR_SYMBOL_MAP: dict[str, str] = {
    "VUSD": "VUSD.L",  # Vanguard FTSE All-World UCITS ETF (LSE, USD share class)
}

# IBKR `listingExchange` codes (as reported in the Flex Query XML) mapped to
# the exchange suffix Yahoo/Tiingo/FMP expect. Extend this as new non-US
# positions show up in the account.
_LISTING_EXCHANGE_SUFFIX_MAP: dict[str, str] = {
    "LSE": ".L",  # London Stock Exchange, main market
    "LSEETF": ".L",  # London Stock Exchange, ETF/ETC segment
    "LSEIOB1": ".L",  # London Stock Exchange, International Order Book
}

# IBKR listing exchanges known to need no suffix at all. Anything seen that is
# neither in this set nor in `_LISTING_EXCHANGE_SUFFIX_MAP` is logged as a
# WARNING and left unresolved rather than silently guessed — better to drop a
# price than to price against the wrong instrument.
_US_LISTING_EXCHANGES: frozenset[str] = frozenset(
    {
        "NYSE",
        "NASDAQ",
        "ARCA",
        "BATS",
        "AMEX",
        "IEX",
        "EDGX",
        "EDGA",
        "PSX",
        "BEX",
        "PINK",  # OTC Markets "Pink" tier (US OTC, e.g. micro-cap ADRs/penny stocks)
        "OTCBB",
        "OTCQB",
        "OTCQX",
        "GREY",
    }
)


def _resolve_provider_symbol(symbol: str, listing_exchange: str) -> str:
    """
    Translate an IBKR symbol into the ticker the data providers expect.

    Resolution order:
        1. `_IBKR_SYMBOL_MAP` - verified one-off overrides.
        2. `_LISTING_EXCHANGE_SUFFIX_MAP` - automatic suffix derived from the
           position's `listingExchange` (requires that Flex Query column to
           be enabled; see the module docstring).
        3. Unchanged - either a recognized US listing, or a non-US listing
           this map doesn't know about yet (logged as a WARNING).

    Args:
        symbol: Upper-cased IBKR symbol.
        listing_exchange: Upper-cased `listingExchange` attribute, or `""` if
            the XML export doesn't carry that column.

    Returns:
        The symbol to use for price lookups.
    """
    if symbol in _IBKR_SYMBOL_MAP:
        return _IBKR_SYMBOL_MAP[symbol]

    if not listing_exchange or listing_exchange in _US_LISTING_EXCHANGES:
        return symbol

    suffix = _LISTING_EXCHANGE_SUFFIX_MAP.get(listing_exchange)
    if suffix is None:
        _logger.warning(
            "Symbol %s has an unrecognized non-US listing exchange %s; using "
            "bare symbol for price lookup. This may collide with a same-named "
            "US ticker (e.g. GOLD/LSEETF vs GOLD/NYSE) - add a mapping to "
            "_LISTING_EXCHANGE_SUFFIX_MAP or _IBKR_SYMBOL_MAP in ibkr_xml_loader.py.",
            symbol,
            listing_exchange,
        )
        return symbol

    resolved = f"{symbol}{suffix}"
    _logger.info("Resolved %s (listingExchange=%s) -> %s", symbol, listing_exchange, resolved)
    return resolved


def resolve_xml_path(path_or_glob: str) -> Path:
    """
    Resolve a path string to a single XML file.

    When the path contains ``*``, all matches are collected and the
    lexicographically last one is returned — ISO-date filenames
    (``Open_Positions-2026-06-10.xml``) sort correctly this way.

    Args:
        path_or_glob: Exact file path or a glob such as
            ``data/portfolio/pnl_alert/Open_Positions-*.xml``.

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
        listing_exchange = pos_el.get("listingExchange", "").strip().upper()
        symbol_raw = _resolve_provider_symbol(symbol_raw, listing_exchange)

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
