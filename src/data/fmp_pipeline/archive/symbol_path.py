"""
Windows-safe, reversible file names for symbols and endpoints in the raw archive (spec §6.6, plan WP-5).

Archive files are named after the request's symbol, so a ticker must become a legal file name on NTFS, on the NAS
and on the Pi's ext4, and two different tickers must never map to the same file on a case-insensitive file
system. ``encode_symbol`` does this in three steps:

1. Percent-encode (``%XX``, upper-case hex) every byte that is unsafe or ambiguous: ``< > : " / \\ | ? *``, the
   escape character ``%`` itself, ``~`` (reserved for step 3), control characters, DEL, non-ASCII (as UTF-8
   bytes), a leading ``.`` or ``_``, and leading or trailing spaces and dots.
2. Prefix reserved Windows device names (``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM0``–``COM9``, ``LPT0``–``LPT9``,
   ``CONIN$``, ``CONOUT$``; any case, also with an extension such as ``NUL.A``) with ``_``. Several of these are
   or were real tickers. Because a literal leading ``_`` is encoded in step 1, the prefix is unambiguous.
3. If the name contains a lower-case letter, append ``~`` plus the first 8 hex digits of the symbol's sha256.
   ``ABC`` and ``abc`` therefore land in ``ABC`` and ``abc~<hash>``. The rule depends only on the symbol, not on
   the rest of the universe, so a file name never changes when a new symbol appears (D32).

``decode_symbol`` reverses all three steps. ``None`` (endpoints without a symbol) is the file name ``_``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Optional

NO_SYMBOL = "_"
MAX_NAME_LEN = 150
CASE_SUFFIX_SEP = "~"
CASE_SUFFIX_LEN = 8

_ALWAYS_ENCODED = set('<>:"/\\|?*%~')
_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {f"{p}{i}" for p in ("COM", "LPT") for i in range(10)}
_CASE_SUFFIX_RE = re.compile(rf"{re.escape(CASE_SUFFIX_SEP)}[0-9a-f]{{{CASE_SUFFIX_LEN}}}$")
_ENDPOINT_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _pct(char: str) -> str:
    return "".join(f"%{b:02X}" for b in char.encode("utf-8"))


def _is_plain(char: str) -> bool:
    return 0x20 <= ord(char) < 0x7F and char not in _ALWAYS_ENCODED


def _percent_encode(symbol: str) -> str:
    chars = [c if _is_plain(c) else _pct(c) for c in symbol]
    # Leading "." (hidden file, "." and "..") and "_" (would be read as the reserved-name prefix).
    if symbol[0] in "._":
        chars[0] = _pct(symbol[0])
    # Leading and trailing spaces and dots: Windows strips trailing ones silently, leading spaces are error-prone.
    for i in range(len(symbol)):
        if symbol[i] != " ":
            break
        chars[i] = _pct(" ")
    for i in range(len(symbol) - 1, -1, -1):
        if symbol[i] not in " .":
            break
        chars[i] = _pct(symbol[i])
    return "".join(chars)


def _is_reserved(name: str) -> bool:
    return name.split(".", 1)[0].rstrip(" ").upper() in _RESERVED


def _case_suffix(symbol: str) -> str:
    return CASE_SUFFIX_SEP + hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:CASE_SUFFIX_LEN]


def encode_symbol(symbol: Optional[str]) -> str:
    """
    Map a symbol to a Windows-safe file name stem (no extension).

    Args:
        symbol: Ticker as sent to FMP, or ``None`` for endpoints without a symbol.

    Returns:
        The file name stem; ``"_"`` for ``None``.

    Raises:
        ValueError: If ``symbol`` is empty or its encoded name exceeds ``MAX_NAME_LEN`` characters.
    """
    if symbol is None:
        return NO_SYMBOL
    if not symbol:
        raise ValueError("symbol must be None or a non-empty string")
    name = _percent_encode(symbol)
    if _is_reserved(name):
        name = "_" + name
    if any("a" <= c <= "z" for c in name):
        name += _case_suffix(symbol)
    if len(name) > MAX_NAME_LEN:
        raise ValueError(f"encoded symbol is longer than {MAX_NAME_LEN} characters: {symbol!r}")
    return name


def _percent_decode(name: str) -> str:
    out = bytearray()
    i = 0
    while i < len(name):
        if name[i] == "%":
            hex_digits = name[i + 1:i + 3]
            if len(hex_digits) != 2 or not all(c in "0123456789ABCDEF" for c in hex_digits):
                raise ValueError(f"bad percent escape in {name!r}")
            out.append(int(hex_digits, 16))
            i += 3
        else:
            out.extend(name[i].encode("utf-8"))
            i += 1
    return out.decode("utf-8")


def decode_symbol(name: str) -> Optional[str]:
    """
    Reverse ``encode_symbol``.

    Args:
        name: File name stem produced by ``encode_symbol``.

    Returns:
        The original symbol, or ``None`` for ``"_"``.

    Raises:
        ValueError: If ``name`` is not exactly what ``encode_symbol`` produces for some symbol (bad escape, wrong
            case suffix, a needless escape or prefix).
    """
    if name == NO_SYMBOL:
        return None
    stem = _CASE_SUFFIX_RE.sub("", name)
    symbol = _percent_decode(stem[1:] if stem.startswith("_") else stem)
    if not symbol or encode_symbol(symbol) != name:
        raise ValueError(f"{name!r} is not a canonical symbol encoding")
    return symbol


def endpoint_dir(endpoint: str) -> str:
    """
    Validate an FMP endpoint path for use as a relative directory (``historical-price-eod/full``).

    Args:
        endpoint: Endpoint path relative to ``/stable/``.

    Returns:
        The endpoint unchanged.

    Raises:
        ValueError: If a segment is empty, ``.``/``..``, or contains characters outside ``[A-Za-z0-9._-]``.
    """
    segments = endpoint.split("/")
    if not all(_ENDPOINT_SEGMENT_RE.match(s) and s not in (".", "..") for s in segments):
        raise ValueError(f"endpoint {endpoint!r} is not a safe relative path")
    return endpoint


def find_case_collisions(symbols: Iterable[Optional[str]]) -> dict[str, list[Optional[str]]]:
    """
    Plan-time check: group symbols whose encoded file names are equal ignoring case.

    With the case suffix of ``encode_symbol`` a collision needs two symbols with the same 32-bit hash prefix, so
    the planner treats any result as a hard error rather than something to work around.

    Args:
        symbols: Every symbol of one endpoint (duplicates are ignored).

    Returns:
        ``{casefolded name: [symbols]}`` for every group with more than one distinct symbol; empty if none.
    """
    groups: dict[str, set[Optional[str]]] = {}
    for symbol in symbols:
        groups.setdefault(encode_symbol(symbol).casefold(), set()).add(symbol)
    return {k: sorted(v, key=lambda s: (s is None, s or "")) for k, v in groups.items() if len(v) > 1}
