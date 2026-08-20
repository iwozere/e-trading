# src/common/sentiments/entity/resolver.py
"""
Entity resolution: ticker <-> company/product name mapping.

Adapters never see ticker semantics beyond a search term (see sentiment-spec-rev2.md §2.1) --
all ticker <-> entity logic lives here. This is what lets Hacker News, which needs name-based
resolution rather than cashtag search, use the same adapter contract as Bluesky/StockTwits.

Two distinct concerns share this module because they share the same curated data file
(``tickers.yml``):

- ``EntityResolver`` -- whole-word company/product name matching against free text (used by the
  Hacker News adapter's entity match pass, spec §2.4).
- ``ambiguous_tickers`` -- a flat set of ticker symbols that collide with common English words
  (``$ALL``, ``$IT``, ``$ON`` ...). Cashtag-based adapters (Bluesky) use this to require the
  ``$`` prefix and drop bare-name matches for those tickers (spec §2.3).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_TICKERS_PATH = Path(__file__).resolve().parent / "tickers.yml"


@dataclass(frozen=True)
class EntityDef:
    """Curated alias set for a single ticker."""

    ticker: str
    names: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    ambiguous: bool = False


@dataclass(frozen=True)
class EntityMatchResult:
    """Result of matching a piece of text against the entity map."""

    tickers: List[str]
    multi_entity: bool


class EntityResolver:
    """
    Whole-word, case-insensitive matcher between free text and tickers.

    Matching is driven entirely by curated ``names``/``products`` aliases per ticker
    (``entity/tickers.yml``) -- never by substring, and never by the raw ticker symbol,
    since source text (e.g. Hacker News comments) never contains ``$NVDA``.
    """

    def __init__(self, entities: Dict[str, EntityDef], ambiguous_tickers: List[str] | None = None):
        self._entities = entities
        self.ambiguous_tickers = {t.upper() for t in (ambiguous_tickers or [])}
        self._patterns: Dict[str, re.Pattern[str]] = {}
        for ticker, edef in entities.items():
            aliases = list(edef.names) + list(edef.products)
            if not aliases:
                continue
            # Longest-first so multi-word aliases are tried before their shorter substrings.
            escaped = sorted((re.escape(a) for a in aliases), key=len, reverse=True)
            pattern = r"\b(?:" + "|".join(escaped) + r")\b"
            self._patterns[ticker] = re.compile(pattern, re.IGNORECASE)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "EntityResolver":
        """
        Load an entity map from a YAML file shaped like ``tickers.yml``.

        Args:
            path: Path to the YAML file. Defaults to ``entity/tickers.yml``.

        Returns:
            A configured EntityResolver.
        """
        resolved_path = Path(path) if path else DEFAULT_TICKERS_PATH
        with open(resolved_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        entities: Dict[str, EntityDef] = {}
        for ticker, data in (raw.get("entities") or {}).items():
            data = data or {}
            entities[ticker.upper()] = EntityDef(
                ticker=ticker.upper(),
                names=[str(n) for n in (data.get("names") or [])],
                products=[str(p) for p in (data.get("products") or [])],
                ambiguous=bool(data.get("ambiguous", False)),
            )

        ambiguous_tickers = [str(t) for t in (raw.get("ambiguous_tickers") or [])]
        _logger.info(
            "Loaded entity map: %d tickers, %d ambiguous cashtags", len(entities), len(ambiguous_tickers)
        )
        return cls(entities, ambiguous_tickers=ambiguous_tickers)

    def known_tickers(self) -> List[str]:
        """Return every ticker present in the entity map."""
        return list(self._entities.keys())

    def is_covered(self, ticker: str) -> bool:
        """Return True if ``ticker`` has a curated entry in the entity map."""
        return ticker.upper() in self._entities

    def get_names(self, ticker: str) -> List[str]:
        """
        Return the curated company/product names for ``ticker``, or ``[]`` if uncovered.

        Used by cashtag+name search adapters (Bluesky, spec §2.3) to build a company-name query
        alongside the cashtag one -- cashtag-only search undercounts on platforms where the
        convention is less established than it was on Twitter.
        """
        entity = self._entities.get(ticker.upper())
        return list(entity.names) if entity else []

    def match(self, text: str) -> EntityMatchResult:
        """
        Match free text against every ticker's aliases.

        Args:
            text: Free text to search (e.g. a cleaned Hacker News comment body).

        Returns:
            EntityMatchResult listing every matched ticker, with ``multi_entity=True`` when
            two or more distinct tickers matched (spec §2.4: comparison threads are legitimate
            signal for both names).
        """
        if not text:
            return EntityMatchResult(tickers=[], multi_entity=False)
        matched = [ticker for ticker, pattern in self._patterns.items() if pattern.search(text)]
        return EntityMatchResult(tickers=matched, multi_entity=len(matched) >= 2)
