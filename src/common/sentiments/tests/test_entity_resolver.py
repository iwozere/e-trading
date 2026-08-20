"""
Unit tests for EntityResolver.

Covers spec §2.9's explicit test list: whole-word matching, ambiguous-ticker rejection,
multi-entity messages, and case insensitivity -- in particular that "ON" never matches inside
the word "on".
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sentiments.entity.resolver import EntityDef, EntityResolver


@pytest.fixture
def resolver() -> EntityResolver:
    """Build a small resolver directly from EntityDef objects (no file I/O)."""
    entities = {
        "NVDA": EntityDef(ticker="NVDA", names=["nvidia"], products=["cuda", "h100"], ambiguous=False),
        "COIN": EntityDef(ticker="COIN", names=["coinbase"], products=["base l2"], ambiguous=True),
        "ON": EntityDef(ticker="ON", names=["onsemi", "on semiconductor"], products=[], ambiguous=True),
        "META": EntityDef(ticker="META", names=["meta platforms"], products=["llama"], ambiguous=False),
    }
    return EntityResolver(entities, ambiguous_tickers=["ALL", "IT", "ON"])


class TestWholeWordMatching:
    def test_matches_company_name(self, resolver: EntityResolver) -> None:
        result = resolver.match("Nvidia just announced a new chip.")
        assert result.tickers == ["NVDA"]

    def test_matches_product_alias(self, resolver: EntityResolver) -> None:
        result = resolver.match("Training this model needs a lot of CUDA cores.")
        assert result.tickers == ["NVDA"]

    def test_no_match_on_unrelated_text(self, resolver: EntityResolver) -> None:
        result = resolver.match("The weather today is lovely.")
        assert result.tickers == []
        assert result.multi_entity is False

    def test_case_insensitive(self, resolver: EntityResolver) -> None:
        assert resolver.match("NVIDIA is up today").tickers == ["NVDA"]
        assert resolver.match("nvidia is up today").tickers == ["NVDA"]
        assert resolver.match("NvIdIa is up today").tickers == ["NVDA"]

    def test_never_matches_substring(self, resolver: EntityResolver) -> None:
        """"ON" (onsemi) must never match inside an unrelated word containing "on"."""
        result = resolver.match("Turn the button on and continue.")
        assert "ON" not in result.tickers

    def test_word_boundary_at_string_edges(self, resolver: EntityResolver) -> None:
        result = resolver.match("onsemi")
        assert result.tickers == ["ON"]
        # "reonsemiconductor" should not match despite containing the alias as a substring
        result = resolver.match("reonsemiconductorx")
        assert result.tickers == []


class TestMultiEntity:
    def test_two_tickers_both_flagged(self, resolver: EntityResolver) -> None:
        result = resolver.match("Comparing Nvidia and Meta Platforms on AI infrastructure spend.")
        assert set(result.tickers) == {"NVDA", "META"}
        assert result.multi_entity is True

    def test_single_ticker_not_flagged(self, resolver: EntityResolver) -> None:
        result = resolver.match("Nvidia earnings beat expectations.")
        assert result.tickers == ["NVDA"]
        assert result.multi_entity is False


class TestAmbiguousTickers:
    def test_ambiguous_tickers_list_populated(self, resolver: EntityResolver) -> None:
        assert resolver.ambiguous_tickers == {"ALL", "IT", "ON"}

    def test_ambiguous_entity_still_matches_full_name(self, resolver: EntityResolver) -> None:
        # "coinbase" (full name) still matches even though COIN is flagged ambiguous --
        # the ambiguous flag documents that bare "coin" must never be used as an alias,
        # it doesn't disable matching on the curated, specific alias itself.
        result = resolver.match("Coinbase reported strong Q3 volume.")
        assert result.tickers == ["COIN"]

    def test_bare_ticker_word_never_configured_as_alias(self, resolver: EntityResolver) -> None:
        # "coin" alone was deliberately never added as an alias for COIN (spec §2.4) --
        # confirm generic text containing the bare word does not match.
        result = resolver.match("Flip a coin to decide.")
        assert result.tickers == []


class TestCoverage:
    def test_is_covered(self, resolver: EntityResolver) -> None:
        assert resolver.is_covered("nvda") is True
        assert resolver.is_covered("NVDA") is True
        assert resolver.is_covered("GME") is False

    def test_known_tickers(self, resolver: EntityResolver) -> None:
        assert set(resolver.known_tickers()) == {"NVDA", "COIN", "ON", "META"}


class TestEmptyInput:
    def test_empty_string(self, resolver: EntityResolver) -> None:
        result = resolver.match("")
        assert result.tickers == []
        assert result.multi_entity is False

    def test_resolver_with_no_entities(self) -> None:
        empty_resolver = EntityResolver({})
        assert empty_resolver.match("Nvidia is great").tickers == []
        assert empty_resolver.known_tickers() == []


class TestFromYaml:
    def test_loads_real_tickers_yml(self) -> None:
        """Smoke test against the actual seeded entity map shipped with the module."""
        resolver = EntityResolver.from_yaml()
        assert resolver.is_covered("NVDA")
        assert resolver.is_covered("COIN")
        assert len(resolver.known_tickers()) >= 100
        assert "ALL" in resolver.ambiguous_tickers
        assert "ON" in resolver.ambiguous_tickers

        result = resolver.match("NVIDIA unveiled its new H100 successor at the keynote.")
        assert "NVDA" in result.tickers
