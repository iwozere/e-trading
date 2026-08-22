"""
P21 Momentum — Selection: ranking, hysteresis, sector cap (docs/pipeline-specification.md §6).

Operation order is **strict** (spec is explicit that order matters):
rank -> retain (<=HOLD_RANK) -> forced exits -> sector cap on retained ->
fill from top-ENTRY_RANK -> widen to top-FALLBACK_POOL_RANK on underfill ->
accept-smaller-and-WARN_UNDERFILLED as last resort.

Deterministic tie-break (`signal_desc, ticker_asc`, spec §16) is applied at
the rank step AND at the sector-cap-drop step — §14.9 B10 requires
bit-identical reruns, which Python's stable sort alone does not guarantee
once dict ordering or an unstable comparator is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from src.ml.pipeline.p21_momentum.config import (
    ENTRY_RANK,
    FALLBACK_POOL_RANK,
    HOLD_RANK,
    MAX_PER_SECTOR,
    TARGET_COUNT,
)


@dataclass(slots=True)
class RankedCandidate:
    """A survivor of F1-F6, ranked by signal."""

    ticker: str
    signal: float
    sector: Optional[str]
    rank: int = 0  # filled in by rank_candidates()


@dataclass(slots=True)
class CurrentHolding:
    """A currently-held position, for the retain/exit decision."""

    ticker: str
    sector: Optional[str]


@dataclass(slots=True)
class SelectionResult:
    """Output of select_portfolio()."""

    selected: List[RankedCandidate]
    underfilled: bool  # True -> log WARN_UNDERFILLED (spec §6 step 7)


def _tie_break_key(c: RankedCandidate):
    """Deterministic sort key: signal descending, ticker ascending (spec §16)."""
    return (-c.signal, c.ticker)


def rank_candidates(survivors: List[RankedCandidate]) -> List[RankedCandidate]:
    """
    Rank survivors by descending signal with a deterministic tie-break.

    Args:
        survivors: Post-filter candidates (F1-F6 passed), rank field ignored
            on input.

    Returns:
        New list, sorted, with .rank set to 1-based position.
    """
    ranked = sorted(survivors, key=_tie_break_key)
    for i, c in enumerate(ranked):
        c.rank = i + 1
    return ranked


def enforce_sector_cap(
    holdings: List[RankedCandidate], max_per_sector: int = MAX_PER_SECTOR
) -> List[RankedCandidate]:
    """
    Drop the worst-ranked names in any sector exceeding max_per_sector.

    Ties at the drop boundary are broken by the same (signal_desc,
    ticker_asc) key as rank_candidates(), not by dict/list insertion order
    (spec §14.9 B10 determinism requirement).

    Args:
        holdings: Candidates, each with .rank and .sector already set.
        max_per_sector: Spec §6 MAX_PER_SECTOR (default 4).

    Returns:
        New list with excess names removed, best-ranked-per-sector kept.
    """
    by_sector: Dict[Optional[str], List[RankedCandidate]] = {}
    for h in holdings:
        by_sector.setdefault(h.sector, []).append(h)

    kept: List[RankedCandidate] = []
    for names in by_sector.values():
        if len(names) <= max_per_sector:
            kept.extend(names)
            continue
        # Keep the best max_per_sector by rank ascending (= best signal first),
        # deterministic tie-break on ties.
        ordered = sorted(names, key=lambda c: (c.rank, c.ticker))
        kept.extend(ordered[:max_per_sector])

    # Preserve overall rank order in the output for readability/determinism.
    kept.sort(key=lambda c: c.rank)
    return kept


def select_portfolio(
    ranked: List[RankedCandidate],
    current_positions: List[CurrentHolding],
    forced_exits: Set[str],
    entry_rank: int = ENTRY_RANK,
    hold_rank: int = HOLD_RANK,
    max_per_sector: int = MAX_PER_SECTOR,
    target_count: int = TARGET_COUNT,
    fallback_pool_rank: int = FALLBACK_POOL_RANK,
) -> SelectionResult:
    """
    Run the full 7-step selection procedure from spec §6, in the exact order given.

    Args:
        ranked: Output of rank_candidates() — every survivor, ranked.
        current_positions: Currently-held tickers (from _state/current_positions.json).
        forced_exits: Tickers dropped from the index / failed F1/F2/F3/F5 /
            delisted this run — always exited regardless of rank.
        entry_rank, hold_rank, max_per_sector, target_count, fallback_pool_rank:
            Spec §16 parameters (defaults from config.py).

    Returns:
        SelectionResult with the final selected list (<=target_count) and an
        underfilled flag for WARN_UNDERFILLED logging.
    """
    rank_of: Dict[str, int] = {c.ticker: c.rank for c in ranked}
    by_ticker: Dict[str, RankedCandidate] = {c.ticker: c for c in ranked}

    # Step 2: retain current positions with rank <= hold_rank
    keep_tickers = [p.ticker for p in current_positions if rank_of.get(p.ticker, hold_rank + 1) <= hold_rank]

    # Step 3: forced exits removed regardless of rank
    keep_tickers = [t for t in keep_tickers if t not in forced_exits]

    keep = [by_ticker[t] for t in keep_tickers if t in by_ticker]

    # Step 4: sector cap on retained names
    keep = enforce_sector_cap(keep, max_per_sector)
    kept_tickers = {c.ticker for c in keep}

    # Step 5: fill to target_count from ranked[:entry_rank], skipping held/capped names
    selected = list(keep)
    selected_sectors: Dict[Optional[str], int] = {}
    for c in selected:
        selected_sectors[c.sector] = selected_sectors.get(c.sector, 0) + 1

    def _try_fill(pool: List[RankedCandidate]) -> None:
        for c in pool:
            if len(selected) >= target_count:
                break
            if c.ticker in kept_tickers or c.ticker in {s.ticker for s in selected}:
                continue
            if c.ticker in forced_exits:
                # Defensive: a forced-exit ticker (dropped from index / failed a
                # filter / delisted) must never be filled in as a *new* entry
                # either, even though in practice it should already be absent
                # from `ranked` (filter failures and delistings both prevent a
                # ticker from surviving to the ranked survivor pool).
                continue
            sector_count = selected_sectors.get(c.sector, 0)
            if sector_count >= max_per_sector:
                continue
            selected.append(c)
            selected_sectors[c.sector] = sector_count + 1

    top_entry_pool = [c for c in ranked if c.rank <= entry_rank]
    _try_fill(top_entry_pool)

    # Step 6: widen to top-fallback_pool_rank if still underfilled (sector cap stays enforced)
    underfilled_after_step5 = len(selected) < target_count
    if underfilled_after_step5:
        widened_pool = [c for c in ranked if c.rank <= fallback_pool_rank]
        _try_fill(widened_pool)

    # Step 7: accept smaller portfolio if still underfilled
    underfilled = len(selected) < target_count
    selected.sort(key=lambda c: c.rank)
    return SelectionResult(selected=selected, underfilled=underfilled)
