"""Unit tests for src.ml.pipeline.p21_momentum.strategy.selection."""

from __future__ import annotations

import unittest

from src.ml.pipeline.p21_momentum.strategy.selection import (
    CurrentHolding,
    RankedCandidate,
    enforce_sector_cap,
    rank_candidates,
    select_portfolio,
)


def _cand(ticker: str, signal: float, sector: str = "Tech") -> RankedCandidate:
    return RankedCandidate(ticker=ticker, signal=signal, sector=sector)


class TestRankCandidates(unittest.TestCase):
    def test_sorts_descending_by_signal(self):
        survivors = [_cand("A", 0.5), _cand("B", 1.5), _cand("C", 1.0)]
        ranked = rank_candidates(survivors)
        self.assertEqual([c.ticker for c in ranked], ["B", "C", "A"])
        self.assertEqual([c.rank for c in ranked], [1, 2, 3])

    def test_deterministic_tie_break_ticker_ascending(self):
        survivors = [_cand("ZZZ", 1.0), _cand("AAA", 1.0), _cand("MMM", 1.0)]
        ranked = rank_candidates(survivors)
        self.assertEqual([c.ticker for c in ranked], ["AAA", "MMM", "ZZZ"])


class TestEnforceSectorCap(unittest.TestCase):
    def test_drops_worst_ranked_over_cap(self):
        holdings = [_cand(f"T{i}", 10 - i, sector="Tech") for i in range(6)]
        for i, h in enumerate(holdings):
            h.rank = i + 1
        kept = enforce_sector_cap(holdings, max_per_sector=4)
        self.assertEqual(len(kept), 4)
        self.assertEqual([c.ticker for c in kept], ["T0", "T1", "T2", "T3"])

    def test_tie_break_at_drop_boundary_is_ticker_ascending(self):
        # All same rank (artificial tie) -> tie-break must be ticker ascending
        holdings = [_cand(t, 1.0, sector="Tech") for t in ["ZZZ", "AAA", "MMM", "BBB", "CCC"]]
        for h in holdings:
            h.rank = 1  # force a tie
        kept = enforce_sector_cap(holdings, max_per_sector=3)
        self.assertEqual(sorted(c.ticker for c in kept), ["AAA", "BBB", "CCC"])

    def test_under_cap_sector_untouched(self):
        holdings = [_cand("A", 1.0, sector="Tech"), _cand("B", 0.9, sector="Tech")]
        for i, h in enumerate(holdings):
            h.rank = i + 1
        kept = enforce_sector_cap(holdings, max_per_sector=4)
        self.assertEqual(len(kept), 2)


class TestSelectPortfolio(unittest.TestCase):
    def test_fills_to_target_from_top_entry_pool_no_current_positions(self):
        ranked = rank_candidates([_cand(f"T{i}", 100 - i, sector=f"S{i % 10}") for i in range(50)])
        result = select_portfolio(ranked, current_positions=[], forced_exits=set())
        self.assertFalse(result.underfilled)
        self.assertEqual(len(result.selected), 20)

    def test_retains_current_holding_within_hold_rank(self):
        ranked = rank_candidates([_cand(f"T{i}", 100 - i, sector=f"S{i % 10}") for i in range(70)])
        # T50 has rank 51 (0-indexed T50 -> signal=50, rank ~51), within HOLD_RANK=60
        current = [CurrentHolding(ticker="T50", sector="S0")]
        result = select_portfolio(ranked, current_positions=current, forced_exits=set())
        self.assertIn("T50", [c.ticker for c in result.selected])

    def test_forced_exit_removes_current_holding_regardless_of_rank(self):
        ranked = rank_candidates([_cand(f"T{i}", 100 - i, sector=f"S{i % 10}") for i in range(30)])
        current = [CurrentHolding(ticker="T5", sector="S5")]
        result = select_portfolio(ranked, current_positions=current, forced_exits={"T5"})
        self.assertNotIn("T5", [c.ticker for c in result.selected])

    def test_underfill_widens_to_fallback_pool(self):
        # Concentrate top-20 into 2 sectors so sector cap (4) leaves room for
        # only 8 of them; must widen to top-40 to fill the remaining 12.
        candidates = []
        for i in range(40):
            sector = "A" if i < 10 else ("B" if i < 20 else f"S{i}")
            candidates.append(_cand(f"T{i}", 100 - i, sector=sector))
        ranked = rank_candidates(candidates)
        result = select_portfolio(ranked, current_positions=[], forced_exits=set())
        self.assertFalse(result.underfilled)
        self.assertEqual(len(result.selected), 20)
        # Sector cap respected even after widening
        from collections import Counter

        counts = Counter(c.sector for c in result.selected)
        for sector, count in counts.items():
            self.assertLessEqual(count, 4)

    def test_true_underfill_sets_warn_flag(self):
        # Only 5 names total exist in the entire ranked pool -> can never reach 20
        candidates = [_cand(f"T{i}", 100 - i, sector=f"S{i}") for i in range(5)]
        ranked = rank_candidates(candidates)
        result = select_portfolio(ranked, current_positions=[], forced_exits=set())
        self.assertTrue(result.underfilled)
        self.assertEqual(len(result.selected), 5)

    def test_sector_cap_never_relaxed_even_when_underfilled(self):
        # 20 names, all one sector -> cap of 4 caps selection at 4, not 20
        candidates = [_cand(f"T{i}", 100 - i, sector="OnlySector") for i in range(20)]
        ranked = rank_candidates(candidates)
        result = select_portfolio(ranked, current_positions=[], forced_exits=set())
        self.assertTrue(result.underfilled)
        self.assertEqual(len(result.selected), 4)


if __name__ == "__main__":
    unittest.main()
