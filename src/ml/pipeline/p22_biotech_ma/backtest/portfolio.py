"""
P22 — portfolio returns of each ranking (spec §5.3), 2026-11-10.

At every snapshot, each ranking buys its top k investable names (those with a
`holding_returns.HoldingReturn`) equal-weighted and holds them; the benchmark is every investable
target held the same way (`universe`). Pooled over snapshots:

- `mean_return`: the average snapshot portfolio's holding-period return. Snapshots are monthly and
  holdings last up to 24 months, so consecutive portfolios overlap; the average is what a strategy
  that opens one tranche a month would earn per tranche, and the snapshots are not independent.
- `excess`: `mean_return` minus the universe's average over the same snapshots.
- `beat_share`: share of snapshots whose portfolio beat the universe.
- `stressed_*`: the same with every `price_ended` exit at -100% (both sides), the pessimistic bound.
- `deal_share` / `price_ended_share`: how the picks exited; `mean_months`: average holding period.

Returns are total holding-period returns, not annualised: a deal exit after six months and a
24-month hold count the same way.
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest.holding_returns import EXIT_DEAL, EXIT_PRICE_ENDED, HoldingReturn
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

PORTFOLIO_KS = (25, 50)
UNIVERSE = "universe"


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _stressed(h: HoldingReturn) -> float:
    return -1.0 if h.exit_reason == EXIT_PRICE_ENDED else h.ret


@dataclass
class _Accumulator:
    portfolio: List[float] = field(default_factory=list)
    universe: List[float] = field(default_factory=list)
    stressed: List[float] = field(default_factory=list)
    stressed_universe: List[float] = field(default_factory=list)
    pick_returns: List[float] = field(default_factory=list)
    deals: int = 0
    price_ended: int = 0
    months: float = 0.0

    def add(self, picks: Sequence[HoldingReturn], universe: Tuple[float, float]) -> None:
        if not picks:
            return
        self.portfolio.append(sum(h.ret for h in picks) / len(picks))
        self.stressed.append(sum(_stressed(h) for h in picks) / len(picks))
        self.universe.append(universe[0])
        self.stressed_universe.append(universe[1])
        self.pick_returns.extend(h.ret for h in picks)
        self.deals += sum(1 for h in picks if h.exit_reason == EXIT_DEAL)
        self.price_ended += sum(1 for h in picks if h.exit_reason == EXIT_PRICE_ENDED)
        self.months += sum(h.holding_months for h in picks)

    def as_dict(self) -> Dict[str, Any]:
        picks = len(self.pick_returns)
        mean, uni = _mean(self.portfolio), _mean(self.universe)
        stressed, stressed_uni = _mean(self.stressed), _mean(self.stressed_universe)
        return {
            "snapshots": len(self.portfolio),
            "picks": picks,
            "mean_return": mean,
            "universe_return": uni,
            "excess": mean - uni if mean is not None and uni is not None else None,
            "beat_share": sum(1 for p, u in zip(self.portfolio, self.universe) if p > u) / len(self.portfolio) if self.portfolio else None,
            "median_pick_return": statistics.median(self.pick_returns) if picks else None,
            "stressed_return": stressed,
            "stressed_excess": stressed - stressed_uni if stressed is not None and stressed_uni is not None else None,
            "deal_share": self.deals / picks if picks else None,
            "price_ended_share": self.price_ended / picks if picks else None,
            "mean_months": self.months / picks if picks else None,
        }


class PortfolioTable:
    """Portfolio returns per `(ranking, k)`, pooled over snapshots; `universe` is the benchmark row."""

    def __init__(self, ks: Sequence[int] = PORTFOLIO_KS) -> None:
        self.ks = tuple(ks)
        self._cells: Dict[Tuple[str, int], _Accumulator] = {}

    def add_snapshot(self, rankings: Mapping[str, Sequence[int]], held: Mapping[int, HoldingReturn]) -> None:
        """One snapshot: every ranking's top k investable names, against all investable names."""
        if not held:
            return
        everyone = list(held.values())
        universe = (sum(h.ret for h in everyone) / len(everyone), sum(_stressed(h) for h in everyone) / len(everyone))
        for name, ranked in rankings.items():
            investable = [held[cid] for cid in ranked if cid in held]
            for k in self.ks:
                self._cells.setdefault((name, k), _Accumulator()).add(investable[:k], universe)
        self._cells.setdefault((UNIVERSE, 0), _Accumulator()).add(everyone, universe)

    def as_rows(self) -> List[Dict[str, Any]]:
        """One row per `(ranking, k)`, the universe first (k = all investable names)."""
        keys = sorted(self._cells, key=lambda key: (key[0] != UNIVERSE, key[0], key[1]))
        return [{"ranking": name, "k": k if name != UNIVERSE else "all", **self._cells[(name, k)].as_dict()} for name, k in keys]
