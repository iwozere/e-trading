"""
P22 — price archive adjustment math (spec §2.0.7, added v0.6).

**Raw storage, read-time adjustment is mandatory, not merely convenient**
(spec §2.0.7): `market_cap = price x shares_outstanding`, where shares
outstanding is the as-filed, unadjusted `dei:EntityCommonStockSharesOutstanding`.
Multiplying it by a *retro-adjusted* price yields a market cap wrong by
exactly the split factor — after a 1-for-20 reverse split, every historical
market cap is overstated 20x. Retro-adjusted price *levels* are also a
lookahead leak: a 2019 price becomes a different number once adjusted for a
2023 split, so any level-keyed filter (size floor, `ev_to_cash`) would use
2023 information to decide 2019 eligibility. `P22Repo.get_adjusted_close`
combines this pure math with a live raw-price/corporate-action lookup; kept
here, separately, so the adjustment logic — including the `known_from <=
as_of` lookahead guard — is unit-testable without a database connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

# Ratios only apply to these action types (spec §2.0.7's `adjusted_close`
# pseudocode); dividend/spinoff/ticker_change do not affect the price
# adjustment factor computed here.
_RATIO_ACTION_TYPES = frozenset({"split", "reverse_split"})


@dataclass(frozen=True)
class CorporateActionRatio:
    """The subset of `p22_corporate_action` fields the adjustment math needs."""

    ex_date: date
    action_type: str
    ratio: Optional[float]
    known_from_date: date


def compute_adjustment_factor(
    actions: Iterable[CorporateActionRatio],
    trade_date: date,
    as_of: date,
) -> float:
    """
    Product of split/reverse_split ratios for actions known and effective by
    `as_of` (spec §2.0.7).

    An action contributes to the factor only if:
      - it is a `split` or `reverse_split` (dividends etc. don't rescale price levels here)
      - it took effect strictly after `trade_date` and on or before `as_of`
      - it was already known by `as_of` (`known_from <= as_of`) — this guard is
        what prevents the lookahead leak: a split filed in 2023 must not adjust
        a 2019 price for a backtest running `as_of=2019-06-01`.

    Args:
        actions: Candidate corporate actions for the company (any date range —
            filtering happens here).
        trade_date: The date of the raw price being adjusted.
        as_of: The date the adjustment is being computed as of.

    Returns:
        The multiplicative adjustment factor (1.0 if no qualifying action).
    """
    factor = 1.0
    for action in actions:
        if action.action_type not in _RATIO_ACTION_TYPES:
            continue
        if action.ratio is None:
            continue
        if trade_date < action.ex_date <= as_of and action.known_from_date <= as_of:
            factor *= action.ratio
    return factor


def adjusted_close(
    raw_close: Optional[float],
    actions: Iterable[CorporateActionRatio],
    trade_date: date,
    as_of: date,
) -> Optional[float]:
    """
    Split-adjusted close to `as_of`. `None` propagates as missing, never as
    zero (spec §4's feature-function contract applies here too — a missing
    raw price must not silently become a zero price).
    """
    if raw_close is None:
        return None
    factor = compute_adjustment_factor(actions, trade_date, as_of)
    if factor == 0:
        return None
    return raw_close / factor
