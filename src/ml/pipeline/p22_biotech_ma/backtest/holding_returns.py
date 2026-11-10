"""
P22 — holding-period returns for the portfolio backtest (spec §5.3), 2026-11-10.

The deal-probability metrics (`metrics`) say how many of the top k were acquired; they say nothing
about the names that were not, which is most of any list. This values every scored target at
every snapshot as if bought at that snapshot's close and held:

- **Entry:** the latest raw close on or before `as_of` that was also known by `as_of`, at most
  `MARKET_CAP_MAX_PRICE_AGE_DAYS` old. No such close, or a price series that
  `price_quality.is_price_suspect` rejects, means the name is not investable at that snapshot.
- **Deal exit (`deal`):** the first `strategic_acquisition` or `pe_take_private` naming the company,
  announced in `(as_of, as_of + horizon]` and not terminated, with an upfront price: the holder
  receives the consideration (`labels.consideration_value`, v1 CVRs at zero) on completion. The
  entry close is put on the announcement's share basis first, as in `labels.deal_return`.
- **Market exit:** otherwise the last close on or before the horizon end, with the entry close put
  on that day's share basis. `held` when that close is recent; `price_ended` when the series
  stopped earlier (delisted, halted, liquidated, reverse-merged), valued at its last close, which
  overstates the bankruptcies. `PortfolioTable` also reports a stressed variant with every
  `price_ended` at -100% so the truth lies between the two. A deal without an upfront price
  (`deal_unpriced`) is valued the same way, at the last close, but never stressed.

Exit-side prices are the outcome, not an input, so they are read without the `known_from` guard.
Dividends are ignored (few of these companies pay one). Prices are loaded once per company and
kept as numpy arrays, so the 60,000-odd target-snapshots cost one query pair per company.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest.labels import PE_DEAL_TYPE, CvrPolicy, add_months, consideration_value
from src.ml.pipeline.p22_biotech_ma.config import MARKET_CAP_MAX_PRICE_AGE_DAYS
from src.ml.pipeline.p22_biotech_ma.ingest.deal_entry import POSITIVE_DEAL_TYPE
from src.ml.pipeline.p22_biotech_ma.ingest.market_cap import split_factor_between
from src.ml.pipeline.p22_biotech_ma.ingest.price_quality import is_price_suspect
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

HOLDING_HORIZON_MONTHS = 24
EXIT_DEAL = "deal"
EXIT_DEAL_UNPRICED = "deal_unpriced"
EXIT_HELD = "held"
EXIT_PRICE_ENDED = "price_ended"
DEAL_EXIT_TYPES = (POSITIVE_DEAL_TYPE, PE_DEAL_TYPE)
_DAYS_PER_MONTH = 30.4375


@dataclass(frozen=True)
class HoldingReturn:
    """One target bought at one snapshot."""

    ret: float
    exit_reason: str
    entry_date: date
    exit_date: date

    @property
    def holding_months(self) -> float:
        """Months from the entry close to the exit."""
        return (self.exit_date - self.entry_date).days / _DAYS_PER_MONTH


@dataclass(frozen=True)
class _Series:
    dates: np.ndarray  # date ordinals, ascending
    closes: np.ndarray
    known: np.ndarray  # known_from date ordinals
    splits: List[Dict[str, Any]]


def _to_date(value: Any) -> date:
    return value.date() if isinstance(value, datetime) else value


class HoldingReturns:
    """Holding-period returns over one backtest, with prices cached per company."""

    def __init__(self, repo: Any, deals: Sequence[Dict[str, Any]], cvr_policy: CvrPolicy, data_as_of: date,
                 horizon_months: int = HOLDING_HORIZON_MONTHS,
                 suspect: Callable[[Any, int, date], bool] = is_price_suspect) -> None:
        """
        Args:
            repo: A `P22Repo`-shaped object (`get_raw_close_series`, `get_splits_as_of`).
            deals: Every entered deal (`DealLabels.deals`).
            cvr_policy: How deal consideration is valued.
            data_as_of: Date the split list and the price-quality check are read at (the labels'
                completion date): a broken series is a data defect, not a signal, so today's view of it is used.
            horizon_months: Holding period without a deal.
            suspect: Price-quality check, injectable for tests.
        """
        self.repo = repo
        self.cvr_policy = cvr_policy
        self.data_as_of = data_as_of
        self.horizon_months = horizon_months
        self._suspect_check = suspect
        self._series: Dict[int, Optional[_Series]] = {}
        self._suspect: Dict[int, bool] = {}
        self._deals: Dict[int, List[Dict[str, Any]]] = {}
        for deal in deals:
            if deal.get("deal_type") in DEAL_EXIT_TYPES and deal.get("status") != "terminated":
                self._deals.setdefault(deal["target_id"], []).append(deal)
        for company_deals in self._deals.values():
            company_deals.sort(key=lambda d: d["announcement_date"])

    def _load(self, company_id: int) -> Optional[_Series]:
        if company_id not in self._series:
            rows = self.repo.get_raw_close_series(company_id)
            if not rows:
                self._series[company_id] = None
            else:
                self._series[company_id] = _Series(
                    dates=np.array([r["trade_date"].toordinal() for r in rows], dtype=np.int64),
                    closes=np.array([r["close_raw"] for r in rows], dtype=float),
                    known=np.array([_to_date(r["known_from"]).toordinal() for r in rows], dtype=np.int64),
                    splits=self.repo.get_splits_as_of(company_id, self.data_as_of),
                )
        return self._series[company_id]

    def _is_suspect(self, company_id: int) -> bool:
        if company_id not in self._suspect:
            self._suspect[company_id] = bool(self._suspect_check(self.repo, company_id, self.data_as_of))
        return self._suspect[company_id]

    def _deal_in_window(self, company_id: int, as_of: date, end: date) -> Optional[Dict[str, Any]]:
        for deal in self._deals.get(company_id, []):
            if as_of < deal["announcement_date"] <= end:
                return deal
        return None

    @staticmethod
    def _entry(series: _Series, as_of: date) -> Optional[int]:
        """Index of the latest close on or before `as_of` known by `as_of`, if recent enough."""
        as_of_ord = as_of.toordinal()
        idx = int(np.searchsorted(series.dates, as_of_ord, side="right")) - 1
        while idx >= 0 and as_of_ord - int(series.dates[idx]) <= MARKET_CAP_MAX_PRICE_AGE_DAYS:
            if int(series.known[idx]) <= as_of_ord and series.closes[idx] > 0:
                return idx
            idx -= 1
        return None

    def _entry_on_basis(self, series: _Series, idx: int, basis: date) -> float:
        entry_date = date.fromordinal(int(series.dates[idx]))
        return float(series.closes[idx]) / split_factor_between(series.splits, entry_date, basis)

    def holding_return(self, company_id: int, as_of: date) -> Optional[HoldingReturn]:
        """The return of buying `company_id` at the `as_of` close, or None when it was not investable."""
        series = self._load(company_id)
        if series is None:
            return None
        idx = self._entry(series, as_of)
        if idx is None or self._is_suspect(company_id):
            return None
        entry_date = date.fromordinal(int(series.dates[idx]))
        end = add_months(as_of, self.horizon_months)
        deal = self._deal_in_window(company_id, as_of, end)
        if deal is not None:
            consideration = consideration_value(deal, self.cvr_policy)
            if consideration is not None:
                entry = self._entry_on_basis(series, idx, deal["announcement_date"])
                exit_date = deal.get("completion_date") or deal["announcement_date"]
                return HoldingReturn(consideration / entry - 1.0, EXIT_DEAL, entry_date, exit_date)
        last = int(np.searchsorted(series.dates, end.toordinal(), side="right")) - 1  # >= idx, since end > as_of
        exit_date = date.fromordinal(int(series.dates[last]))
        exit_close = float(series.closes[last])
        if exit_close <= 0:
            return None
        if deal is not None:
            reason = EXIT_DEAL_UNPRICED
        elif (end - exit_date).days <= MARKET_CAP_MAX_PRICE_AGE_DAYS:
            reason = EXIT_HELD
        else:
            reason = EXIT_PRICE_ENDED
        entry = self._entry_on_basis(series, idx, exit_date)
        return HoldingReturn(exit_close / entry - 1.0, reason, entry_date, exit_date)

    def for_snapshot(self, company_ids: Sequence[int], as_of: date) -> Dict[int, HoldingReturn]:
        """`holding_return` for every investable id in `company_ids`."""
        out: Dict[int, HoldingReturn] = {}
        for company_id in company_ids:
            held = self.holding_return(company_id, as_of)
            if held is not None:
                out[company_id] = held
        _logger.debug("Holding returns %s: %d of %d targets investable", as_of, len(out), len(company_ids))
        return out


def score_row_columns(held: Mapping[int, HoldingReturn], company_id: int) -> Dict[str, Any]:
    """`hold_return` / `hold_exit` / `hold_months` columns for one `scores.csv` row (None when not investable)."""
    h = held.get(company_id)
    if h is None:
        return {"hold_return": None, "hold_exit": None, "hold_months": None}
    return {"hold_return": h.ret, "hold_exit": h.exit_reason, "hold_months": round(h.holding_months, 2)}
