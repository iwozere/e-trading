"""Tests for backtest/holding_returns.py and backtest/portfolio.py (spec §5.3 portfolio backtest)."""

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest.holding_returns import (
    EXIT_DEAL,
    EXIT_DEAL_UNPRICED,
    EXIT_HELD,
    EXIT_PRICE_ENDED,
    HoldingReturn,
    HoldingReturns,
    score_row_columns,
)
from src.ml.pipeline.p22_biotech_ma.backtest.labels import CvrPolicy
from src.ml.pipeline.p22_biotech_ma.backtest.portfolio import UNIVERSE, PortfolioTable

_ZERO = CvrPolicy(valuation_method="zero", return_basis="upfront_per_share")
_AS_OF = date(2020, 1, 1)
_DATA_AS_OF = date(2026, 11, 1)


class _PriceRepo:
    def __init__(self) -> None:
        self.prices: Dict[int, List[Dict[str, Any]]] = {}
        self.splits: Dict[int, List[Dict[str, Any]]] = {}
        self.series_calls = 0

    def add(self, company_id: int, day: date, close: float, known: date | None = None) -> None:
        known_from = datetime.combine(known or day, datetime.min.time(), tzinfo=timezone.utc)
        self.prices.setdefault(company_id, []).append({"trade_date": day, "close_raw": close, "known_from": known_from})
        self.prices[company_id].sort(key=lambda r: r["trade_date"])

    def get_raw_close_series(self, company_id: int) -> List[Dict[str, Any]]:
        self.series_calls += 1
        return self.prices.get(company_id, [])

    def get_splits_as_of(self, company_id: int, as_of: date) -> List[Dict[str, Any]]:
        return self.splits.get(company_id, [])


def _book(repo: _PriceRepo, deals: List[Dict[str, Any]] | None = None, suspect: frozenset[int] = frozenset()) -> HoldingReturns:
    return HoldingReturns(repo, deals or [], _ZERO, _DATA_AS_OF, suspect=lambda r, cid, d: cid in suspect)


def _deal(target_id: int, announced: date, upfront: float | None, **extra: Any) -> Dict[str, Any]:
    return {"target_id": target_id, "announcement_date": announced, "deal_type": "strategic_acquisition",
            "upfront_per_share": upfront, **extra}


def test_held_to_horizon_returns_price_change():
    repo = _PriceRepo()
    repo.add(1, date(2019, 12, 31), 10.0)
    repo.add(1, date(2021, 12, 31), 15.0)
    held = _book(repo).holding_return(1, _AS_OF)
    assert held is not None and held.exit_reason == EXIT_HELD
    assert held.ret == pytest.approx(0.5)
    assert held.exit_date == date(2021, 12, 31)


def test_not_investable_without_recent_known_entry_price_or_when_suspect():
    repo = _PriceRepo()
    repo.add(1, date(2019, 11, 1), 10.0)  # too old at as_of
    repo.add(2, date(2019, 12, 31), 10.0, known=date(2020, 3, 1))  # landed after as_of
    repo.add(3, date(2019, 12, 31), 10.0)
    repo.add(3, date(2021, 12, 31), 12.0)
    book = _book(repo, suspect=frozenset({3}))
    assert book.holding_return(1, _AS_OF) is None
    assert book.holding_return(2, _AS_OF) is None
    assert book.holding_return(3, _AS_OF) is None
    assert book.holding_return(4, _AS_OF) is None  # no prices at all


def test_entry_skips_a_later_known_row_for_an_earlier_known_one():
    repo = _PriceRepo()
    repo.add(1, date(2019, 12, 30), 8.0)
    repo.add(1, date(2019, 12, 31), 10.0, known=date(2020, 2, 1))
    repo.add(1, date(2021, 12, 31), 16.0)
    held = _book(repo).holding_return(1, _AS_OF)
    assert held is not None and held.ret == pytest.approx(1.0)  # 16 / 8


def test_series_that_stops_is_price_ended_at_last_close():
    repo = _PriceRepo()
    repo.add(1, date(2019, 12, 31), 10.0)
    repo.add(1, date(2020, 6, 30), 2.0)
    held = _book(repo).holding_return(1, _AS_OF)
    assert held is not None and held.exit_reason == EXIT_PRICE_ENDED
    assert held.ret == pytest.approx(-0.8)


def test_deal_exit_pays_consideration_on_announcement_basis():
    repo = _PriceRepo()
    repo.add(1, date(2019, 12, 31), 10.0)
    repo.splits[1] = [{"ex_date": date(2020, 6, 1), "ratio": 0.1}]  # 1:10 reverse split before the deal
    deal = _deal(1, date(2021, 1, 15), 150.0, completion_date=date(2021, 4, 1))
    held = _book(repo, [deal]).holding_return(1, _AS_OF)
    assert held is not None and held.exit_reason == EXIT_DEAL
    assert held.ret == pytest.approx(0.5)  # entry 10 -> 100 on the post-split basis, paid 150
    assert held.exit_date == date(2021, 4, 1)


def test_terminated_or_out_of_window_deal_is_ignored_and_unpriced_deal_uses_last_close():
    repo = _PriceRepo()
    repo.add(1, date(2019, 12, 31), 10.0)
    repo.add(1, date(2021, 12, 31), 9.0)
    deals = [_deal(1, date(2020, 5, 1), 20.0, status="terminated"), _deal(1, date(2022, 6, 1), 30.0)]
    held = _book(repo, deals).holding_return(1, _AS_OF)
    assert held is not None and held.exit_reason == EXIT_HELD and held.ret == pytest.approx(-0.1)

    repo.add(2, date(2019, 12, 31), 10.0)
    repo.add(2, date(2020, 9, 1), 13.0)
    held2 = _book(repo, [_deal(2, date(2020, 7, 1), None)]).holding_return(2, _AS_OF)
    assert held2 is not None and held2.exit_reason == EXIT_DEAL_UNPRICED and held2.ret == pytest.approx(0.3)


def test_prices_load_once_per_company_and_snapshot_keeps_only_investable():
    repo = _PriceRepo()
    repo.add(1, date(2019, 12, 31), 10.0)
    repo.add(1, date(2021, 12, 31), 11.0)
    book = _book(repo)
    out = book.for_snapshot([1, 2], _AS_OF)
    book.for_snapshot([1], date(2020, 1, 2))
    assert set(out) == {1}
    assert repo.series_calls == 2  # company 1 once, company 2 once
    assert score_row_columns(out, 1)["hold_exit"] == EXIT_HELD
    assert score_row_columns(out, 2) == {"hold_return": None, "hold_exit": None, "hold_months": None}


def _h(ret: float, reason: str = EXIT_HELD) -> HoldingReturn:
    return HoldingReturn(ret, reason, date(2020, 1, 1), date(2022, 1, 1))


def test_portfolio_table_buys_top_k_investable_against_universe():
    held = {1: _h(1.0, EXIT_DEAL), 2: _h(0.0), 3: _h(-0.5, EXIT_PRICE_ENDED), 4: _h(0.1)}
    table = PortfolioTable(ks=(2,))
    table.add_snapshot({"good": [9, 1, 2, 3, 4], "bad": [3, 4, 1]}, held)  # 9 has no price: skipped
    rows = {(r["ranking"], r["k"]): r for r in table.as_rows()}
    universe = rows[(UNIVERSE, "all")]
    assert universe["mean_return"] == pytest.approx(0.15)
    good, bad = rows[("good", 2)], rows[("bad", 2)]
    assert good["mean_return"] == pytest.approx(0.5) and good["excess"] == pytest.approx(0.35)
    assert good["beat_share"] == 1.0 and good["deal_share"] == 0.5 and good["price_ended_share"] == 0.0
    assert bad["mean_return"] == pytest.approx(-0.2) and bad["beat_share"] == 0.0
    # stressed: company 3's -0.5 becomes -1.0 on both sides
    assert bad["stressed_return"] == pytest.approx(-0.45)
    assert bad["stressed_excess"] == pytest.approx(-0.45 - 0.025)
    assert table.as_rows()[0]["ranking"] == UNIVERSE


def test_portfolio_table_ignores_snapshot_without_investable_names():
    table = PortfolioTable(ks=(2,))
    table.add_snapshot({"good": [1]}, {})
    assert table.as_rows() == []
