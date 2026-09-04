"""Unit tests for `insider_activity.load_insider_activity`."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from src.portfolio.pnl_alert.insider_activity import _describe_role, load_insider_activity

AS_OF = date(2026, 9, 4)  # a Friday


def _day_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(
    ticker: str,
    insider_name: str = "Jensen Huang",
    transaction_code: str = "P",
    acquired_disposed_code: str = "A",
    shares: int = 1000,
    price_per_share: float = 25.0,
    transaction_date: str = "2026-09-01",
    filed_date: str = "2026-09-02",
    is_director: bool = False,
    is_officer: bool = True,
    is_ten_percent_owner: bool = False,
    officer_title: str = "CEO",
    is_10b5_1_plan: bool = False,
) -> dict:
    return {
        "ticker": ticker,
        "issuer_cik": "1",
        "insider_name": insider_name,
        "transaction_code": transaction_code,
        "acquired_disposed_code": acquired_disposed_code,
        "shares": shares,
        "price_per_share": price_per_share,
        "total_value_usd": shares * price_per_share,
        "filed_date": filed_date,
        "transaction_date": transaction_date,
        "is_director": is_director,
        "is_officer": is_officer,
        "is_ten_percent_owner": is_ten_percent_owner,
        "officer_title": officer_title,
        "is_10b5_1_plan": is_10b5_1_plan,
        "is_derivative": False,
    }


def test_empty_tickers_returns_empty_without_calling_edgar():
    edgar = MagicMock()
    result = load_insider_activity([], edgar=edgar, as_of=AS_OF)
    assert result == {}
    edgar.download_form4_filings.assert_not_called()


def test_filters_to_requested_tickers_only():
    edgar = MagicMock()
    edgar.download_form4_filings.return_value = _day_df([_row("NVDA"), _row("AAPL")])

    # lookback_days=1 walks exactly one weekday, so the mocked single-day
    # response is only combined once.
    result = load_insider_activity(["NVDA"], edgar=edgar, as_of=AS_OF, lookback_days=1)

    assert set(result.keys()) == {"NVDA"}
    assert len(result["NVDA"]) == 1


def test_ticker_lookup_is_case_insensitive():
    edgar = MagicMock()
    edgar.download_form4_filings.return_value = _day_df([_row("NVDA")])

    result = load_insider_activity(["nvda"], edgar=edgar, as_of=AS_OF, lookback_days=3)

    assert "NVDA" in result


def test_never_fetches_todays_date():
    """The window must never include `as_of` itself — see the module docstring."""
    edgar = MagicMock()
    edgar.download_form4_filings.return_value = pd.DataFrame()

    load_insider_activity(["NVDA"], edgar=edgar, as_of=AS_OF, lookback_days=5)

    fetched_dates = {call.kwargs["as_of_date"] for call in edgar.download_form4_filings.call_args_list}
    assert AS_OF not in fetched_dates
    assert all(d < AS_OF for d in fetched_dates)


def test_weekends_are_skipped():
    edgar = MagicMock()
    edgar.download_form4_filings.return_value = pd.DataFrame()

    load_insider_activity(["NVDA"], edgar=edgar, as_of=AS_OF, lookback_days=7)

    fetched_dates = {call.kwargs["as_of_date"] for call in edgar.download_form4_filings.call_args_list}
    assert all(d.weekday() < 5 for d in fetched_dates)


def test_transactions_sorted_by_transaction_date_descending():
    edgar = MagicMock()
    edgar.download_form4_filings.return_value = _day_df(
        [
            _row("NVDA", transaction_date="2026-08-20"),
            _row("NVDA", transaction_date="2026-09-01"),
        ]
    )

    result = load_insider_activity(["NVDA"], edgar=edgar, as_of=AS_OF, lookback_days=20)

    dates = [t.transaction_date for t in result["NVDA"]]
    assert dates == sorted(dates, reverse=True)


def test_no_activity_omits_ticker_from_result():
    edgar = MagicMock()
    edgar.download_form4_filings.return_value = pd.DataFrame()

    result = load_insider_activity(["NVDA"], edgar=edgar, as_of=AS_OF, lookback_days=3)

    assert result == {}


def test_form4_read_failure_for_one_day_does_not_abort_the_window():
    edgar = MagicMock()
    edgar.download_form4_filings.side_effect = Exception("boom")

    result = load_insider_activity(["NVDA"], edgar=edgar, as_of=AS_OF, lookback_days=3)

    assert result == {}


def test_is_10b5_1_plan_and_role_fields_round_trip():
    edgar = MagicMock()
    edgar.download_form4_filings.return_value = _day_df(
        [_row("NVDA", is_director=True, is_officer=False, officer_title="", is_10b5_1_plan=True)]
    )

    result = load_insider_activity(["NVDA"], edgar=edgar, as_of=AS_OF, lookback_days=3)

    txn = result["NVDA"][0]
    assert txn.role == "Director"
    assert txn.is_10b5_1_plan is True


def test_describe_role_combines_multiple_flags():
    row = pd.Series({"is_director": True, "is_officer": True, "officer_title": "CFO", "is_ten_percent_owner": True})
    assert _describe_role(row) == "Director / Officer (CFO) / 10% Owner"


def test_describe_role_defaults_to_insider_when_no_flags_set():
    row = pd.Series({"is_director": False, "is_officer": False, "is_ten_percent_owner": False})
    assert _describe_role(row) == "Insider"
