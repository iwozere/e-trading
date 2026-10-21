"""Regression tests for the 2026-10 P18 13F data fixes (value units, stale index cache, duplicate rows, daily consensus)."""

import os
import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.downloader.edgar_downloader import EdgarDownloader, read_13f_holdings_file
from src.ml.pipeline.p18_institutional_flow_tracker.processors.position_delta_calculator import PositionDeltaCalculator

_INFOTABLE = """<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable><nameOfIssuer>ALPHA</nameOfIssuer><cusip>020764106</cusip><value>132199012</value>
    <shrsOrPrnAmt><sshPrnamt>644025</sshPrnamt></shrsOrPrnAmt><investmentDiscretion>SOLE</investmentDiscretion></infoTable>
</informationTable>"""


def _holdings(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["cik", "institution_name", "ticker", "shares", "value_usd", "pct_of_portfolio", "put_call"])


def test_infotable_value_is_whole_dollars_since_2023(tmp_path: Path) -> None:
    edgar = EdgarDownloader(cache_dir=tmp_path)
    assert edgar.parse_13f_infotable(_INFOTABLE, 24046, "X", "2026Q1")["value_usd"].iloc[0] == 132_199_012
    assert edgar.parse_13f_infotable(_INFOTABLE, 24046, "X", "2022Q3")["value_usd"].iloc[0] == 132_199_012_000


def test_legacy_cached_holdings_are_scaled_back(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.csv.gz"
    pd.DataFrame({"cusip": ["1"], "value_usd": [132_199_012_000], "pct_of_portfolio": [0.2]}).to_csv(legacy, index=False, compression="gzip")
    fixed = tmp_path / "fixed.csv.gz"
    pd.DataFrame({"cusip": ["1"], "value_usd": [132_199_012], "pct_of_portfolio": [0.2], "value_scale": ["usd"]}).to_csv(
        fixed, index=False, compression="gzip")

    assert read_13f_holdings_file(legacy)["value_usd"].iloc[0] == pytest.approx(132_199_012)
    assert read_13f_holdings_file(fixed)["value_usd"].iloc[0] == 132_199_012


def _hits(n: int) -> list:
    return [{"_source": {"ciks": [str(1000 + i)], "display_names": [f"Fund {i}"], "adsh": f"000-{i}", "file_date": "2026-08-01"}}
            for i in range(n)]


def test_index_cached_inside_filing_window_is_refreshed(tmp_path: Path) -> None:
    edgar = EdgarDownloader(cache_dir=tmp_path)
    stale = edgar._13f_index_dir / "2026_Q2.csv.gz"
    stale.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cik": ["1"], "institution_name": ["Early"], "accession_number": ["a"], "filed_date": ["2026-07-01"]}).to_csv(
        stale, index=False, compression="gzip")
    cached_on = time.mktime(date(2026, 7, 2).timetuple())
    os.utime(stale, (cached_on, cached_on))

    with patch.object(edgar, "_efts_search", return_value=_hits(5)) as search:
        df = edgar.download_13f_index(2026, 2)

    search.assert_called_once()
    assert len(df) == 5
    assert len(pd.read_csv(stale, compression="gzip")) == 5


def test_index_cached_after_filing_window_is_reused(tmp_path: Path) -> None:
    edgar = EdgarDownloader(cache_dir=tmp_path)
    final = edgar._13f_index_dir / "2025_Q4.csv.gz"
    final.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cik": ["1"], "institution_name": ["A"], "accession_number": ["a"], "filed_date": ["2026-01-05"]}).to_csv(
        final, index=False, compression="gzip")

    with patch.object(edgar, "_efts_search") as search:
        assert len(edgar.download_13f_index(2025, 4)) == 1
    search.assert_not_called()


def test_duplicate_rows_do_not_create_false_exits(tmp_path: Path) -> None:
    rows = [
        {"cik": "1", "institution_name": "F", "ticker": "BAC", "shares": 1000, "value_usd": 40_000, "pct_of_portfolio": 0.1, "put_call": None},
        {"cik": "1", "institution_name": "F", "ticker": "BAC", "shares": 50, "value_usd": 2_000, "pct_of_portfolio": 0.005, "put_call": None},
        {"cik": "1", "institution_name": "F", "ticker": "BAC", "shares": 900, "value_usd": 36_000, "pct_of_portfolio": 0.09, "put_call": "Call"},
    ]
    delta = PositionDeltaCalculator(results_dir=tmp_path).calculate(_holdings(rows), _holdings(rows[:2]))

    assert len(delta) == 1
    assert delta["shares_prev"].iloc[0] == 1050
    assert delta["exit_type"].iloc[0] == "unchanged"


def test_daily_consensus_uses_every_cached_filer(tmp_path: Path) -> None:
    from src.ml.pipeline.p18_institutional_flow_tracker.pipeline import InstitutionalFlowPipeline

    pipeline = InstitutionalFlowPipeline.__new__(InstitutionalFlowPipeline)
    pipeline._edgar = EdgarDownloader(cache_dir=tmp_path)
    pipeline.config = MagicMock(min_aum_usd=0)
    consensus = MagicMock(return_value=pd.DataFrame({"ticker": ["X"]}))
    pipeline._consensus_from_frames = consensus  # type: ignore[method-assign]
    for year, quarter in ((2026, 1), (2026, 2)):
        folder = pipeline._edgar._13f_holdings_dir / f"{year}_Q{quarter}"
        folder.mkdir(parents=True)
        for cik in (1, 2, 3):
            pd.DataFrame({"cusip": ["c"], "value_usd": [10.0], "value_scale": ["usd"]}).to_csv(
                folder / f"{cik:010d}.csv.gz", index=False, compression="gzip")
    pipeline._edgar.download_13f_infotable = MagicMock()  # type: ignore[method-assign]

    today = pd.DataFrame([{"cik": "3", "accession_number": "a", "institution_name": "Fund 3"}])
    result = pipeline._build_quarterly_consensus(2026, 2, today, force_refresh=False)

    current, prior = consensus.call_args.args[:2]
    assert len(current) == 3 and len(prior) == 3
    assert result["ticker"].tolist() == ["X"]
