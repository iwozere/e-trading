from datetime import date

from src.data.db.models.model_short_squeeze import ScreenerSnapshot, FINRAShortInterest


def test_screener_snapshot_repr_and_conversion():
    s = ScreenerSnapshot()
    s.ticker = "ABC"
    s.run_date = date.today()
    s.short_interest_pct = 0.05
    s.days_to_cover = 2.5
    s.float_shares = 1000000
    s.avg_volume_14d = 50000
    s.market_cap = 100000000
    s.screener_score = 0.8

    r = repr(s)
    assert "ScreenerSnapshot" in r


def test_finra_short_interest_basic_properties():
    f = FINRAShortInterest()
    f.ticker = "XYZ"
    f.settlement_date = date.today()
    f.short_interest_shares = 1000
    f.short_interest_pct = 1.23
    f.days_to_cover = 3.2
    out = repr(f)
    assert "FINRAShortInterest" in out
