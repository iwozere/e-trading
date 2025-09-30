# tests/indicators_v2/test_service_fund.py
"""
from indicators_v2.service import IndicatorService
from indicators_v2.models import TickerIndicatorsRequest

async def test_pe_and_market_cap():
    svc = IndicatorService()
    req = TickerIndicatorsRequest(ticker="AAPL", indicators=["pe","market_cap","rsi"], timeframe="1d", period="6mo")
    res = await svc.compute_for_ticker(req)
    assert "pe" in res.fundamental
    assert "market_cap" in res.fundamental
"""