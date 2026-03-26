import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.ml.pipeline.p14_ath.ath_pipeline import ATHPipeline
from src.ml.pipeline.p14_ath.config import ATHPipelineConfig

def test_ath_algorithm_logic():
    # Setup mock data
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    # Prices: 10, 12 (ATH1), 11, 10, 13 (ATH2), 12, 11, 14 (ATH3), 13, 15 (ATH4)
    prices = [10, 12, 11, 10, 13, 12, 11, 14, 13, 15]
    df = pd.DataFrame({'close': prices}, index=dates)
    
    config = ATHPipelineConfig.create_default()
    config.generate_plots = False
    pipeline = ATHPipeline(config)
    
    # Mock data_manager.get_ohlcv
    pipeline.data_manager.get_ohlcv = lambda symbol, timeframe, start_date, end_date: df
    
    results = pipeline.analyze_ticker("MOCK")
    
    # Sequential ATHs expected:
    # 1. Price 12 at Day 1. Drawdown min between Day 1 and Day 4 is Price 10 at Day 3.
    # 2. Price 13 at Day 4. Drawdown min between Day 4 and Day 7 is Price 11 at Day 6.
    # 3. Price 14 at Day 7. Drawdown min between Day 7 and Day 9 is Price 13 at Day 8.
    # 4. Price 15 at Day 9. Final window drawdown is Price 15 (no subsequent drops) or itself.
    
    assert len(results) == 5
    
    # First ATH (Price 10)
    assert results.iloc[0]['ATH_Price'] == 10.0
    assert results.iloc[0]['Max_Drop_Price'] == 10.0
    assert results.iloc[0]['Drop_Percent'] == 0.0
    
    # Second ATH (Price 12)
    assert results.iloc[1]['ATH_Price'] == 12.0
    assert results.iloc[1]['Max_Drop_Price'] == 10.0
    assert results.iloc[1]['Drop_Percent'] == -16.67
    
    # Third ATH (Price 13)
    assert results.iloc[2]['ATH_Price'] == 13.0
    assert results.iloc[2]['Max_Drop_Price'] == 11.0
    assert results.iloc[2]['Drop_Percent'] == -15.38
    
    # Fourth ATH (Price 14)
    assert results.iloc[3]['ATH_Price'] == 14.0
    assert results.iloc[3]['Max_Drop_Price'] == 13.0
    assert results.iloc[3]['Drop_Percent'] == -7.14
    
    # Fifth ATH (Price 15)
    assert results.iloc[4]['ATH_Price'] == 15.0
    assert results.iloc[4]['Max_Drop_Price'] == 15.0
    assert results.iloc[4]['Drop_Percent'] == 0.0
