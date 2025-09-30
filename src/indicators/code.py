# Project layout (proposed)
#
# src/indicators/
#   __init__.py
#   models.py                  # Pydantic models for config & results
#   registry.py                # Indicator catalog (tech + fundamentals)
#   service.py                 # Orchestrator: compute() for df + config or ticker request
#   adapters/
#       __init__.py
#       base.py                # BaseAdapter
#       ta_lib_adapter.py      # TA-Lib adapter (RSI/EMA/MACD/BBANDS/etc.)
#       pandas_ta_adapter.py   # pandas_ta fallback
#       fundamentals_adapter.py# Fundamentals provider via DataManager/common.fundamentals
#   utils.py                   # OHLCV coercion, resample, warmup masking, naming
# src/indicators/tests/
#   test_models.py
#   test_registry.py
#   test_service_tech.py
#   test_service_fund.py
#   data/ohlcv_sample.parquet
#
