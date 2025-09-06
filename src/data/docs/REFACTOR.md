All set!

1. I translated the whole TS to English.
2. Renamed the method to `get_olhcv` everywhere.
3. Reworked fundamentals to use JSON file cache with `provider_symbol_timestamp.json`, a 7-day cache-first rule for all stock providers (yfinance/IBKR/FMP/Alpha Vantage, etc.), and combination of multiple providers’ snapshots.

If you want, I can also drop in code skeletons for:

* `DataManager` with the new `get_olhcv`
* JSON fundamentals cache helper (`find_latest_json`, `write_json`, index maintenance)
* A simple `combine_snapshots(...)` with pluggable strategies.
