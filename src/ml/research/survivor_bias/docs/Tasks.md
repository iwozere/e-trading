# Tasks

## Implementation Status

### ✅ COMPLETED
- [x] Rules frozen in [Design](Design.md) before the first run (2026-11-22)
- [x] Delisting rule, month-end panel, monthly strategies, P24 on the EODHD store, job with bias tables
- [x] Unit tests on synthetic data; pyright, mypy, flake8 clean

- [x] Runs 2026-11-22: `20261122-full` (as frozen; bad prints dominate `full`), `20261122-amend1` (two-source check),
      `20261122-amend2` (bad-print screen; the run to read). See [Design](Design.md) amendments.

### 🚀 PLANNED
- [ ] Quality-value: needs fundamentals for delisted stocks (not in EODHD, not in the vintage); only its benchmark bias
      is measured here
- [ ] Use the measured bias to restate the "upper bound" numbers in the strategy notes

## Known Issues
- P21 is reduced to its momentum core: no sector cap, regime overlay, inverse-volatility sizing or earnings blackout,
  and the S&P 500 is approximated by the 500 most liquid eligible stocks.
- EODHD has no delisting reasons or deal prices; the delisting rule is a stand-in (two versions bracket it).
- The `vintage_like` universe is the EODHD-to-FMP mapping, which misses a few vintage stocks (different symbol or name).
- EODHD's history mostly starts on 1999-01-04 and lacks about a fifth of the delisted tickers Sharadar lists, so `full`
  is still not complete before 2003.
- Some bad prints remain in small stocks (e.g. a −87% / +808% pair); the clipped sensitivity shows their effect.

## Testing Requirements
- [x] Delisting classification and haircut
- [x] Panel month rows and terminal flags
- [x] Strategies, costs, period statistics, bias view

## Documentation Updates
- [ ] Record the outcome in the project memory and in the strategy notes it changes
