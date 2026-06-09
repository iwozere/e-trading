# Design

## Purpose

Identify when institutional investors ($1B+ AUM) are systematically reducing
equity positions, generating avoidance and contrarian-entry signals for the
portfolio.

## Architecture

### Three-Layer Signal Model

```
Layer 1 — Quarterly (13F, updated daily during filing windows)
  ┌─────────────────────────────────────────────────────────┐
  │ EdgarDownloader.download_13f_index()                     │
  │   ↓ EFTS search → all 13F-HR filers for quarter         │
  │ EdgarDownloader.download_13f_infotable()                 │
  │   ↓ XML → CSV.gz per (CIK, quarter)                     │
  │ OpenFigiMapper.map_cusips()                              │
  │   ↓ CUSIP → ticker mapping (persistent CSV.gz cache)    │
  │ PositionDeltaCalculator.calculate()                      │
  │   ↓ Q vs Q-1 delta per (institution, ticker)            │
  │ ExitScreener.screen()                                    │
  │   ↓ filter: reduction ≥30%, position ≥$25M or ≥0.5%    │
  │ ConsensusDetector.detect()                               │
  │   ↓ aggregate: stocks exited by ≥3 institutions         │
  └─────────────────────────────────────────────────────────┘

Layer 2 — Near-real-time (Form 4 + 13D/G, filed within 2 business days)
  ┌─────────────────────────────────────────────────────────┐
  │ Form4Monitor.get_significant_sells()                     │
  │   ↓ insider open-market sales ≥$500K                    │
  │ Form4Monitor.get_13dg_drops()                           │
  │   ↓ Schedule 13D/A and 13G/A amendments                 │
  └─────────────────────────────────────────────────────────┘

Layer 3 — Daily (volume)
  ┌─────────────────────────────────────────────────────────┐
  │ VolumeAnomalyDetector.detect()                          │
  │   ↓ DataManager OHLCV → spike_ratio > 3.5×             │
  └─────────────────────────────────────────────────────────┘

All three layers → CompositeScorer.score() → alerts (score ≥ 60)
                                           → Telegram via scheduler
```

### Data Flow

1. Daily job starts at 07:00 UTC
2. `get_new_13f_filings_today()` — EFTS query for today's 13F-HR filings
3. If new filings → download infotables → compute deltas → update consensus CSV.gz
4. If no new filings → load cached consensus from prior run
5. Download Form 4 and 13D/G for yesterday
6. Collect watchlist tickers (consensus ∪ Form 4)
7. Run volume anomaly on watchlist (capped at 200 tickers)
8. Composite score → write `results/p18_institutional_flow/<date>/signals.csv`
9. Return result dict → scheduler evaluates `notification_rules`

### Cache Strategy

- **Per-institution 13F holdings**: immutable once written, keyed by (year, quarter, CIK)
- **Consensus CSV.gz**: overwritten each run with new filings added to the aggregation
- **OpenFIGI CUSIP map**: append-only, each CUSIP resolved once forever
- **Form 4 / 13D/G**: one file per calendar day, written once (force=False by default)

## Design Decisions

### Why CSV.gz over DuckDB?
Consistent with all other downloaders in `src/data/downloader/`. No extra
dependency; files are readable by any tool; easy to inspect/debug.

### Why EFTS search over form.idx?
EFTS returns JSON with clean entity metadata (CIK, institution name, period).
form.idx is fixed-width text that requires brittle column-offset parsing.

### AUM filter ($1B) applied at download time
Portfolio value is computed from the infotable sum. Institutions below $1B are
skipped immediately after parsing, saving storage and compute.

### Daily run on non-filing days
Layers 2 and 3 (Form 4, volume) deliver value year-round. The pipeline never
sits idle between 13F filing windows.
