# Requirements

## Python Dependencies

All dependencies are already present in the project:
- `requests` — SEC EDGAR and OpenFIGI HTTP calls
- `pandas` — DataFrame processing
- `xml.etree.ElementTree` — 13F infotable XML parsing (stdlib)

No new packages required.

## External Dependencies

| Module | Purpose |
|--------|---------|
| `src.data.downloader.edgar_downloader` | 13F, Form 4, 13D/G EDGAR downloads |
| `src.data.downloader.openfigi_mapper` | CUSIP → ticker resolution |
| `src.data.data_manager` | OHLCV for volume anomaly detection |
| `src.notification` | Telegram alert delivery |
| `src.scheduler` | `DATA_PROCESSING` job execution |

## External Services

| Service | Purpose | Cost |
|---------|---------|------|
| SEC EDGAR | 13F, Form 4, 13D/G filings | Free |
| OpenFIGI API | CUSIP → ticker mapping | Free (keyed: 250 req/min; unkeyed: 25 req/min) |
| Yahoo Finance (via DataManager) | OHLCV for volume anomaly | Free |

## Configuration

Optional: add `OPENFIGI_API_KEY=<your_key>` to `.env` for higher rate limits.
Free-key registration: https://www.openfigi.com/api#req-form

## System Requirements

- Disk: ~500 MB for a full 13F history backfill (2 years, all institutions >$1B AUM)
- Typical daily run: <50 MB download, <2 min wall time outside filing windows
- During filing windows (Feb 1–15, May 1–15, Aug 1–15, Nov 1–15): up to 15 min
- Network: must reach `data.sec.gov`, `efts.sec.gov`, `api.openfigi.com`
