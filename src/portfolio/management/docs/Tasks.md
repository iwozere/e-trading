# Tasks

## Implementation Status

### COMPLETED FEATURES
- [x] Package scaffolding (`__init__.py`, README, docs, config YAML)
- [x] Config dataclass + YAML loader (`config.py`)
- [x] Pure earnings-trigger window math (`earnings_window.py`)
- [x] Earnings-date source wrapping P05's `EarningsCalendar` (`earnings_source.py`)
- [x] Pure stop-loss coverage classification (`coverage.py`)
- [x] Live (read-only) IBKR open-orders fetch (`open_orders.py`)
- [x] Notifier (combined Telegram + Email digest) (`notifier.py`)
- [x] Orchestrator `runner.run_once` + CLI (`runner.py`, `cli.py`, `__main__.py`)
- [x] Scheduler integration: `portfolio.management` dispatch branch +
      `seed_schedule.py`
- [x] New `IBKR_LIVE_STOP_GUARD_CLIENT_ID` env var (config/donotshare/donotshare.py)
- [x] Unit tests for every pure module + an integration-style `runner` test
      suite using fakes for the earnings source, open-orders feed, and
      notification client (36 tests total)
- [x] `pyright` + `mypy` clean

### PLANNED ENHANCEMENTS
- [ ] Session (BMO/AMC) detection — extend `EarningsCalendar` (or add a
      sibling method) to surface FMP's session field if/when verified to be
      present on the plan in use; today every event is `session="unknown"`
      and anchors to market open (see `Requirements.md` "Known Limitations").
- [ ] Re-alert policy beyond "every time" — first-crossing dedup, or a
      snooze/ack mechanism, once it's clear whether the current behavior is
      too noisy in practice (explicitly deferred by the user).
- [ ] Consider a second earnings-date source (e.g. Finnhub, matching P20
      Kestrel's) as a cross-check if FMP's calendar proves unreliable for
      specific tickers.

## Technical Debt
- None yet — this is a new module.

## Known Issues
- Session (BMO/AMC) timing is not sourced from FMP today — see
  `Requirements.md` "Known Limitations". Not a correctness bug (the default
  anchor is the safer, earlier one), but triggers for AMC-reporting tickers
  fire earlier than their "true" T-1day/T-1hour would be.
- Local dev/test note: this environment has real `IBKR_FLEX_TOKEN` /
  `IBKR_FLEX_QUERY_ID` configured, so any test exercising the holdings
  pipeline **must** patch out `download_open_positions_xml` (see
  `tests/test_runner.py`'s `_no_live_flex_download` autouse fixture) —
  otherwise it silently overwrites the tmp_path fixture XML with a real
  Flex Query download of the live account before `load_ibkr_xml` reads it
  back. Caught during development; worth remembering for any new test file
  in this module.

## Testing Requirements
- [x] Unit tests for `earnings_window` (anchor resolution incl. DST,
      trigger-window matching)
- [x] Unit tests for `coverage` (classification, capping, missing-symbol
      handling)
- [x] Unit tests for `earnings_source` (empty input, mapping, failure)
- [x] Unit tests for `open_orders` (side/type filtering, partial-fill
      remaining-qty preference, malformed-trade skipping, connect failure)
- [x] Unit tests for `notifier` (plain-text + HTML formatting, zero-reminder
      case, unknown-session omission)
- [x] Integration-style tests for `runner.run_once` (no-trigger exit,
      T-1day/T-1hour trigger firing, ticker-not-held filtering, no-holdings
      early exit, live-IBKR-unreachable still notifies)
- [ ] Integration smoke test against a live IBKR account (manual, once the
      Master API Client ID is configured — see `README.md` "Quick Start")
