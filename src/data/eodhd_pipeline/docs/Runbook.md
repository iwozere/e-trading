# Runbook — EODHD purchase day

All commands run from the project root with the project `.venv`. `CLI` below means
`.venv/Scripts/python.exe -m src.data.eodhd_pipeline.cli`. Every command takes `--data-root`.

## Before buying (done 2026-11-22)

- Tests pass; mechanics ran on the free key in a scratch data root: universe 22,595 codes (16,721 delisted,
  5,874 live), 5 stocks downloaded, built, cross-checked (closes and volumes equal to the FMP vintage on 4 pairs).

## Purchase day

1. Buy one month of "EOD Historical Data — All World" ($19.99) on the existing EODHD account, so the key stays the
   same. Cancel the renewal right away; access runs to the end of the month. Nothing has to be deleted afterwards.
2. `CLI quota` — must say a paid plan and about 100,000 calls left. Costs no call.
3. **Depth check first** (Sharadar module, about 700 calls), so the $19.99 tells us something even if we stop here:
   `.venv/Scripts/python.exe -m src.data.sharadar_pipeline.cli download --sample` then
   `.venv/Scripts/python.exe -m src.data.sharadar_pipeline.cli eodhd-check`. See
   [its runbook](../../sharadar_pipeline/docs/Runbook.md). Go on if `full_name_ok` is 0.8 or more in the 1998–2020
   buckets; below about 0.5, stop: EODHD is not a usable survivorship fix.
4. `CLI universe` — 2 calls.
5. Trial: `CLI download --limit 30` — about 90 calls. Then `CLI build` and `CLI crosscheck` and read the summary
   (see below). Fix anything odd before spending the rest.
6. `CLI download` — the rest, about 68,000 calls. Resumable: if it stops (exit code 3, budget or quota), run it
   again, the next day if the quota is spent. Failed codes are retried on every run. `CLI status` shows what is left.
7. `CLI build`, then `CLI crosscheck`, then `CLI status` (shows the delisted stocks per period that the FMP vintage
   lacks).
8. Copy `<DATA_CACHE_DIR>/eodhd-data/` to the Pi and the backup disk, as with the FMP vintage.
9. Optional while the month lasts: a second `CLI universe` and `CLI download` near the end catches stocks that
   delisted in between (only new codes are downloaded).

## Reading the cross-check (`reports/crosscheck_summary.json`)

| Number | Expect | If not |
|---|---|---|
| `adjustment.share_fully_ok` | 0.9 or more | `worst` lists symbols and the day the ratio jumps: a missing or misplaced split or dividend there |
| `vendor.median_close_ratio` | 1.00 | Not 1: EODHD's close is not as traded for these stocks |
| `vendor.share_pairs_close_ok` | 0.9 or more | Look at the worst pairs in `vendor_check.csv` (often FMP's own split-basis problems) |
| `vendor.median_volume_ratio` | about 1.0 | About the split ratio: volume is not split-adjusted after all; set `VOLUME_SPLIT_ADJUSTED = False` and rebuild (no calls) |
| `splits_matched` / `splits_fmp` | most | Splits EODHD lacks show as jumps in P24's `jump_no_split` |
| `dividends_matched` / `dividends_fmp` | most | One-day offsets are ex-date conventions; large gaps need a look |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Done |
| 2 | The key is on the free plan or the plan cuts the history: nothing usable was downloaded |
| 3 | Stopped early on the call budget or the daily quota: run `download` again |

## Free-key dry run (mechanics only)

`CLI --data-root <scratch> download --codes AAPL KO --allow-limited`: one year of bars, no splits or dividends,
marked `plan_limited` in the progress file. Use a scratch data root; if it was the real one, the paid run needs
`download --force` for those codes (`status` warns).
