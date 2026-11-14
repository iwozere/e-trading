# Runbook — Sharadar access window

All commands run from the project root with the project `.venv`. `CLI` below means
`.venv/Scripts/python.exe -m src.data.sharadar_pipeline.cli`.

## EODHD check (separate purchase, do it first)

Goal: find out whether EODHD's delisted price history is deep enough to be the kept price layer
([Findings](Findings.md) §7). Sharadar's free ticker list is the yardstick, so no Sharadar subscription is needed.

1. Buy one month of EODHD "EOD Historical Data — All World" ($19.99 on 2026-11-14) on the existing EODHD account,
   so the key in the project config stays the same. Cancel the renewal right away; access runs to the end of the month.
2. `CLI download --sample` (fetches Sharadar's free ticker list).
3. `CLI eodhd-check` — 100 delisted common stocks from each of six periods, about 700 API calls of the 100,000 a day.
   `--all` checks all 12,700 or so.
4. Read the summary it logs (also saved to `<DATA_CACHE_DIR>/eodhd-check/summary.csv`):

   | Column | Meaning |
   |---|---|
   | `found` | EODHD lists a delisted common stock with that ticker |
   | `full` | Its bars start and end within 10 days of Sharadar's dates and cover at least 90% of sessions |
   | `full_name_ok` | Full history and the company name matches: the number to trust |
   | `partial` / `none` | Some bars in the range / none |
   | `fmp_close_ok` | Where the FMP vintage has the stock: share of pairs whose closes agree on 99% of days |

5. Exit code 2 and a warning mean EODHD cut the history to a plan limit: the key is not on the paid plan yet.
6. Decision rule: `full_name_ok` of 0.8 or more in the 1998–2020 buckets makes EODHD worth downloading in full
   and keeping. Below about 0.5 it is not a usable survivorship fix.

The detail file `reports/eodhd_check.csv` contains Sharadar ticker rows and stays in the Sharadar data root.

## Before buying Sharadar

1. The deletion rule is accepted ([Findings](Findings.md), section 3).
2. The dry run on the free sample is green: `CLI download --sample`, `CLI build`, `CLI crosscheck`,
   `CLI purge --confirm`.
3. Research queue frozen and its scripts written ([Tasks](Tasks.md)).
4. Two free weeks ahead: the window is short and should not start before a holiday or a busy period.
5. Check the price on https://sharadar.com/subscribe (Bundle, Full History, monthly: $69 on 2026-11-14).

## Day 0 — purchase

1. Subscribe as a private person: Bundle, **Full History**, monthly. The lower "from" prices are 5-year history.
2. Copy the key from https://sharadar.com/account into `.env` as `SHARADAR_API_KEY`.
3. Set a calendar reminder to cancel before day 30, and one for the deletion deadline (end of subscription + 30 days).
4. Check access and file sizes without downloading: `CLI bulk-status`.
5. Confirm the full history is served: a delisted pre-2021 ticker must return rows, for example

   ```bash
   curl "https://api.sharadar.com/v1.0/data/stocks?api_key=$SHARADAR_API_KEY&ticker=LEH&from=2008-01-01&to=2008-12-31"
   ```

6. Download everything: `CLI download` (resumable; add `--subscription-start YYYY-MM-DD` if the plan did not start today).
7. Build the store: `CLI build`, then `CLI crosscheck`.
8. `CLI status` shows the files and the deletion deadline at any time.
9. Download a second time near the end of the month only if a fix was announced; otherwise one pull is enough.

## Day 0–1 — checks before any research

| Check | Pass condition |
|---|---|
| Row and ticker counts | Match the `tickers` table (21,037 price tickers on 2026-11-14, give or take new listings) |
| Earliest price date | 1997-12-31 |
| Delisted stocks per year | 250–900 in every year 1998–2020 |
| Raw close vs FMP raw close on the overlap | Median ratio 1.00; list the outliers (`reports/vendor_check.csv`) |
| Splits vs FMP splits on the overlap | Same dates and ratios; list the differences |
| Our total-return adjustment vs `closeadj` | Agreement within 1% (`reports/adjustment_check.csv`) |
| Fundamentals filing date vs FMP `filing_date` | Same date for most filings; list the differences |
| Point-in-time | No MR dimension in the store; no row used before `available_from` |

The check reports stay in the data root.

## Day 1–55 — research

Run the queue in order. Each item writes aggregates only to `results/`.

## Cancel

Cancel in the account before day 30. Cancelling stops future billing; nothing already paid is refunded.
Record the last paid day: `CLI set-end YYYY-MM-DD`. The deadline in `CLI status` is that day plus 30.

## Deletion deadline (end of subscription + 30 days)

1. Run `CLI purge` (dry run), then `CLI purge --confirm`, on the dev machine and on the Pi if it holds a copy.
2. Delete the data root from every backup disk.
3. Remove `SHARADAR_API_KEY` from `.env`.
4. Check that `results/` holds no per-ticker-per-day tables derived from Sharadar.
5. Keep the purge report (written next to the data root) in case Sharadar asks for confirmation of deletion.

## Never

- Commit Sharadar data, the key, or the cross-check report to this (public) repository.
- Copy Sharadar rows into the FMP vintage or any database.
- Share the key or the files.
