# FMP data pipeline: owner runbook

These are the steps from buying FMP Premium to cancelling it, the routine after that, and the repeat dump in 2 to
3 years (spec §12.2). Commands are PowerShell, run from the project root on the laptop, unless marked
**[Pi]**. The schema and read semantics are in [storage_schema.md](storage_schema.md), every design decision is in
[decisions.md](decisions.md), and the full specification is
[FMP_DATA_PIPELINE_SPEC-2.md](FMP_DATA_PIPELINE_SPEC-2.md).

```powershell
function fmp { & .venv\Scripts\python.exe -m src.data.fmp_pipeline.cli @args }   # then: fmp status
```

Every command can be re-run, and a stopped command continues where it stopped. Exit codes:

| Code | Meaning |
|---|---|
| 0 | Done |
| 1 | Stopped on purpose (quota, auth, bandwidth cap, archive problem) or build rejected |
| 2 | Usage error, lock held, or nothing planned |
| 3 | Not enough disk space |

## 0. Machines and one-time setup

| Machine | Role | Runs |
|---|---|---|
| Windows laptop | Writer of the FMP dump | `probe-*`, `plan`, `run`, `build`, `validate`, `verify`, `backup`, `publish` |
| Raspberry Pi | Reader, and the increment writer (Q4) | `sync`, `ingest-increments`, `snapshot-universe`, `archive-increments` |
| USB HDD | Backup volume, connected by hand (Q3') | target of `backup` and `verify --final --backup` |
| `R:` | The Pi's SSD mapped on the laptop: the hand-off folder (Q3'') | target of `publish` |

**Laptop setup:**
- Install the pipeline profile: `.venv\Scripts\python.exe -m pip install -r requirements-fmp-pipeline.txt`.
- `data_root` defaults to `DATA_CACHE_DIR\fmp-data`. Keep the path short, because some archive paths are long
  (an alternative is enabling Windows long paths).
- Recommended, your choice: exclude `data_root` from real-time antivirus scanning and from search indexing. Both
  can hold files open and slow down or interrupt atomic renames (the code retries).
- The pipeline keeps the laptop awake during `run` and `build`. Windows Update restarts can still interrupt a
  run; just start it again.

**Pi setup [Pi]:**
- Install the reader profile: `pip install -r requirements-fmp-reader.txt`. Check that the `duckdb` aarch64 wheel
  installs.
- Set `fmp.increments.writer_host` in `config/fmp/fmp.yaml` to the Pi's host name (run `hostname` on the Pi).
  Every other machine then refuses to ingest increments.

**API key.** It is read from `FMP_API_KEY` in the environment, or from `config/donotshare/donotshare.py`. It is
sent only in a request header. Never put it in `fmp.yaml`, a URL, a log or a ticket.

## 1. Before buying (gate G1)

All of this is done for the 2026-10 dump; repeat it for a later dump (section 7).

1. **Re-check the facts in spec §3 on FMP's pricing page:**
   - price;
   - the 50 GB per 30 days cap;
   - 750 requests per minute;
   - **whether `1hour` charts are in Premium**;
   - delisted coverage.

   The API cannot show any of these before purchase.
2. **Read** `docs/fmp_probe_report.md`, the free-key findings and the `[VERIFY]` audit.
3. **Dry-run the plan:** `fmp plan --dry-run`. Read `reports/<dump_id>/budget_plan.md` and
   `universe_report.md`. With the free key the universe is incomplete; the real plan comes after the paid probe.
4. **Free disk space:** at least 100 GB on the laptop's working volume and 40 GB on the backup HDD. `fmp plan`
   prints the projected need next to the free space.
5. **Ask whether a month will be attended:** the run is unattended, but G2 and G3 need you.

The agent never buys, cancels or changes the FMP account (D10).

## 2. Purchase day: paid probe (Phase 2, gate G2)

1. Put the Premium key in place (section 0).
2. Run `fmp probe-paid`. It takes about 100 calls and a few MB. Responses are cached under
   `probe-paid/`, so re-running costs nothing, and `--offline` rebuilds the report from the cache.
3. Read `docs/fmp_paid_probe_report.md`. Its measured settings, such as the intraday row cap, time zone, label
   and page size, are written to `probe-paid/probe_results.json`. Every command applies them over `fmp.yaml`
   (`use_probe_results: false` switches this off).
   - **If `1hour` is not served, stop.** Nothing substitutes another interval (spec §6.3.6). The choices are
     Ultimate, building hours from `30min` or `15min` (more traffic), or no intraday.
   - Reused tickers: if the report shows FMP merging different companies under one symbol, set
     `build.segment_reused_tickers: true` (D46).
4. Run `fmp plan`. This writes the tasks and `budget_plan.md` with the real universe.
5. **Gate G2: approve the plan.**
   - Check the projected GB per block and tier under both counting bases (wire and decoded).
   - Check what fits under 0.90 × 50 GB.
   - Check what is left for a possible second month.
   - Edit the tiers in `fmp.yaml` and re-plan if needed.

## 3. Download (Phase 3)

```powershell
fmp run                       # everything, in spec order: reference, events, index/macro, daily, recon,
                                     # fundamentals, then 1h by tier; delisted before active in every block
fmp run --phase 3             # one block only (0-6, see budget_plan.md)
fmp run --tier A              # one intraday tier
fmp status                    # progress, rolling 30-day GB against the cap, ETA, top failures
fmp retry-failed              # re-queue failed/retryable tasks, then run again
```

Every run prints a progress line once a minute: tasks done, requests per minute, pending tasks, ETA, and
bandwidth against the hard stop. It also writes `logs\<dump_id>\run-<UTC start>.jsonl`: one JSON record per
request (endpoint, symbol, status, rows, bytes, duration, error) plus the progress records. To see what failed,
for example:

```powershell
$root = "d:\data-cache\fmp-data"   # your data_root
Get-Content (Get-ChildItem $root\logs\2026-10\run-*.jsonl | Select-Object -Last 1) |
  ConvertFrom-Json | Where-Object { $_.event -eq "request" -and $_.status -ne "ok" } |
  Group-Object endpoint, status | Sort-Object Count -Descending
```

- **Interrupted** (crash, restart, Ctrl+C): run the same command again. The manifest resumes, and nothing is
  duplicated or lost.
- **Stops on its own** (exit 1), each with a clear log message:
  - FMP says the quota is reached;
  - the key is rejected;
  - the archive disk disappears;
  - the rolling 30-day traffic reaches 0.90 × cap.

  At the cap, wait for the window to roll, or decide on a second month.
- **After the first few GB**, read the bandwidth figure on the FMP dashboard and run
  `fmp calibrate-usage --dashboard-gb 3.2`. If the figure matches compressed bytes, the basis switches to
  `wire` and about 7× more fits (probe F10). If the result is not clear-cut, nothing changes.

## 4. Build and validate (Phase 4)

```powershell
fmp build                     # raw archive -> vintages/<dump_id>/, then switches current.txt
fmp validate --cache-dir d:\data-cache   # -> reports/<dump_id>/validation_report.md/.json, coverage_report.md
```

`--cache-dir` points at the legacy yfinance cache for the V-YF-XCHECK comparison. Without it, that check is
skipped.

- `build` reads only the archive, so it can run while you are offline. It builds in `vintages/.<dump_id>.building`.
  If structural validation fails, it keeps the result as `.<dump_id>.rejected` for inspection, and the previous
  vintage and `current.txt` stay as they were. `--no-switch` builds without switching.
- **Review:**
  - every `FAIL` (blocks G3) and `WARN` in `validation_report.md`;
  - `missing_splits_candidates.csv` (probable missing splits; nothing is inserted automatically);
  - `identity_uncertain.csv`.
- **Calibrate the flag thresholds** (`fmp.build.flags`) on the real data if the flag rates look wrong, record
  the choice in `decisions.md`, and rebuild. The archive is the source; rebuilding costs no FMP traffic.

## 5. Hand the data to the Pi and turn it on (Phase 5)

```powershell
fmp golden compare                        # DataManager output unchanged with the adapter still off
fmp publish --to R:\fmp-handoff           # packs the vintage onto the Pi's SSD, verifies, writes READY
```

On the Pi:

```bash
python -m src.data.fmp_pipeline.cli sync --from /path/to/fmp-handoff    # [Pi] verify, unpack locally, switch
```

`sync` picks the newest `READY` publication. It re-checks every file after unpacking and only then switches
`current.txt`, so a running strategy keeps reading the old vintage until the switch.

**Then switch on:**
1. Set `fmp.adapter.enabled: true` in `config/fmp/fmp.yaml` on each machine that should read FMP data
   (`docs/datamanager_integration.md`).
2. **[Pi]** Run `ingest-increments --limit 20` once as a trial, then the full `ingest-increments`.
3. Enable the scheduled jobs on the Pi only. "FMP Increments Ingest" runs 23:00 UTC Mon–Fri and "FMP Universe
   Snapshot" runs monthly. Both are registered disabled in `src/data/pipeline/specs/fmp_specs.py`; enable them in
   the Pi's `job_schedules` (see `src/data/pipeline/README.md`).

## 6. Gate G3: before cancelling

1. Connect the backup HDD. Then run:

   ```powershell
   fmp backup --to E:\fmp-backup            # packed tars + checksums, verified member by member there
   fmp verify --final --backup E:\fmp-backup --confirm-postponed
   ```

2. `verify --final` runs the spec §14 checklist:
   - manifest complete;
   - no FAIL validators;
   - archive checksums and a full decompression test;
   - the backup checked file by file;
   - V-REPRO rebuild;
   - adapter tests;
   - free space.

   It then marks the raw archive read-only. `--confirm-postponed` is your confirmation that anything not
   downloaded (listed with its estimated GB) is intentionally postponed or abandoned.
3. **Only when it prints `READY TO CANCEL`**, cancel the subscription yourself on the FMP site.
4. Keep the HDD backup offline. A second copy of `raw/` on the external SSD is optional.

## 7. Routine after cancelling

| What | Where | How often | Why |
|---|---|---|---|
| `ingest-increments` | Pi (scheduled) | Mon–Fri | Keeps current symbols up to date, and stores a symbol's history before it can be delisted (spec §10.2.7) |
| `snapshot-universe` | Pi (scheduled) | Monthly | Records listings and removals, and last-downloads removed symbols (no FMP key) |
| `backup` of new snapshots/increments | Laptop | When convenient | The Pi's SSD is the only copy otherwise |
| Check the ingest log for `1h gap` errors | Pi | Monthly | yfinance keeps only about 730 days of hourly data. A warning starts at 700 days of gap, and older hours are lost for good. |

If the Pi is down for a long time, run `ingest-increments` as soon as it is back. Daily bars catch up completely,
even across a split; hourly bars only for the last ~730 days.

## 8. Repeat dump (about 2029)

Use a new `dump_id`, for example `2029-03`. Nothing from the old dump is overwritten: the old raw archive is
read-only, and the old vintage stays until you delete it.

1. **Pre-purchase:**
   - Set `fmp.dump_id` in `config/fmp/fmp.yaml`.
   - Re-check spec §3 against FMP's current plans (section 1).
   - Re-run the free probe after renaming the old `probe-free\` folder, so that both the 200-call budget and the
     cached responses start fresh: `fmp probe-free`. Compare the report with the 2026 one.
   - Rename the old `probe-paid\` folder too, so that its `probe_results.json` is not applied to the new dump.
   - Update the code first if endpoints or fields changed.
2. **Purchase and download:** sections 2 to 3 unchanged.
3. **Bring the Pi's increments to the laptop** for the drift report:

   ```bash
   # [Pi] use a folder of its own, not the laptop's publish folder (the two would share a dump id)
   python -m src.data.fmp_pipeline.cli publish --to /path/to/fmp-outbox --increments
   ```

   ```powershell
   fmp sync --from R:\fmp-outbox --increments --no-switch   # laptop: increments only, current.txt untouched
   ```

4. **Build:** `fmp build`.
   - Security ids carry over through `security_ids.json`, so the same company keeps its id.
   - Because increments from the old vintage exist, `build` also writes `reports/<new>/drift_report.md/.json`
     (D73): close differences, missing bars and event mismatches between yfinance and the new FMP data.
5. **Validate** and pass G3 (sections 4 and 6). `validate` and `verify` check the active vintage, which `build`
   has just switched to the new one.
6. **Promote:** `publish --to R:\fmp-handoff` on the laptop, then **[Pi]** `sync --from …`. `current.txt` now
   names the new vintage, and readers switch on their next start.
7. **Retire old increments [Pi]:**
   `python -m src.data.fmp_pipeline.cli archive-increments`. Bars and events the new vintage now covers move to
   `increments/archive/<old>/` and `increments/archive/before-<new>/`. Nothing is deleted, and increments of
   securities missing from the new vintage stay in place.
8. Delete the old vintage only when you no longer need to reproduce results based on it. Keep its raw archive
   and HDD backup.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `lock held` (exit 2) | Another command on the same dump, the increments or a sync is running. Wait for it. A stale lock left by a crash is released by the OS. |
| `FMP_API_KEY is not set` | Section 0. `plan --offline` and `probe-* --offline` work without a key. |
| `run` exits 1 straight away | Read the last log line: quota reached, key rejected (401), bandwidth cap, or the archive disk is missing. |
| Build rejected | Inspect `vintages\.<dump_id>.rejected` and the logged structural errors. The previous vintage is still active. |
| `sync` rejected | The publication failed verification (damaged copy). Run `publish` again on the laptop. Nothing was switched. |
| `ingest-increments` refuses | This host is not `writer_host`. Correct, unless this is the Pi with a changed host name. |
| `PermissionError` on rename (Windows) | Antivirus or the indexer is holding the file. The code retries; exclude `data_root` (section 0). |
| DataManager output changed | Run `golden compare` (adapter off) to see which series changed. Set `fmp.adapter.enabled: false` to go back to the legacy path immediately. |
