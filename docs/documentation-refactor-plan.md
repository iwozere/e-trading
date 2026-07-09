# Documentation Refactor — Execution Plan

Status: **PLAN ONLY** — no docs have been edited or deleted yet. This document is
the playbook for the follow-up sessions that will do the actual rewrite/prune.

Prepared: 2026-07-09.

---

## 1. Why this is a plan, not a first pass

There are **39 `docs/` folders** and **~250 files** (root `docs/` + every `src/*/docs/`
and `src/*/*/docs/` submodule folder, per the structure mandated in
`.claude/CLAUDE.md` §10/§14). Reading and judging every file in one sitting isn't
workable in a single context — this plan defines the rubric, the batching
strategy, and the concrete problems already found by recon, so the execution
sessions can move fast and consistently instead of re-deriving methodology each
time.

---

## 2. Recon findings (drive the plan's priorities)

A structural scan (file counts + `git log` dates + non-`.md` file listing) surfaced
four distinct problem classes, **not just "content is stale"**:

### 2a. Code/data files misplaced inside `docs/` folders
`docs/` is supposed to hold `README.md` / `Requirements.md` / `Design.md` / `Tasks.md`
(§13 templates), but several folders have accumulated **runnable code, notebooks,
SQL dumps, and result artifacts**:

| Folder | Misplaced files |
|---|---|
| `docs/examples/` | 27 standalone `*.py` example scripts (this is a whole example-app folder living under `docs/`) |
| `src/notification/docs/` | `archival_example.py`, `health_integration_example.py`, `utilities/*.py` (5 scripts + `__init__.py`) |
| `src/trading/docs/` | `demo_bot_config_integration.py`, `demo_mock_file_trading.py`, `todo.txt` |
| `src/data/db/docs/` | `extract_schema_as_md.py`, `db_schema.sql` |
| `src/ml/pipeline/p10_emps3/docs/` | `threshold_calibration.ipynb` |
| `src/ml/pipeline/p20_kestrel/docs/` | `manual_run.txt` |
| `src/strategy/docs/` | `results/*.json` (a committed backtest output file) |

These need a **relocate-or-delete** decision per file (e.g. `utilities/*.py` likely
belongs under `src/notification/` proper as real modules with tests; one-off
`results/*.json` is probably just noise to delete) — this is a different action
than "rewrite the prose," so it's tracked as its own checklist (§6).

### 2b. Committed `__pycache__` inside `docs/`
Because code lives in `docs/`, its bytecode cache got committed too:
`docs/examples/__pycache__/`, `src/notification/docs/__pycache__/`,
`src/notification/docs/utilities/__pycache__/`, `src/trading/docs/__pycache__/`.
Pure hygiene fix — delete + confirm `.gitignore` covers `**/__pycache__/` (it
likely already does for `src/`, but evidently not for `docs/`, or these predate
the ignore rule).

### 2c. Doc sprawl beyond the standard 4-file template
Several pipelines have accumulated ad hoc reports alongside the standard
`README/Requirements/Design/Tasks` set, with unclear ongoing relevance:
- `p03_cnn_xgboost/docs/`: 8 files incl. `Architecture_Recommendations.md`,
  `Data_Format_Guide.md`, `Data_Scaling_Strategy.md`, `Scaling_Summary.md`
- `p06_emps2/docs/`: 9 files incl. `CACHE_AND_CHECKPOINT.md`, `TIMING_ANALYSIS.md`,
  `TRADING_PLAYBOOK.md`, `TODO.md`, `refactoring.md`
- `src/notification/docs/`: 10 files incl. 5 different `README_*.md` variants
  (`README_analytics`, `README_channels`, `README_CLIENT`, `README_delivery_history_api`,
  `README_fallback_system`, `README_migration`) plus `OPTIMIZATION_SUMMARY.md`

These are prime candidates for **consolidation** (fold still-true content into
`Design.md`/`README.md`, delete the rest) rather than 1:1 update.

### 2d. Git-date staleness signal is noisy — don't rely on it alone
`git log` dates cluster heavily at **2025-10-30** (64 files) — almost certainly a
squash/import commit, not real authorship date. Absolute git age is therefore a
**weak** signal. The reliable signal is *relative* staleness: compare a doc's last
touch against its **sibling source code's** last touch, and — more importantly —
whether the doc's factual claims (function names, file paths, "planned" vs
"done" status, config values) still match the code. Recent memory already flags
known status mismatches to check for specifically:
  - P15 GDELT: docs should say **deferred**, not "planned/next."
  - P17: docs should say **paused** (2026-06-28, resume ~Sept, spec §15.2).
  - P19: docs should say **Phase 1 built, paused for data** (resume ~late July, spec §19).
  - P20 Kestrel: docs should say **fully implemented**, deploy steps `migrate → register_jobs → enable jobs`.
  - `docs/SUBMODULES.md` and `docs/CHANGELOG.md` (both touched 2026-07-05) are
    likely the freshest, most-trustworthy cross-reference for "what's the module
    actually for" — use them as ground truth when judging older per-module docs,
    but still spot-check them since even 4-day-old docs can drift after
    fast-moving changes (e.g. this week's mypy-error fixes touched signatures
    across `tests, validators, indicators, trading service ids`).

---

## 3. Classification rubric

Every file gets exactly one verdict:

| Verdict | Meaning | Action |
|---|---|---|
| **KEEP** | Accurate, current, well-scoped | No change |
| **UPDATE** | Core content still relevant but has stale facts (old status, renamed functions, wrong paths, outdated config) | Rewrite the stale parts in place |
| **CONSOLIDATE** | Overlaps with another doc in the same folder (e.g. 5 `README_*.md` variants) | Merge surviving content into the canonical file, delete the rest |
| **DELETE** | Superseded, describes an abandoned approach, or is a point-in-time report with no lasting reference value (e.g. `*_SUMMARY.md`, `TIMING_ANALYSIS.md` from a one-off investigation) | Remove |
| **RELOCATE** | Not documentation at all (code, notebook, SQL dump, result artifact) | Move to an appropriate `src/` location (with tests, per CLAUDE.md conventions) or delete if it's throwaway output |

A file only earns **KEEP** if someone actually re-read it against current code —
"looks recent" is not sufficient given §2d.

---

## 4. Per-file verification method (what an executor session actually does)

For each doc file:
1. Read it.
2. Grep the referenced module for the specific claims: function/class names,
   file paths, CLI flags, config keys, "Status: planned/in-progress/done" markers.
3. Check the module's own `Tasks.md` "Implementation Status" checklist against
   what's actually in the code (things get built and the checklist doesn't get
   ticked, or get ripped out and the checklist doesn't get updated).
4. Check the memory/project context (this session already carries P15/P17/P19/P20
   status — future sessions should pull that from memory or ask) for any
   pause/deferred/resume state that must be reflected.
5. Assign a verdict + one-line reason.

This is naturally parallelizable per-folder, which is why execution is batched
(§5) rather than one giant linear pass.

---

## 5. Execution phases

Each phase below is a separate follow-up session/task. Phases 1–4 can run as
**parallel subagents** (one per batch) since folders are independent; a human
(or the orchestrating session) reviews the combined backlog before Phase 5
actually deletes/moves anything.

### Phase 0 — Repo hygiene (do first, ~15 min, no judgment calls)
- Delete all committed `__pycache__/` dirs under any `docs/` folder.
- Confirm/fix `.gitignore` so `**/__pycache__/` and `*.pyc` are excluded
  everywhere, not just under `src/`.
- Delete `src/strategy/docs/results/*.json` (committed backtest output, not docs).
- No content judgment needed — safe to do immediately, independent of the rest.

### Phase 1 — Root `docs/` (7 top-level files + `HLA/` 20 files + `examples/` 27 scripts)
This is the highest-visibility, most cross-cutting layer — do it before the
submodules so it can serve as an accurate map while judging module docs.
- Audit: `CHANGELOG.md`, `ROADMAP-2026.md`, `SUBMODULES.md`, `USER_GUIDE.md`,
  `monitoring-setup.md`, `security-audit-api.md`, `security-audit-summary.md`.
- Audit `HLA/` (16 architecture docs + 6 `.mmd` diagrams) for drift against
  current module boundaries (cross-check against the just-refreshed `SUBMODULES.md`).
- Decide the fate of `docs/examples/` as a whole: keep as a curated example-app
  (move under e.g. `examples/` at repo root, outside `docs/`), or prune to a
  handful of maintained examples and delete the rest — 27 example scripts is a
  lot to keep working as the API evolves.

### Phase 2 — Core/infra modules (parallel batch A, ~10 folders)
`src/api`, `src/data` + `src/data/db`, `src/common` + `src/common/sentiments`,
`src/notification`, `src/scheduler`, `src/trading`, `src/telegram` + `src/telegram/screener`,
`src/indicators`. These are shared infrastructure — highest blast radius if wrong,
so worth doing before the pipelines that depend on them.

### Phase 3 — ML pipelines p01–p20 (parallel batch B, split into two waves of ~10)
- Wave 1: p01–p10 (older/likely more stale — several are pre-2026).
- Wave 2: p11–p20 (newer; p17/p19/p20 need the pause/deferred/implemented status
  checked per §2d).
- `src/ml/pipeline/docs/` (the pipeline-level folder, not a specific `pXX`) audited
  alongside as the pipeline-family overview.

### Phase 4 — Remaining feature modules (parallel batch C)
`src/analytics`, `src/backtester/optimizer`, `src/portfolio/pnl_alert`,
`src/screeners`, `src/strategy`, `src/strategy_pack`, `src/vectorbt`, `bin/`.

### Phase 5 — Synthesis & cleanup
- Merge the per-phase backlogs into one master list (file → verdict → reason → action).
- Execute RELOCATE actions (move code out of `docs/` into real `src/` locations,
  add minimal tests per CLAUDE.md §9 if the relocated code becomes a real module).
- Execute DELETE/CONSOLIDATE actions.
- Execute UPDATE actions.
- Re-run the Phase 0 hygiene check (new `__pycache__` may appear from relocated code).

---

## 6. Deliverable format for each phase

Each executor session produces a table (in its own scratch note or PR description,
not a permanent repo file) with columns:
`file | verdict | reason (1 line) | action taken`
— then applies KEEP/UPDATE/CONSOLIDATE/DELETE/RELOCATE directly, since re-review
of a table before every single edit would be slower than just doing it and
flagging anything ambiguous.

**Ask before deleting**, in each phase, only for files where the executor is
genuinely unsure whether the content still has reference value (e.g. a design
doc for an approach that was replaced — might still be useful "why we didn't do
X" context). Everything else (dead pycache, stray result JSON, duplicate
README variants) can be deleted without a check-in.

---

## 7. Open questions before execution starts

- `docs/examples/`: keep as a maintained example app (and if so, where should it
  live — is `docs/` acceptable long-term, or should it move to repo-root `examples/`
  or `src/examples/`), or prune aggressively? This changes Phase 1 scope a lot.
- Notification `utilities/*.py` and `trading/docs/demo_*.py`: are these still used
  by anyone (ad hoc ops scripts), or safe to delete outright rather than relocate
  with tests?
- Should Phase 5's master backlog be committed to the repo (e.g. as a changelog
  entry) or is it disposable once the edits land?

---

## 8. Suggested order

Phase 0 → Phase 1 → (Phase 2, Phase 3, Phase 4 in parallel) → Phase 5.
Phases 2–4 don't depend on each other, only on Phase 1's refreshed `SUBMODULES.md`/
`HLA/` being available as ground truth to check individual module docs against.
