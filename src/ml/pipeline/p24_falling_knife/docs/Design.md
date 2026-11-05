# Design

## Purpose
Test, then (if it passes) run, a rule-based version of the chapter-11 falling-knife reversal method. The binding
definitions are in `pipeline-specification.md` §4; this file records how the code is organized and why.

## Architecture

```
FMP vintage ──► data.py ──► universe.py (eligibility) ──► indicators.py ──► geometry.py ──► detector.py ──► exits.py
 (raw bars,      (split-adjust,                                 (ADX/DI/CCI  (ZigZag, P1,      (conditions,     (trade
  events, flags)  raw cols, weekly)                              TA-Lib; PPO  low since P1, W)  state machine)   sim)
                                                                 own EMA)
backtest.py: chunks → process pool → per security: one context per (zz_pct, support_bounce_pct), every variant walks it
validation.py / report.py: metrics, month-block bootstrap, controls, cost repricing, gate → report.md
```

### Component Design
- **`SeriesContext`** holds every array of one security/timeframe. Built once, shared by all ablation variants (they
  only change `require_*` switches and exit settings).
- **`evaluate_bar`** is pure: bar `t` and the context in, a `BarEval` with every condition and metric out. Only bars
  `<= t` are read. `BarEval.failed(params)` lists missing conditions, which feeds the Phase 2 "forming" watchlist.
- **`detect`** walks only candidate bars (eligible ∧ drop ≥ 40% ∧ fresh low), with a small state machine for
  one-signal-per-knife and the re-arm rule.
- **`simulate_trade`** enters next open, applies exits in priority order, fills next open, costs tiered by the
  as-traded price (`raw_scale`).

## Data Flow
- Input: vintage `1d_raw` + splits/dividends → split-adjusted OHLCV with `open_raw/close_raw/volume_raw` and
  `jump_no_split`. Weekly = W-FRI resample labelled with the week's last session.
- Output: one trade row per signal per variant (spec §6.2 columns plus SPY excess), random-control draws, survivorship
  table, manifest, report.

## Design Decisions
- **Causality first.** ZigZag swings exist only from their confirmation bar; rolling windows are trailing; the truncation
  test (`test_truncation_does_not_change_past_signals`) guards it.
- **Channel width normalization.** `W = (1 − ADX/maxADX) + (1 + PPO/|minPPO|)` over the PPO context window. It is 0 when
  both indicators sit at their window extremes at the same bar, so on its own it rewards *coincidence* of the two
  extremes; the depth rules (ADX ≥ 30, PPO lower third) supply the absolute part.
- **Second leg window.** After a re-arm, the ADX/PPO/W extremes are measured from the widest channel point since the
  re-arm. Measuring from the re-arm bar picked up the first leg's decaying ADX (found while building the fixture).
- **Raw price for price floor and costs.** Split-adjusted history of a stock that later reverse-split looks far
  pricier than it traded; floor and cost tier use `close_raw` / `raw_scale`.
- **`JUMP_NO_SPLIT` is not enough to call a bar bad.** FMP sets it when the close ratio is near a split ratio; real
  −33%/−50% crash days match too (ADEA, ACAD, AGTC at 20–35× volume). A jump counts as suspect only on volume < 3× the
  prior 20-day median; a quiet ≥ 30% fall (< 1.5× volume) also counts. Suspect trades are reported, not used.
- **Event-driven, not vectorbt.** Path state (swings, re-arm, shakeout) makes `from_signals` a poor fit; numpy
  pre-computation plus a candidate-only loop is fast enough (~0.08 s per security for 8 variants).
- **Month-block bootstrap.** Signals cluster in crashes (2008-10, 2020-03); resampling single trades would overstate
  confidence.
- **Coverage proxy.** The vintage cannot say what it is missing; coverage = year's delisting rate / median rate of
  2021–2025.

## Integration Patterns
- Read-only use of the FMP vintage through `FmpStore`; no writes outside `results/p24_falling_knife/`.
- Phase 2 will publish `results/p24_falling_knife/YYYY-MM-DD/signals.csv` for the signal ledger.
