# Biotech M&A Target Screening Pipeline — Technical Specification

**Version:** 0.6 (price archive design added)
**Audience:** Backend / data engineer building the system
**Status:** Ready for estimation and scaffolding

**Changelog 0.5 → 0.6:** Added §2.0.7, price archive and corporate actions. Raw-price storage with read-time adjustment is now mandatory, replacing any implicit assumption that adjusted vendor series can be archived directly. Documents the market-cap corruption and level-lookahead failures that adjusted storage causes, and the yfinance hazards specific to a microcap universe.

**Changelog 0.4 → 0.5:** Added §2.0, universe construction — the spec previously described what to fetch per company without defining where the company list comes from. Universe is now built point-in-time from SEC DERA Financial Statement Data Sets, with a source capability matrix assigning explicit roles to EDGAR / IBKR / FMP / yfinance and an identified paid gap for delisted-ticker price history.

**Changelog 0.3 → 0.4:** Moved `fis_risk` and `antitrust_overlap` out of the target-level friction block and into `fit()` as multiplicative pairwise gates (§4.4.1, §4.4.2), so they affect the acquirer argmax instead of applying a flat per-company penalty. Restructured `fit()` to separate additive preference terms from multiplicative feasibility gates, resolving the v0.2 inconsistency where `size_feasibility` was described as a gate but written as an additive term. Added a single-valued-per-company invariant to Block E. Closed the §5.4 numbering gap.

**Changelog 0.2 → 0.3:** Disambiguated `score.rank` into `rank_by_expected_value` (default) and `rank_by_composite`, with §5.4 making the choice normative. Named the Block G latency tradeoff as an explicit risk. Added a CVR valuation policy pinned at M6 rather than M7. Squared the §6.1 worked example with §0.3.

**Changelog 0.1 → 0.2:** Added Block G (corporate process signals: strategic-alternatives disclosures, 13D activist stakes, incumbent-partner and option-to-acquire structures). Restructured PoA handling to be stage-conditional and separated `P(deal)` from `E[return | deal]`. Added equity-currency capacity to acquirer dry powder. Moved foreign-investment-screening friction to the acquirer side with country tiering. Replaced strawman baselines in the success criteria. Extended bitemporal requirements to vendor-sourced data. Added reverse-merger exclusion to label construction. Expanded therapeutic-area base-rate coverage.

---

## 0. Purpose and Non-Goals

### 0.1 Purpose

Build a reproducible, point-in-time-correct data pipeline that ranks US-listed biotech companies by their **likelihood of being acquired within the next 12–24 months**, and surfaces the reasoning behind each rank.

The core hypothesis the system operationalises: large-cap pharma acquires when it has (a) revenue about to fall off a patent cliff, (b) balance-sheet capacity to pay, and (c) a de-risked asset available in a therapeutic area it already sells into. The pipeline models all three sides and scores the intersection.

### 0.2 Non-Goals

These are explicit. Do not build them, do not design for them "just in case":

- **No order execution.** The system emits a ranked list and dossiers. It never places, sizes, or routes a trade.
- **No price prediction.** No forecasting of returns, targets, or entry levels.
- **No investment advice.** Every output artifact carries the disclaimer in §11.
- **No clinical-outcome prediction.** The system uses published base rates and trial metadata; it does not attempt to predict whether a drug will work.
- **No scraping of sources that prohibit it.** See §2.6.

### 0.3 Success Criteria

The system is working if, on a walk-forward backtest over 2016–2025:

- **Lift ≥ 3.0** at k=50 — i.e. the top-50 ranked names contain acquisition targets at ≥3× the base rate of the eligible universe.
- **Precision@25 ≥ 0.12** on a 24-month forward window.
- **Recall@200 ≥ 2.0× random** — with universe size `N`, random ranking yields `recall@k ≈ k/N`. At `N ≈ 700`, random recall@200 ≈ 0.29, so the bar is ≈ 0.57. Report the random figure alongside the achieved figure in every backtest run; never report a raw recall number without it.
- **Zero lookahead violations** on the audit test suite (§8.3).

**Baselines.** Three, all reported side by side:

| Baseline | Definition | Why |
|---|---|---|
| Random | Shuffle the universe | Floor |
| Naive-informed | Rank by `phase_max` desc, tie-broken by `cash_runway_months` desc | **The bar that matters** — this is what a junior analyst does with a Bloomberg terminal in an afternoon |
| Catalyst-proximity | Rank by inverse days-to-next-known-catalyst, restricted to Phase II+ | Second credible heuristic |

Ranking by market cap ascending is **not** an acceptable baseline. The count-weighted bulk of the universe is pre-clinical and shell-adjacent micro-caps that will never be acquired; ranking those first is close to the worst available strategy, and beating it demonstrates nothing. If the model does not beat *naive-informed* by a documented margin, it has no reason to exist.

---

## 1. System Overview

```
                    ┌──────────────────────────────────────┐
                    │            SCHEDULER                 │
                    │   (Prefect / Dagster / Airflow)      │
                    └──────────────┬───────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   ┌────▼─────┐              ┌─────▼──────┐            ┌──────▼──────┐
   │ INGEST   │              │  INGEST    │            │   INGEST    │
   │  SEC     │              │ CLINICAL   │            │   MARKET    │
   │  EDGAR   │              │ TRIALS/FDA │            │    DATA     │
   └────┬─────┘              └─────┬──────┘            └──────┬──────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   RAW ZONE (S3)     │
                        │  immutable, dated   │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   NORMALIZE / DBT   │
                        │  entity resolution  │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   FEATURE STORE     │
                        │  (bitemporal)       │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
       ┌──────▼──────┐     ┌───────▼───────┐    ┌───────▼───────┐
       │  ACQUIRER   │     │    TARGET     │    │  FEASIBILITY  │
       │  PRESSURE   │     │   QUALITY     │    │   FRICTION    │
       └──────┬──────┘     └───────┬───────┘    └───────┬───────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   FIT MATRIX +      │
                        │   COMPOSITE SCORE   │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  RANKING / DOSSIER  │
                        │  API + alerts + UI  │
                        └─────────────────────┘
```

**Cadence:** daily incremental ingest; weekly full re-score; monthly model recalibration.

---

## 2. Data Sources

### 2.0 Universe construction

**The universe is built from EDGAR filings, never from a market-data provider.** Every price/fundamentals vendor exposes a *current* roster. Building the universe from one silently drops every company that was acquired or delisted during the backtest window — which is precisely the set carrying the positive labels. The resulting backtest looks excellent and is meaningless.

EDGAR is the only source here that is inherently survivorship-free: filings are immutable and a delisted company's filing history remains permanently retrievable by CIK.

**2.0.1 Point-in-time roster**

Use the SEC **Financial Statement Data Sets** (DERA), published quarterly as ZIP archives, one record per XBRL submission in `sub.txt` with `cik`, `name`, `sic`, `adsh` (accession number), `period`, and `filed`. Landing page: `https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets` — derive archive URLs from the page rather than hardcoding, the path has changed historically. The companion **Financial Statement and Notes Data Sets** are published monthly and consolidated to quarterly after a year; use them if monthly granularity is needed.

Construction:

```python
def universe_as_of(quarter: str) -> set[int]:
    """Every biotech filer that was alive and reporting in this quarter."""
    sub = load_fsds(quarter)                      # sub.txt for e.g. '2019q3'
    BIOTECH_SIC = {
        "2833",  # Medicinal Chemicals & Botanical Products
        "2834",  # Pharmaceutical Preparations
        "2835",  # In Vitro & In Vivo Diagnostic Substances
        "2836",  # Biological Products (No Diagnostic Substances)
        "8731",  # Services — Commercial Physical & Biological Research
    }
    return {r.cik for r in sub
            if r.sic in BIOTECH_SIC
            and r.form in ("10-K", "10-Q")
            and r.countryba == "US" or r.is_us_listed}
```

Walk every quarter from 2010 to present. The union is the full historical universe; the per-quarter set is the eligible universe for that `as_of`. A CIK's disappearance from the roster is itself a signal — cross-reference against `deal` (§2.5) to classify it as acquired, delisted, or merely late-filing.

**2.0.2 Ticker and exchange resolution**

`https://www.sec.gov/files/company_tickers.json` and `company_tickers_exchange.json` map CIK → ticker → exchange, but both are **current snapshots** and will not resolve a company that no longer trades. For historical ticker resolution, parse the cover page of the company's own 10-K/10-Q (`dei:TradingSymbol`, `dei:SecurityExchangeName`), which is point-in-time by construction. Persist to `company_alias` with the filing date as `known_from`.

**2.0.3 Eligibility filters (applied per `as_of`, not once)**

| Filter | Rule | Rationale |
|---|---|---|
| Reporting status | Filed a 10-K or 10-Q in the trailing 6 months | Excludes dark and defunct registrants |
| Exchange | NYSE, NYSE American, or Nasdaq | Excludes OTC, where the phenomenon differs and data is unreliable |
| Filer type | Operating company, not a SPAC or blank-check | SPACs carry biotech SIC codes and pollute the universe |
| Size floor | Market cap > $25M | Below this, acquisitions are shell transactions (§2.5) |
| Asset floor | ≥1 program at Phase I or later in `trial` | Excludes preclinical-only and platform-services companies |

Expect roughly 600–900 companies eligible at any given `as_of`, with the full historical union substantially larger.

**2.0.4 Acquirer universe**

Separate and much smaller (~25 companies): top global pharma plus large-cap biotech with demonstrated acquisition capacity. Maintained as a **hand-curated list** in `config/acquirers.yaml`, not screened — the set is small, stable, and known. Include entry and exit dates so that a company which was itself acquired (or only reached acquirer scale mid-period) is not treated as a possible acquirer before it was one.

#### 2.0.5 Source capability matrix

None of the freely available sources covers everything. Assign roles explicitly rather than reaching for whichever is nearest:

| Need | EDGAR | FMP (free) | yfinance | IBKR |
|---|---|---|---|---|
| Point-in-time universe roster | **Yes — sole source** | No | No | No |
| Survivorship-free (delisted names) | **Yes** | Paid tiers only | No | Unreliable |
| Fundamentals: cash, burn, debt, shares | **Yes, as filed** | Basic, 250 calls/day | Patchy | No |
| Filings, 13D/13F/Form 4, 8-K text | **Yes — sole source** | 13F on paid only | No | No |
| Daily prices, current names | No | EOD only | Yes | **Yes, adjusted** |
| Historical prices, delisted names | No | Paid | **No** | Unreliable |
| Options IV, short interest / borrow | No | No | Limited | **Yes — sole source** |
| Redistribution rights | Public domain | Requires agreement | No ToS grant | Account-only |

**Practical assignment:**

- **EDGAR** — universe, all fundamentals, all filing-derived features (Blocks A, B, C, E, G). This is the backbone and it is free. Budget most of the ingest effort here.
- **IBKR** — daily adjusted prices and corporate actions for currently listed names, plus options IV and borrow data that nothing else in this set provides. You are already paying for it. Respect the pacing limits (historical-data requests are throttled aggressively; a naive loop over 700 tickers will trigger pacing violations and silent truncation).
- **yfinance** — prototyping only. It is an unofficial scraper of undocumented endpoints with no stability guarantee, no delisted coverage, and no redistribution grant. Acceptable for a scratch notebook, unacceptable in the pipeline.
- **FMP free tier** — **cannot serve this system.** At 250 calls/day with a 500MB trailing-30-day bandwidth cap and end-of-day data only, a single pass over a 700-name universe takes three days. 13F, intraday, and bulk endpoints are paid-only, and display or redistribution requires a separate licensing agreement.

**2.0.6 The one gap worth paying for**

Historical prices for **delisted** tickers is the single requirement none of the four satisfies, and it is not optional: `E[return | deal]` (§5.3) needs the price at `as_of` for companies that no longer exist, and those are exactly the positive labels.

Options, cheapest first:

1. **FMP Starter (~$15/mo)** — 30+ years of US history on paid tiers. The cheapest path to a usable backtest; validate delisted-ticker coverage against a sample of 20 known acquisitions *before* committing.
2. Any vendor offering an explicitly survivorship-bias-free US equity file.
3. **Reduced-scope fallback** — reconstruct only the prices actually needed: monthly closes for label companies between `as_of` and announcement. This is a few thousand data points, not a full history, and can be assembled semi-manually if no budget exists. Document the reduced coverage in the backtest report; do not silently narrow the universe to names with available prices, which reintroduces the survivorship bias by the back door.

Do not attempt option 4, scraping a price site. It violates ToS, breaks without warning, and puts an unverifiable input under a hard gate.

#### 2.0.7 Price archive and corporate actions

The intended pattern — buy one month of vendor access, bulk-download history, then append daily from a free source — is workable, but **only if the archive stores raw prices**. Storing adjusted prices makes every split a retroactive rewrite of the entire history for that ticker.

**Rule: store unadjusted OHLCV plus a separate corporate-actions table. Adjust at read time, never at write time.**

```sql
CREATE TABLE price_daily (          -- RAW. As traded. Never rewritten.
  company_id  BIGINT REFERENCES company,
  trade_date  DATE NOT NULL,
  open_raw    NUMERIC, high_raw NUMERIC, low_raw NUMERIC, close_raw NUMERIC,
  volume_raw  BIGINT,
  vendor      TEXT NOT NULL,        -- 'fmp'|'ibkr'|'yfinance'
  known_from  TIMESTAMPTZ,
  PRIMARY KEY (company_id, trade_date, vendor)
);

CREATE TABLE corporate_action (
  company_id  BIGINT REFERENCES company,
  ex_date     DATE NOT NULL,
  action_type TEXT CHECK (action_type IN ('split','reverse_split','dividend',
                                          'spinoff','ticker_change')),
  ratio       NUMERIC,              -- 4.0 for 4:1 fwd, 0.05 for 1:20 reverse
  cash_amount NUMERIC,
  new_ticker  TEXT,
  source      TEXT NOT NULL,        -- 'sec_8k'|'fmp'|'yfinance'|'ibkr'|'manual'
  is_verified BOOLEAN DEFAULT FALSE,
  known_from  TIMESTAMPTZ, source_url TEXT,
  PRIMARY KEY (company_id, ex_date, action_type)
);
```

```python
def adjusted_close(company_id, trade_date, as_of):
    """Split-adjusted to `as_of`. Only actions KNOWN by as_of are applied."""
    raw = get_raw_close(company_id, trade_date)
    factor = prod(a.ratio for a in corporate_actions(company_id)
                  if trade_date < a.ex_date <= as_of
                  and a.known_from <= as_of
                  and a.action_type in ("split", "reverse_split"))
    return raw / factor
```

**Why raw storage is mandatory here, not merely convenient:**

1. **Market cap breaks silently otherwise.** `market_cap = price × shares_outstanding`, where shares outstanding comes from the as-filed `dei:EntityCommonStockSharesOutstanding` on an EDGAR cover page — an unadjusted, point-in-time figure. Multiplying it by a *retro-adjusted* price yields a market cap wrong by exactly the split factor. After a 1-for-20 reverse split, every historical market cap for that company is overstated 20×. This propagates into `enterprise_value`, `ev_to_risk_adjusted_npv`, and the $25M size floor (§2.0.3), and nothing in the output looks wrong.

2. **Retro-adjusted price levels are a lookahead leak.** Returns are invariant to adjustment; levels are not. A 2019 price of $0.80 becomes $16.00 in an archive adjusted for a 2023 reverse split. Any filter keying on level — the size floor, penny-stock exclusion, `ev_to_cash` — would then be using 2023 information to decide 2019 eligibility. The `as_of` guard in `adjusted_close()` above is what §8.3 must test.

3. **Vendor adjustment methodologies differ.** Raw prints agree across vendors because they are the actual traded price; adjusted series do not, because dividend-reinvestment conventions vary. Splicing a vendor's adjusted history onto another vendor's adjusted daily feed creates a discontinuity at the seam that is invisible in the data and shows up as a phantom return.

**Sourcing corporate actions.** Do not rely on a price vendor alone for this universe. Precedence:

1. **SEC filings** (8-K Item 5.03, 8-A, S-1/A) — authoritative and survivorship-free. A discontinuity in `dei:EntityCommonStockSharesOutstanding` between consecutive filings that is not explained by an equity raise is a strong detector for an unrecorded split; run it as a reconciliation job.
2. Vendor split endpoints — convenient, and correct for large caps.
3. yfinance `actions` — usable, but treat as unverified until reconciled against (1).

**yfinance-specific hazards in this universe** (all more frequent for microcap biotech than for large caps):

- `auto_adjust` defaults to `True` in current versions. Pass `auto_adjust=False, actions=True` explicitly, or you will silently archive adjusted prices and defeat the whole design.
- Reverse splits are sometimes reported late, with the wrong ratio, or not at all for small tickers.
- Reverse splits in this universe frequently coincide with a **ticker change**. Yahoo may rebind the old symbol to an unrelated company or drop the history. Key the archive on `company_id` (CIK-derived), never on ticker string.
- When a company is delisted, the symbol may vanish and the daily append **stops silently**. Alert on any universe member with no price row for 5 consecutive trading days.

**Licensing.** Verify before relying on the plan: bulk historical data downloaded during a paid month may be subject to retention and redistribution restrictions after the subscription lapses. Vendor terms in this segment commonly distinguish personal from commercial use and require a separate agreement for display or redistribution. Record the answer in `config/vendor_terms.md`.

### 2.1 SEC EDGAR (primary, free, authoritative)



| What | Endpoint / Form | Use |
|---|---|---|
| Company index | `https://www.sec.gov/files/company_tickers.json` | CIK ↔ ticker map |
| Filing history | `https://data.sec.gov/submissions/CIK{cik:010d}.json` | Detect new 8-K, 10-Q, DEFM14A |
| XBRL facts | `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json` | Cash, burn, debt, shares |
| Single concept | `https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json` | Targeted pulls |
| Full-text search | `https://efts.sec.gov/LATEST/search-index?q=...&forms=8-K` | Event detection |
| Institutional holdings | Form 13F-HR | Specialist-fund ownership |
| Insider transactions | Form 4 | Open-market buys/sells |
| Deal labels | 8-K Item 1.01, SC 14D9, DEFM14A, S-4 | Backtest ground truth |

**Hard requirements:**
- `User-Agent` header must be set to `"{AppName} {contact-email}"`. Requests without it are blocked.
- Rate limit: **10 requests/second maximum**, enforced client-side with a token bucket. Back off on HTTP 403.
- All responses cached to the raw zone with retrieval timestamp. Never re-fetch what you already have unless the filing index shows a new accession number.

**Key XBRL tags** (fall back through the list; small biotechs are inconsistent):
```
cash:        CashAndCashEquivalentsAtCarryingValue
             CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents
investments: ShortTermInvestments, MarketableSecuritiesCurrent
burn:        NetCashProvidedByUsedInOperatingActivities
debt:        LongTermDebtNoncurrent, ConvertibleNotesPayable
shares:      dei:EntityCommonStockSharesOutstanding
R&D:         ResearchAndDevelopmentExpense
```

### 2.2 ClinicalTrials.gov API v2

Base: `https://clinicaltrials.gov/api/v2/studies`

Pull for every target-universe company (match on `sponsor.leadSponsor.name`, then hand-verify aliases into an override table):

```
fields = NCTId, briefTitle, overallStatus, phase, studyType,
         conditions, interventions, leadSponsor, collaborators,
         primaryOutcomes, enrollmentInfo, startDateStruct,
         primaryCompletionDateStruct, completionDateStruct,
         locationCountries, designInfo, hasResults, lastUpdatePostDate
```

**Critical:** also pull the **version history** for each NCT ID. Primary-endpoint changes, enrollment changes, and completion-date slips mid-trial are high-signal features and are only visible in the diff between versions.

### 2.3 FDA

| Source | Access | Use |
|---|---|---|
| openFDA Drugs@FDA | `https://api.fda.gov/drug/drugsfda.json` | Approvals, application numbers, sponsor |
| Orange Book | Quarterly ZIP download (`products.txt`, `patent.txt`, `exclusivity.txt`) | **Small-molecule patent expiry — the core of the acquirer-pressure model** |
| Purple Book | Downloadable CSV | Biologic exclusivity dates |
| AdCom calendar | FDA website, scheduled scrape or manual entry | Catalyst dates |

Orange Book `patent.txt` gives `Patent_Expire_Date_Text` per application/product. Join to product revenue (§2.4) to compute revenue-at-risk by year. This join is the highest-value and highest-effort part of the build; budget for it accordingly.

### 2.4 Commercial market and fundamentals data

Required, requires a paid vendor. Candidates: Financial Modeling Prep, Polygon.io, Tiingo, Nasdaq Data Link, Refinitiv.

Needed fields:
- Daily OHLCV, shares outstanding, market cap (point-in-time, **survivorship-bias-free** — delisted tickers must remain queryable)
- Segment/product-level revenue for large-cap pharma (may require manual extraction from 10-K exhibits if vendor lacks product granularity)
- Short interest (bi-monthly, FINRA)
- Options implied volatility, if available — optional, used only for context in dossiers

**Vendor data is subject to the same bitemporal discipline as SEC data (§3.1).** This is not implied — state it in the contract and enforce it in code:

- Every vendor-sourced fact lands in `financial_fact` with an explicit `known_from`, set to the date the vendor first published that value, **not** the period it describes and **not** the date we happened to fetch it.
- **Restatements must be stored as new rows**, closing the prior row's `valid_to`. Never update in place. Vendors silently revise shares-outstanding and market-cap history; an in-place update destroys the point-in-time record and the corruption is undetectable after the fact.
- If a vendor cannot supply first-publication timestamps, treat their data as **known at period_end + a conservative lag** (45 days for fundamentals, 0 for prices) and document the assumption in `config/vendor_lag.yaml`.
- `enterprise_value` and `cash_runway_months` feed a hard gate (`dilution_gate`, §5.1). A silent lookahead leak here propagates directly into the score, so the §8.3 audit **must** sample vendor-sourced facts specifically, not only SEC-sourced ones.

### 2.5 Deal label dataset (for backtest)

Ground truth of completed and announced acquisitions of US-listed biotechs, 2010–present. Build by:
1. EDGAR full-text search for `SC 14D9`, `DEFM14A`, `S-4` filings by SIC codes 2836, 8731, 2834.
2. Parse announcement date, acquirer, consideration per share, CVR presence, premium to prior close.
3. Hand-verify. Expect 400–700 events. **This dataset must be manually reviewed; automated extraction alone will be too noisy for labels.**

Store `announcement_date` separately from `completion_date`. Labels key on announcement.

**Mandatory exclusions — screen for these explicitly during hand-verification.** EDGAR full-text search over SIC 2836/8731/2834 will return a substantial number of transactions that are structurally *not* the phenomenon being modelled. Each label row carries `deal_type`, and only `strategic_acquisition` is a positive label:

| `deal_type` | Description | Label |
|---|---|---|
| `strategic_acquisition` | Operating pharma/biotech buys the target for its assets | **Positive** |
| `reverse_merger` | Failed clinical-stage company merges with a private company, which inherits the listing | **Excluded** |
| `shell_transaction` | Cash-shell acquisition, no asset rationale | **Excluded** |
| `liquidation` | Dissolution, cash distribution to shareholders | **Excluded** |
| `asset_sale` | Assets sold, entity survives | **Excluded** |
| `pe_take_private` | Financial sponsor, no strategic acquirer | Separate label; do not pool |

Reverse mergers are the dangerous case because they *look* like acquisitions in the filings. They are the exact inverse of the modelled phenomenon — a company gets "acquired" **because** its science failed and its listing is worth more than its pipeline. Left un-flagged in the training set, they teach Phase 2 that low `target_asset_quality` predicts acquisition, directly inverting the signal Block B exists to produce.

**Detection heuristics for the review queue** (flag, don't auto-exclude): acquirer is privately held and has no prior approved product; target market cap < $50M at announcement; consideration is majority stock with target shareholders retaining <20%; a reverse stock split filed within 90 days of announcement; target announced a pipeline discontinuation within 12 months prior.

### 2.6 Corporate process signals and agreement structures

These sources carry the highest-precision signals in the domain and were absent from v0.1. Blocks A–E infer *latent* acquisition attractiveness from fundamentals. This section captures *revealed* process — cases where a sale is already underway or contractually pre-arranged. Conditional probabilities here are an order of magnitude above anything the fundamental model produces, and the scoring layer must treat them accordingly (§5.2).

**2.6.1 Strategic-alternatives disclosures**

A public "exploring strategic alternatives" announcement is the single strongest observable signal that a company may be sold. It is disclosed via **8-K Item 7.01 (Reg FD) or Item 8.01 (Other Events)** plus the press-release exhibit (EX-99.1) — **not** Item 1.01, which v0.1 was alone in watching and which only fires when the deal is already signed. By that point the trade is over.

Detection: EDGAR full-text search over 8-K bodies and EX-99 exhibits for a maintained phrase list:

```yaml
strategic_process_phrases:
  strong:                    # near-explicit sale process
    - "exploring strategic alternatives"
    - "review of strategic alternatives"
    - "engaged {ADVISOR} as financial advisor"
    - "formed a strategic committee"
    - "evaluating a potential sale of the company"
  moderate:                  # ambiguous — may be financing, not sale
    - "strategic review"
    - "exploring options to maximize shareholder value"
  negative:                  # process concluded without a deal
    - "concluded its review of strategic alternatives"
    - "determined to continue as a standalone company"
```

Phrase matching is a **candidate generator, not a classifier**. Every hit goes to the review queue for confirmation of (a) that a process is genuinely open, (b) whether it covers the whole company or only an asset. Asset-level reviews are a materially weaker signal and must be tagged separately. False positives here are expensive: the feature carries a large weight, so a mislabelled financing announcement will badly distort the rank.

State transitions to track per company: `none → rumored → disclosed_open → concluded_deal | concluded_no_deal`. `concluded_no_deal` populates the existing `recent_failed_process` penalty in §4.5, which is the correct opposite pole of this feature.

**2.6.2 Activist and strategic stakes — Schedule 13D**

v0.1 ingested 13F only. 13F is a quarterly passive-holdings snapshot, 45 days stale, and near-worthless as a takeover precursor. **Schedule 13D** is the relevant filing: >5% beneficial ownership with intent to influence control, due within 5 business days, with Item 4 stating purpose in the filer's own words.

Ingest:
- **SC 13D** and every **SC 13D/A** amendment — amendments often escalate ("intends to engage with the board regarding strategic alternatives") and the escalation trajectory is itself the signal.
- **SC 13G** for completeness, but weight it near zero — it is the passive-intent form.
- Parse Item 4 (Purpose of Transaction) text; classify intent into `passive | engagement | board_seats | sale_demand` via the review queue.
- Maintain `config/activist_filers.yaml` — a list of CIKs for known activist funds. A 13D from a healthcare-specialist crossover fund at IPO is not the same event as a 13D from an activist with a campaign history, and the model must distinguish them.

Also ingest **13D filings by strategic (corporate) filers**. A large pharma taking a >5% stake in a small biotech is a distinct and strong pattern — a toehold that frequently precedes a full bid.

**2.6.3 Incumbent partners and option-to-acquire structures**

A large share of biotech acquisitions are not competitive auctions. They are an existing partner exercising rights it already holds. Where such rights exist, `P(acquired by that specific partner)` is far higher than any market-based fit score in §4.4 can express, and the identity of the acquirer is close to predetermined.

Structures, in descending order of signal strength:

| Structure | Where disclosed | Strength |
|---|---|---|
| Explicit option to acquire the company at a milestone | 8-K Item 1.01 + EX-10 collaboration agreement; described in 10-K Business section | **Deterministic-adjacent** |
| Right of first negotiation / first refusal on a change of control | EX-10 exhibit, often redacted | Very high |
| Equity stake held by the partner plus co-commercialisation rights | 10-K, 13D/13G, collaboration agreement | High |
| Ordinary licensing deal with milestone/royalty terms only | 8-K, 10-K | Moderate |

**Scoping caveat, and it matters for effort estimation:** outright options to acquire an entire company are most common in *private* build-to-buy structures with venture backers, and are comparatively rare in the public universe this system screens. The public-market analogue that will actually populate the table is the third and fourth rows — an incumbent partner with an equity stake and commercial rights. Build for that case first; treat the explicit option as a rarer, higher-weight subtype rather than the design centre.

**Extraction approach.** These terms live in EX-10 exhibits, which are long, inconsistently structured, and frequently redacted. Do **not** attempt full automated contract parsing in v1. Instead:
1. Detect the existence of a collaboration/licence agreement (8-K Item 1.01 with an EX-10 exhibit, counterparty resolvable to a company in the acquirer universe).
2. Extract counterparty, date, and — where present in the unredacted portions — the presence of change-of-control language, via keyword search over the exhibit text.
3. Route to the manual-entry queue (§3.4) for a human to record structure type and terms.

Expect this to be the most labour-intensive source in the system. Scope it to the top 200 companies by composite score from the Block A–E model, not the whole universe — this makes it a second-pass enrichment on an already-ranked shortlist rather than a full-universe ingest.

### 2.7 Licensing and compliance

- SEC, ClinicalTrials.gov, and openFDA are public-domain / open-access. Respect rate limits.
- Commercial vendor data is licensed — **redistribution in exported dossiers may be prohibited**. Check the contract before any output leaves the system. Design the export layer so vendor-derived fields can be masked by a config flag.
- No scraping of any site whose `robots.txt` or ToS forbids it. If a needed source forbids automation, it goes in the manual-entry queue (§3.4), not a scraper.

---

## 3. Data Model

Postgres (or DuckDB for a single-user build). All fact tables are **bitemporal**.

### 3.1 Bitemporality — non-negotiable

Every fact row carries:

```sql
valid_from      DATE NOT NULL,   -- when the fact became true in the world
valid_to        DATE,            -- NULL = still true
known_from      TIMESTAMPTZ NOT NULL,  -- when WE learned it
source_id       TEXT NOT NULL,
source_url      TEXT
```

All feature computation queries must filter `known_from <= as_of_date`. A financial fact from a 10-Q filed on 2023-05-08 covering the quarter ended 2023-03-31 has `valid_from = 2023-03-31` and `known_from = 2023-05-08`. Backtests as of 2023-04-15 must not see it. **This is the single most important correctness property in the system.**

### 3.2 Core tables

```sql
CREATE TABLE company (
  company_id      BIGSERIAL PRIMARY KEY,
  cik             TEXT UNIQUE,
  name            TEXT NOT NULL,
  ticker          TEXT,
  exchange        TEXT,
  sic_code        TEXT,
  is_active       BOOLEAN,
  delisted_date   DATE,
  role            TEXT CHECK (role IN ('target','acquirer','both'))
);

CREATE TABLE company_alias (
  company_id  BIGINT REFERENCES company,
  alias       TEXT NOT NULL,       -- CT.gov sponsor strings, FDA applicant names
  source      TEXT NOT NULL,
  is_verified BOOLEAN DEFAULT FALSE
);

CREATE TABLE financial_fact (
  company_id  BIGINT REFERENCES company,
  metric      TEXT NOT NULL,       -- 'cash','quarterly_opex_burn','debt','shares_out'
  value       NUMERIC,
  unit        TEXT DEFAULT 'USD',
  period_end  DATE,
  valid_from  DATE, valid_to DATE,
  known_from  TIMESTAMPTZ, source_id TEXT, source_url TEXT
);

CREATE TABLE asset (                -- a drug program
  asset_id        BIGSERIAL PRIMARY KEY,
  company_id      BIGINT REFERENCES company,
  name            TEXT,
  modality        TEXT,             -- small_molecule|mab|adc|cell|gene|rna|peptide|radioligand
  target_protein  TEXT,
  therapeutic_area TEXT NOT NULL,   -- controlled vocab, see §3.5
  indication      TEXT,
  is_lead         BOOLEAN
);

CREATE TABLE trial (
  nct_id                  TEXT PRIMARY KEY,
  asset_id                BIGINT REFERENCES asset,
  phase                   TEXT,
  status                  TEXT,
  enrollment              INT,
  primary_completion_date DATE,
  uses_biomarker_selection BOOLEAN,
  is_randomized           BOOLEAN,
  has_active_comparator   BOOLEAN,
  primary_endpoint_text   TEXT,
  endpoint_changed_midtrial BOOLEAN,
  countries               TEXT[],
  known_from              TIMESTAMPTZ
);

CREATE TABLE patent_expiry (        -- acquirer side
  acquirer_id       BIGINT REFERENCES company,
  product_name      TEXT,
  application_no    TEXT,
  therapeutic_area  TEXT,
  loe_date          DATE NOT NULL,
  ttm_revenue_usd   NUMERIC,
  exclusivity_type  TEXT,           -- 'patent'|'orphan'|'ped'|'bla_12yr'
  source            TEXT            -- 'orange_book'|'purple_book'|'manual'
);

CREATE TABLE deal (                 -- labels
  deal_id           BIGSERIAL PRIMARY KEY,
  target_id         BIGINT REFERENCES company,
  acquirer_id       BIGINT REFERENCES company,
  announcement_date DATE NOT NULL,
  completion_date   DATE,
  upfront_per_share NUMERIC,
  has_cvr           BOOLEAN,
  cvr_max_per_share NUMERIC,
  premium_1d        NUMERIC,
  premium_30d       NUMERIC,
  status            TEXT             -- announced|completed|terminated
);

CREATE TABLE corporate_process_event (   -- §2.6.1
  event_id      BIGSERIAL PRIMARY KEY,
  company_id    BIGINT REFERENCES company,
  event_date    DATE NOT NULL,
  state         TEXT CHECK (state IN
                  ('rumored','disclosed_open','concluded_deal','concluded_no_deal')),
  scope         TEXT CHECK (scope IN ('whole_company','asset_only','unclear')),
  strength      TEXT CHECK (strength IN ('strong','moderate')),
  advisor_name  TEXT,
  accession_no  TEXT,
  matched_phrase TEXT,
  is_verified   BOOLEAN DEFAULT FALSE,   -- review queue gate
  known_from    TIMESTAMPTZ, source_url TEXT
);

CREATE TABLE activist_position (        -- §2.6.2
  position_id     BIGSERIAL PRIMARY KEY,
  company_id      BIGINT REFERENCES company,
  filer_cik       TEXT NOT NULL,
  filer_name      TEXT,
  filer_type      TEXT CHECK (filer_type IN
                    ('activist','crossover_fund','strategic_corporate','other')),
  form_type       TEXT CHECK (form_type IN ('SC 13D','SC 13D/A','SC 13G','SC 13G/A')),
  pct_of_class    NUMERIC,
  stated_intent   TEXT CHECK (stated_intent IN
                    ('passive','engagement','board_seats','sale_demand')),
  amendment_seq   INT,                   -- escalation trajectory
  filed_date      DATE NOT NULL,
  known_from      TIMESTAMPTZ, source_url TEXT
);

CREATE TABLE partnership_structure (    -- §2.6.3
  structure_id      BIGSERIAL PRIMARY KEY,
  company_id        BIGINT REFERENCES company,   -- the potential target
  partner_id        BIGINT REFERENCES company,   -- the potential acquirer
  asset_id          BIGINT REFERENCES asset,
  structure_type    TEXT CHECK (structure_type IN
                      ('acquisition_option','rofn_rofr','equity_plus_commercial',
                       'license_only')),
  partner_equity_pct NUMERIC,
  agreement_date    DATE,
  option_trigger    TEXT,                -- e.g. 'phase_2_topline'
  is_redacted       BOOLEAN,             -- terms not fully disclosed
  entry_method      TEXT CHECK (entry_method IN ('manual','keyword_detected')),
  is_verified       BOOLEAN DEFAULT FALSE,
  known_from        TIMESTAMPTZ, source_url TEXT
);

CREATE TABLE score_run (
  run_id       BIGSERIAL PRIMARY KEY,
  as_of_date   DATE NOT NULL,
  model_version TEXT NOT NULL,
  config_hash  TEXT NOT NULL,
  created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE score (
  run_id                 BIGINT REFERENCES score_run,
  company_id             BIGINT REFERENCES company,

  -- Two scores, two ranks. Never a bare `rank` column: see §5.4.
  composite              NUMERIC,   -- tiered fundamental score (§5.1–5.2)
  rank_by_composite      INT,       -- ordering under `composite` DESC
  expected_value         NUMERIC,   -- P(deal)-weighted EV (§5.3)
  rank_by_expected_value INT,       -- ordering under `expected_value` DESC — DEFAULT VIEW
  p_deal_24m             NUMERIC,
  expected_return_if_deal NUMERIC,

  tier                   SMALLINT CHECK (tier BETWEEN 0 AND 3),
  tier_reason            TEXT,
  subscores              JSONB,   -- {"acquirer_pressure": 0.71, "target_quality": 0.58, ...}
  contributions          JSONB,   -- per-feature attribution for the dossier
  PRIMARY KEY (run_id, company_id)
);
```

### 3.3 Entity resolution

The hardest data problem in this build. ClinicalTrials.gov sponsor strings, FDA applicant names, and SEC registrant names for the same company routinely differ.

Approach:
1. Deterministic match on exact normalised name (lowercase, strip `inc|corp|ltd|plc|holdings|therapeutics|pharmaceuticals|biosciences`, collapse whitespace).
2. Fuzzy match (token-set ratio ≥ 88) into a **review queue**, never auto-accepted.
3. Human confirms; result written to `company_alias` with `is_verified = TRUE`.
4. Unresolved sponsor strings are logged and reported weekly. Do not silently drop them.

Budget real time for this. Expect ~15% of CT.gov sponsors to need manual mapping in the first pass.

### 3.4 Manual entry and review queue

Several sources cannot be fully automated: entity-resolution candidates (§3.3), Block G process events and 13D intent classification (§2.6), partnership structures buried in redacted EX-10 exhibits (§2.6.3), deal-type classification (§2.5), and any source whose terms forbid automated access (§2.7). These are not edge cases — they are load-bearing inputs, several of which carry the heaviest weights in the model.

```sql
CREATE TABLE review_item (
  item_id      BIGSERIAL PRIMARY KEY,
  item_type    TEXT NOT NULL,     -- 'entity_match'|'process_event'|'activist_intent'
                                  -- |'partnership_structure'|'deal_type'
  payload      JSONB NOT NULL,    -- candidate record awaiting confirmation
  evidence_url TEXT,
  priority     INT DEFAULT 0,     -- process events outrank entity matches
  status       TEXT DEFAULT 'pending'
                 CHECK (status IN ('pending','confirmed','rejected','needs_info')),
  reviewed_by  TEXT,
  reviewed_at  TIMESTAMPTZ,
  note         TEXT
);
```

Requirements:
- A minimal review UI (or, acceptably for v1, a CLI plus a spreadsheet export/import round-trip). Do not defer this — an unreviewable queue means Block G never ships.
- **Confirmation writes back with `known_from` set to the underlying filing date, not the review date.** Reviewing a filing in September does not mean the market learned of it in September. Getting this backwards silently destroys the backtest.
- Queue depth and median age by `item_type` are reported in every run. A process-event backlog older than one business day is an operational failure, not a nuisance — the score is stale in exactly the place it matters most.

### 3.5 Controlled vocabularies

Maintain as versioned YAML in the repo, not in the database:
- `therapeutic_area.yaml` — ~20 values (oncology_solid, oncology_heme, immunology, neurology, cardiometabolic, rare_metabolic, infectious_disease, ophthalmology, ...)
- `modality.yaml`
- `base_rates.yaml` — see §4.2

Changes to these files bump `config_hash` and require a re-score.

---

## 4. Feature Engineering

Features are computed **as of a date**, from the bitemporal store. Every feature function has the signature:

```python
def feature(company_id: int, as_of: date, ctx: FeatureContext) -> float | None
```

Returning `None` is meaningful and must propagate as missing, never as zero.

### 4.1 Block A — Acquirer Pressure Index (computed per large-cap acquirer)

For each of ~25 acquirers (top global pharma + large-cap biotech), compute annually and forward 5 years:

| Feature | Definition |
|---|---|
| `revenue_at_risk_3y` | Σ TTM revenue of products with `loe_date` within 36 months of `as_of`, ÷ total TTM revenue |
| `revenue_at_risk_5y` | Same, 60-month window |
| `cash_capacity` | (cash + marketable securities) + (target_leverage_ratio × EBITDA − existing_net_debt), floored at 0 |
| `equity_capacity` | `market_cap × max_dilution_tolerance × currency_quality` — see below |
| `dry_powder` | `cash_capacity + equity_capacity` |
| `deal_cadence_3y` | Count of completed acquisitions > $500M in trailing 36 months |
| `stock_deal_propensity` | Share of the acquirer's trailing-10y deals using ≥30% stock consideration |
| `pipeline_gap_by_ta` | Per TA: revenue_at_risk in that TA − (count of own Phase III assets in that TA × assumed_peak_sales) |

`pipeline_gap_by_ta` is the key output. It answers: *which therapeutic areas does this acquirer need to buy into?*

**Equity currency.** v0.1 modelled cash and debt only, which understates capacity for acquirers with a strong stock and misranks feasibility on large targets. Stock-and-cash and majority-stock consideration are common enough in this universe to matter materially.

```
currency_quality   = percentile_rank(acquirer.fwd_pe vs peer group)
                     × stability_factor(trailing 12m realised vol of the stock)
max_dilution_tolerance = 0.15    # config; ~15% share issuance before boards balk
equity_capacity    = market_cap × max_dilution_tolerance × currency_quality
```

`currency_quality` encodes the standard logic that management issues stock when it believes the stock is expensive and pays cash when it believes it is cheap. An acquirer trading at a depressed multiple has poor currency regardless of market cap, and `equity_capacity` should collapse accordingly. Weight `equity_capacity` by `stock_deal_propensity` when computing `size_feasibility` in §4.4 — some acquirers are structurally cash-only by policy, and their history reveals it.

### 4.2 Block B — Target Asset Quality

Prior probability of the lead asset reaching approval, from published base rates. Store these in `base_rates.yaml` with citation and date; do not hardcode.

Anchor values (BIO / Informa Biomedtracker / Amplion, ~7,455 programs, 9,985 phase transitions):

```yaml
loa_from_phase_1_overall: 0.096
phase_2_success: 0.31            # lowest of any transition
phase_3_to_filing: 0.58          # ~40%+ of Phase III never file
filing_to_approval: 0.86
first_cycle_approval: 0.61
by_therapeutic_area:                # ALL areas must be populated — see note below
  oncology_solid:      0.05         # lowest LOA from Phase I
  oncology_heme:       null         # distinct from solid; populate from source
  hematology_nonmalig: 0.26         # highest LOA of any area
  infectious_disease:  null
  neurology:           null
  psychiatry:          null
  cardiovascular:      null
  metabolic:           null
  autoimmune:          null
  respiratory:         null
  ophthalmology:       null
  # ... remaining areas from therapeutic_area.yaml
modifiers:
  biomarker_selection: 2.0       # multiplicative, capped; oncology evidence ~5x HR
  prior_crl: -0.28               # additive to PoA, from FDA historical review data
pdufa_priors:
  nda_standard: 0.85
  nda_priority: 0.90
  orphan: 0.93
```

**Completeness requirement.** Shipping with 2 of ~20 therapeutic areas calibrated and the rest silently falling back to a blended overall rate systematically mis-scores 90% of the universe. The BIO/Informa study reports 14 major disease areas — populate every one from the source, and where the study's taxonomy does not map cleanly onto `therapeutic_area.yaml`, record the mapping decision in the YAML comments rather than defaulting.

Areas requiring particular care because their profiles diverge widely from the blend: **gene and cell therapy** (small trials, accelerated pathways, manufacturing-driven failures rather than efficacy failures), **rare/orphan disease** (see the phase-conditional note below), and **neurology and psychiatry** (high Phase II attrition, subjective endpoints, large placebo response). Where a modality-specific rate is better supported than the area rate, prefer the modality rate and record which was used in `contributions` so the dossier can show it.

A feature that falls back to the blended rate must emit a `base_rate_fallback` flag, and the run report must list every company scored on a fallback. If that list is long, the YAML is incomplete, not the universe unusual.

**Implementation note on orphan status:** do not naively treat orphan designation as positive. Published data shows orphan programs have *higher* Phase I POS (0.759 vs 0.664) but *lower* Phase II (0.488 vs 0.583) and Phase III (0.467 vs 0.590), with a lower cumulative POS overall. Encode orphan as a phase-conditional modifier, not a flat bonus. Any implementation that adds a constant for "orphan = true" is wrong and must fail review.

**PoA is an input to valuation, not a monotone proxy for acquisition probability.** v0.1 fed `lead_asset_poa` linearly into `target_asset_quality`, which is a modelling error rather than a gap. Higher PoA does not imply a higher chance of a bid; it implies a higher *price*. Two separate mechanisms are being conflated:

- Acquirers frequently buy **before** a de-risking readout, precisely to capture risk-adjusted NPV rather than pay for certainty. These deals carry the largest premiums to the pre-announcement price.
- Acquirers also frequently buy **immediately after** positive Phase III or on the approach to PDUFA. Empirically this is a very common trigger, so the claim that late-stage assets are unlikely to be acquired is not supported — what changes is not `P(deal)` but the return to an investor who bought beforehand, because the re-rating has already happened in the open market.

The correct encoding is therefore **stage-conditional, and split across two targets** (see §5.3). Do not fold raw PoA into the composite as a linear term.

```python
# WRONG — v0.1
target_asset_quality += w * lead_asset_poa

# RIGHT — stage-conditional, PoA routed to the return model
deal_probability_features = [phase_max, catalyst_window, partner_present, ...]
expected_return_features  = [lead_asset_poa, ev_to_risk_adjusted_npv, ...]
```

Derived features:

| Feature | Definition |
|---|---|
| `lead_asset_poa` | Base rate × modifiers, from phase + TA + biomarker + CRL history. **Feeds the return model (§5.3), not the deal-probability composite** |
| `catalyst_window` | Days to the next value-inflecting readout, bucketed. Deal likelihood is elevated in a window ahead of a major catalyst and again shortly after a positive one; encode as categorical buckets (`>730`, `365–730`, `180–365`, `60–180`, `<60`, `post_positive_0–180`) and let the model learn the shape rather than imposing an inverted-U by hand |
| `ev_to_risk_adjusted_npv` | EV ÷ (PoA × PV(peak sales × multiple)). **The actual cheapness measure** — combines PoA with price instead of using PoA alone. Low values are the interesting ones |
| `phase_max` | Furthest phase reached by any asset |
| `has_positive_ph3` | Boolean: completed Phase III with `hasResults` and no subsequent discontinuation 8-K |
| `pdufa_pending` | Boolean + days to date |
| `endpoint_stability` | 1.0 if no mid-trial primary-endpoint change; penalised otherwise |
| `trial_design_quality` | Composite: randomised + active comparator + biomarker-selected + US-heavy enrollment |
| `asset_count_ph2plus` | Portfolio breadth — single-asset companies are riskier but also cheaper targets |

### 4.3 Block C — Financial Screen

| Feature | Definition |
|---|---|
| `enterprise_value` | market_cap − (cash + short-term investments) + total_debt |
| `cash_runway_months` | (cash + ST investments) ÷ trailing-4Q average quarterly operating burn |
| `ev_to_cash` | EV ÷ cash — flags negative-EV situations |
| `dilution_risk` | Boolean: runway < 12 months **and** a catalyst date inside the runway window |
| `atm_capacity_pct` | Remaining ATM shelf ÷ market cap, parsed from 424B5 / 10-Q |
| `size_band` | Bucket EV into <500M, 500M–2B, 2B–5B, 5B–15B, >15B |

`dilution_risk = TRUE` should sharply penalise the composite. Being right on the asset and wrong on the financing is the dominant failure mode this screen exists to avoid.

### 4.4 Block D — Strategic Fit

Fit is a **pairwise** score between each acquirer and each target. It has two structurally different kinds of term, and they must not be mixed into one weighted sum:

- **Preference terms** — how much this acquirer *wants* this target. Additive, weighted, tradeable against each other.
- **Feasibility gates** — whether this acquirer *can* buy this target at all. Multiplicative, and capable of driving the pair to zero regardless of how attractive it is.

```python
def fit(acquirer, target) -> float:
    preference = (
          w1 * ta_overlap(acquirer.pipeline_gap_by_ta, target.lead_asset.therapeutic_area)
        + w2 * modality_capability(acquirer, target.modality)
        + w3 * geographic_fit(acquirer.commercial_footprint, target.trial_countries)
    )
    gates = (
          size_feasibility(acquirer.dry_powder, target.ev, acquirer.stock_deal_propensity)
        * (1 - fis_penalty(acquirer.bloc, target))          # §4.4.1
        * (1 - antitrust_penalty(acquirer, target))         # §4.4.2
    )
    return preference * gates
```

Making these gates rather than weighted terms is deliberate. An additive `w5 * fis_risk` term lets a large therapeutic-area overlap out-vote a regulatory obstacle that could block the transaction outright — the model would rank a deal highly precisely because the acquirer badly needs the asset, which is exactly backwards. A blocked deal is not a slightly worse deal.

`size_feasibility` is likewise a gate: 1.0 when `target.ev < 0.15 × acquirer.dry_powder`, decaying to 0 as EV approaches dry powder. (v0.2 described it as a soft gate in prose while writing it as an additive `w3` term — that inconsistency is resolved here in favour of the gate.)

Target-level fit is the **max over acquirers**, with the argmax stored as `likely_acquirer`. Store the top three pairs with their per-term breakdown, so the dossier can show why a given acquirer scores where it does — including that acquirer B was gated out while acquirer A cleared.

**Why this placement matters.** Both gates are properties of the *pair*, not of the target. A US-listed target that would face heavy scrutiny from an elevated-scrutiny-bloc acquirer but none from a US or allied one should surface as a **strong fit with the acquirers that clear**, not as a company carrying a generic penalty. Because the gates sit inside `fit()` and therefore inside the argmax, this falls out automatically: gated pairs collapse, ungated pairs survive, and `likely_acquirer` names an acquirer that could actually complete the transaction.

Where *every* plausible acquirer is gated, `max fit` is low, and that consequence propagates to the composite through the `strategic_fit` term (§5.1). No separate target-level penalty is needed or wanted — adding one would double-count.

There is no circularity. The gates depend only on acquirer attributes and target attributes, never on the fit score itself, so a single pass computes everything.

#### 4.4.1 Foreign-investment-screening gate

Foreign-investment screening is a property of the **acquirer**, not the target. CFIUS reviews inbound acquisitions of US businesses, so the relevant variable is the acquirer's nationality relative to a US target. v0.1 flagged `is_foreign_domiciled` on the target, which models the wrong side of the transaction — a large share of the §4.1 acquirer universe is itself foreign-domiciled, so that formulation penalised the wrong companies for the wrong reason.

A flat penalty on all foreign acquirers is equally wrong: acquisitions of US biotechs by Japanese, Swiss, UK, French, and Danish pharma clear routinely. Tier by bloc:

```yaml
fis_penalty_by_acquirer_bloc:
  us:                    0.00
  allied:                0.02   # JP, CH, UK, FR, DK, DE, IE, AU, CA
  neutral:               0.10
  elevated_scrutiny:     0.45   # CN, HK, RU, and jurisdictions under
                                # active outbound/inbound screening regimes
```

Apply an additional increment where the **target** holds US federal funding (BARDA/NIH contracts), genomic data on US persons, or manufacturing designated critical infrastructure. These are the fact patterns that actually trigger review, they are visible in 10-K risk factors and government-contract disclosures, and they interact with acquirer bloc — which is why the increment belongs in the pairwise gate rather than as a standalone target flag.

Target domicile survives as a **separate, mild, target-level** friction in Block E, covering inversion mechanics and cross-border tender-offer complexity. It is a real but much smaller effect and must not be conflated with the above.

#### 4.4.2 Antitrust gate

`antitrust_penalty(acquirer, target)` is inherently pairwise for the same reason: "dominant in the same indication" has no meaning without naming whose portfolio is dominant. Computed from overlap between the acquirer's marketed and late-stage products and the target's lead asset at the indication level, escalating where the acquirer already holds a large share of the treated population. A target with a first-in-class asset is nearly ungated for everyone; a target with a me-too asset is gated specifically against the incumbent leader and clean for everyone else.

This feature was miswired as a target-level penalty from v0.1 through v0.2 and is corrected here.

### 4.5 Block E — Feasibility Frictions (target-level penalties only)

**Invariant: every feature in this block must be single-valued per company.** If a feature's value depends on which acquirer you are asking about, it does not belong here — it is a gate in `fit()` (§4.4). Block E feeds the scalar `friction_penalty` in §5.1, which has no acquirer dimension; putting a pairwise quantity into it forces an implementer to invent an aggregation (average across acquirers, or use the Block D argmax), and either choice silently reintroduces the flat-penalty distortion that pairwise treatment exists to remove.

`fis_risk` and `antitrust_overlap` were in this table through v0.2 and violated the invariant. Both have moved to §4.4.1 and §4.4.2. Adding a new row here requires confirming it is genuinely acquirer-independent.

| Feature | Effect |
|---|---|
| `has_controlling_holder` | >30% single holder → deal needs their consent; flag, don't auto-penalise (can cut either way) |
| `has_poison_pill` | Parsed from 8-K / DEF 14A → penalty |
| `staggered_board` | Penalty |
| `dual_class_shares` | Strong penalty |
| `is_foreign_domiciled` | Target-side only: inversion mechanics, cross-border tender-offer complexity → mild penalty. **Not** foreign-investment screening, which is §4.4.1 |
| `recent_failed_process` | 8-K/press evidence of a terminated strategic review → penalty, decaying over 18 months |
| `royalty_encumbrance` | Existing royalty-financing deal on lead asset → penalty |


### 4.6 Block F — Ownership and Sentiment (context only, low weight)

Specialist-fund presence from 13F (maintain a list of ~30 healthcare-specialist CIKs), quarter-over-quarter change; Form 4 open-market insider buys; short interest as % of float. These are **contextual features for the dossier**. Cap their combined weight at 10% of the composite — 13F data is 45 days stale by construction and will leak lookahead if handled carelessly (use `known_from = filing date`, never `period_end`).

---

### 4.7 Block G — Revealed Process Signals

Blocks A–F infer latent attractiveness. Block G captures **revealed** process: cases where a sale is already underway or contractually pre-arranged. These features are categorically different in strength and must not be averaged into a weighted sum alongside fundamental features, where a 0.05 weight would dissolve them. Handling is specified in §5.2.

| Feature | Source | Definition |
|---|---|---|
| `process_state` | `corporate_process_event` | Current state: `none` / `rumored` / `disclosed_open` / `concluded_no_deal` |
| `process_scope` | ↑ | `whole_company` vs `asset_only` — the latter is far weaker |
| `days_since_process_open` | ↑ | Processes decay; conditional probability falls after ~12 months without an announcement |
| `has_13d_activist` | `activist_position` | 13D from a filer in `activist_filers.yaml` |
| `activist_intent_max` | ↑ | Strongest stated intent across open positions; `sale_demand` is the top rung |
| `activist_escalation` | ↑ | Count of 13D/A amendments — trajectory matters more than the initial filing |
| `has_strategic_toehold` | ↑ | 13D/13G filed by a **corporate** filer in the acquirer universe |
| `strategic_toehold_pct` | ↑ | Size of that stake |
| `partner_structure_max` | `partnership_structure` | Strongest structure with any acquirer-universe partner |
| `partner_equity_pct` | ↑ | Partner's equity stake in the target |
| `partner_identity` | ↑ | Overrides the §4.4 fit argmax when present — see §5.2 |

**Verification gate.** Every Block G feature reads only rows with `is_verified = TRUE`. Unverified keyword hits are visible in the review queue and in the dossier as "pending verification", but they do not enter the score. Given the weight these features carry, an unreviewed false positive would put a random company at rank 1.

**Bitemporal caution.** 13D is due within 5 business days of crossing 5%, so `known_from` is the filing date and the crossing date sits in the past — using the crossing date would leak. 13F is 45 days stale by construction; `known_from` must be the filing date, never `period_end`. This is the most likely place in the system for a subtle lookahead leak, and §8.3 must sample it specifically.

---

## 5. Scoring Model

### 5.1 Phase 1 — Transparent rule-based composite (ship this first)

```
fundamental = (
    0.30 * acquirer_pressure_matched     # Block A, matched via Block D
  + 0.25 * target_asset_quality          # Block B (PoA removed — see §4.2)
  + 0.20 * strategic_fit                 # Block D: max over acquirers of fit(),
                                         #   already net of the pairwise FIS and
                                         #   antitrust gates (§4.4.1, §4.4.2)
  + 0.15 * financial_attractiveness      # Block C
  - 0.10 * friction_penalty              # Block E: target-level frictions ONLY
) * dilution_gate

composite = apply_process_tier(fundamental, block_g)   # §5.2
```

Where `dilution_gate ∈ {0.5, 1.0}` — halved when `dilution_risk = TRUE`.

**No pairwise quantity enters this formula except through `strategic_fit`.** That term is the only acquirer-dependent input, and it arrives already reduced over the acquirer universe by the argmax in §4.4. Any attempt to add a second acquirer-dependent term to a per-company scalar is a defect — see the invariant in §4.5.

All block scores normalised to [0,1] via **cross-sectional rank percentile within the as-of universe**, not z-scores. Rank percentiles are robust to the fat tails endemic to this data.

Weights live in `config/weights.yaml`, versioned, and are inputs to `config_hash`.

### 5.2 Process signals are tiers, not weights

Block G features must not be folded into the weighted sum. Their conditional probabilities are an order of magnitude above what fundamentals produce, and a linear weight would either dissolve the signal (if small) or swamp every other feature (if large). Use tiering:

```python
def apply_process_tier(fundamental: float, g: BlockG) -> float:
    """Tiers are disjoint bands. Fundamental score orders WITHIN a tier."""
    if g.partner_structure_max == "acquisition_option":
        return 3.0 + fundamental          # Tier 3 — contractually pre-arranged
    if (g.process_state == "disclosed_open"
            and g.process_scope == "whole_company"
            and g.days_since_process_open < 365):
        return 3.0 + fundamental          # Tier 3 — sale process underway
    if (g.activist_intent_max == "sale_demand"
            or g.has_strategic_toehold
            or g.partner_structure_max in ("rofn_rofr", "equity_plus_commercial")):
        return 2.0 + fundamental          # Tier 2 — structural pressure or pre-positioning
    if (g.has_13d_activist
            or g.process_state == "rumored"
            or g.process_scope == "asset_only"):
        return 1.0 + fundamental          # Tier 1 — elevated
    return fundamental                    # Tier 0 — fundamentals only
```

Store `tier` explicitly in `score.subscores` and display it as a distinct badge in the dossier, so a user can see immediately whether a name ranks highly because a process is public or because the fundamental model likes it. Those are very different pieces of information and must never be presented as one number.

**Where a Block G partner is present, `partner_identity` overrides the §4.4 fit argmax** as `likely_acquirer`. A market-derived fit score should not out-argue a disclosed contractual relationship.

**Caveat the user must understand, and the dossier must state:** Tier 3 names are the *most likely to be acquired* and simultaneously the *least likely to be profitable entries*, because the market prices a disclosed process within minutes. Tier 3 is primarily useful as a label-generation and validation surface — it is where the model can be checked against reality. The economically interesting names are high-fundamental Tier 0 and Tier 1. This is not a defect in the ranking; it is the reason §5.3 exists.

### 5.3 Two targets, not one

The system's economic purpose is not to predict acquisitions. It is to find positions with favourable expected return. These diverge, and v0.1 conflated them.

Train and report **two** models:

| Model | Target | Use |
|---|---|---|
| `P(deal)` | Binary: acquisition announced within 24 months | Ranking, alerting, recall metrics |
| `E[return \| deal]` | Continuous: total return from `as_of` price to consideration value, including CVR at probability-weighted value | Filtering the ranked list |

The second is what makes the first actionable. A company acquired at a 15% premium after declining 60% over the holding period is a positive label for `P(deal)` and a substantial loss for the holder. Ranking on `P(deal)` alone systematically surfaces exactly this case, because distressed companies with failing pipelines are genuinely more likely to be sold.

Combine as: `expected_value = P(deal) × E[return | deal] + (1 − P(deal)) × E[return | no deal]`

The third term requires modelling the no-deal path — cash burn, dilution, and catalyst outcomes — and is where `lead_asset_poa` (§4.2) belongs. This is why PoA was removed from the deal-probability composite: it is a determinant of the standalone-path value, not of whether someone bids.

Report both metrics for every name in the dossier. Rank the default view by `expected_value`, with `P(deal)` available as an alternate sort.

### 5.4 Which score drives the rank — normative

v0.2 left this ambiguous and it must not be resolved by whoever writes the scoring job. Specified here:

- **`rank_by_expected_value` is the default ordering** for the watchlist, the API, and the dossier index. It is the only ranking that reflects the system's economic purpose.
- **`rank_by_composite` is persisted alongside it** and drives the tier-visibility view — "what does the fundamental model like, and which names has process confirmed."
- **Never persist a bare `rank` column.** Any code path emitting an unqualified `rank` is a defect.

The reason this matters is mechanical, not stylistic. Under §5.2, any Tier ≥ 1 name has `composite ≥ 1.0` by construction, while every Tier 0 name is bounded below 1.0. Ranking on `composite` therefore sorts strictly by tier first and buries every Tier 0 name beneath every Tier 1 name regardless of quality — which inverts the intended use, since §5.2 states that the economically interesting names are high-fundamental Tier 0 and Tier 1.

`expected_value` has no such structural discontinuity: a Tier 3 name whose disclosed process is already priced will carry a low `expected_return_if_deal` and fall in the default view, exactly as intended.

Both ranks appear in the JSON output (§6.1) and both are recomputed per `score_run`. Sorting in the UI switches between them; it never recomputes either.

### 5.5 Phase 2 — Calibrated model (only after Phase 1 backtests cleanly)

Logistic regression or gradient-boosted trees on the labelled deal dataset (§2.5).

- **Label:** `1` if the company was the target of an acquisition announcement within 24 months of `as_of`, else `0`.
- **Sampling:** monthly snapshots of the full eligible universe, 2016–2025. Expect a positive rate of roughly 3–6% per 24-month window; handle imbalance with class weights, not oversampling.
- **Validation:** walk-forward with an **embargo period of 24 months** between train and test to prevent label leakage across the forward window. Standard k-fold is invalid here and must not be used.
- **Interpretability requirement:** SHAP values per prediction, persisted to `score.contributions`. A rank with no explanation is not shippable.

Do not skip Phase 1. If the rule-based version does not beat the **naive-informed** baseline (§0.3) on Tier 0 names specifically, the data or the labels are wrong, and an ML model will only launder that error.

---

## 6. Outputs

### 6.1 Ranked watchlist

JSON + CSV, written per `score_run`:

```json
{
  "as_of": "2026-08-30",
  "model_version": "0.2.0",
  "config_hash": "a3f9...",
  "universe_size": 700,
  "default_sort": "expected_value",
  "results": [
    {
      "rank_by_expected_value": 1,
      "rank_by_composite": 34,
      "ticker": "XXXX",
      "company_name": "...",
      "composite": 0.87,
      "tier": 0,
      "tier_reason": null,
      "p_deal_24m": 0.14,
      "expected_return_if_deal": 0.62,
      "expected_value": 0.09,
      "enterprise_value_usd": 1840000000,
      "ev_to_risk_adjusted_npv": 0.31,
      "cash_runway_months": 26,
      "lead_asset": {
        "name": "...",
        "phase": "Phase 3",
        "therapeutic_area": "oncology_solid",
        "modality": "adc",
        "poa": 0.52,
        "next_catalyst": {"type": "topline", "expected": "2027-Q1", "confidence": "medium"}
      },
      "likely_acquirers": [
        {"name": "...", "fit": 0.81, "rationale": "revenue_at_risk_3y 0.22 in oncology_solid; dry_powder 41B"}
      ],
      "frictions": ["staggered_board"],
      "subscores": {"acquirer_pressure": 0.79, "target_quality": 0.62,
                    "strategic_fit": 0.81, "financial": 0.71, "friction": 0.15}
    }
  ]
}
```

### 6.2 Company dossier

Markdown or HTML, one per top-N company. Sections: identity and capital structure; asset table with trial detail and PoA derivation; cash and runway with the arithmetic shown; acquirer-fit table; friction list; catalyst timeline; **source links for every number**. No figure appears without a resolvable `source_url`.

### 6.3 Change alerts

Diff consecutive runs. Emit on:
- Rank change ≥ 20 positions
- New entry into top 50
- `dilution_risk` flips to TRUE
- New Phase III completion or PDUFA acceptance detected
- **Tier promotion** (§5.2) — any movement from Tier 0/1 into Tier 2/3. This is the highest-priority alert class
- **New verified `disclosed_open` process event**, or a new SC 13D from a filer in `activist_filers.yaml`, or a 13D from a corporate filer in the acquirer universe
- Any 8-K Item 1.01 or SC 14D9 filed by a universe member (deal announcement — the label event)

Delivery: webhook + email. Alerting must be idempotent — re-running a day's job must not re-fire alerts.

---

## 7. Technical Requirements

### 7.1 Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Orchestration | Prefect 2 or Dagster | Dagster preferred for asset-based lineage |
| Storage — raw | S3 / local filesystem, partitioned `source/date/` | Immutable, gzipped JSON |
| Storage — warehouse | Postgres 15+ (or DuckDB for single-user) | |
| Transform | dbt-core | Models mirror §3 tables |
| HTTP | `httpx` with a shared rate-limited client | |
| Validation | Pydantic v2 for schemas, Great Expectations or `pandera` for data quality | |
| Modelling | scikit-learn, optionally LightGBM; SHAP for attribution | |
| API | FastAPI | Read-only |
| Config | YAML + Pydantic Settings, hashed into `config_hash` | |

### 7.2 Rate limiting and retries

Single shared token-bucket limiter per host. SEC = 10 rps hard cap. Exponential backoff with jitter on 429/403/5xx, max 5 attempts. Every failed fetch after retries is logged to a `fetch_failure` table and surfaced in the run report — silent failures are the enemy.

### 7.3 Idempotency

Every job is keyed on `(source, entity, as_of_date)` and safe to re-run. Raw-zone writes are content-addressed; identical payloads are deduplicated by hash.

### 7.4 Observability

Structured JSON logs. Per-run metrics: rows ingested by source, entity-resolution match rate, feature null rate by feature, wall-clock by stage. **Alert if any feature's null rate rises above its historical mean by more than 3σ** — this is how you catch an upstream schema change before it silently corrupts a score.

---

## 8. Testing

### 8.1 Unit
Every feature function with hand-constructed fixtures, including the null path.

### 8.2 Data quality (`pandera` / Great Expectations)
- `cash_runway_months` ∈ [0, 120] or null
- `enterprise_value` may be negative (legitimate); market cap may not
- `loe_date` ∈ [1990-01-01, as_of + 25 years]
- `lead_asset_poa` ∈ [0, 1]
- No `company` row without at least one verified alias

### 8.3 Lookahead audit — mandatory

An automated test that, for a sample of 200 (company, as_of) pairs, asserts every fact used in scoring satisfies `known_from <= as_of`. Wire it into CI. **Any failure blocks the build.** This test exists because lookahead bias in a backtest produces a beautiful, entirely fictional result, and nothing downstream will reveal the error.

The sample must be **stratified to guarantee coverage of the three highest-risk sources**, not drawn uniformly — uniform sampling over a universe dominated by SEC facts will rarely hit the cases that actually leak:

1. **Vendor-sourced facts** (§2.4) — market cap, shares outstanding, segment revenue. These feed `dilution_gate` and are the likeliest silent leak, because vendors restate history in place.
2. **13F holdings** — assert `known_from` equals the filing date and never `period_end`. A 45-day error here is invisible in output and fatal in backtest.
3. **13D and process events** (§4.7) — assert `known_from` equals the filing date, never the beneficial-ownership crossing date or the date a process is later revealed to have begun.

Add a dedicated regression test asserting that no `corporate_process_event` row with `is_verified = FALSE` ever reaches a score computation.

### 8.4 Backtest harness
Walk-forward, monthly snapshots, 24-month embargo. Reports lift@k, precision@k, recall@k against both baselines from §0.3, plus a calibration plot for Phase 2.

---

## 9. Delivery Milestones

| # | Deliverable | Definition of done |
|---|---|---|
| **M1** | Ingest + storage | SEC, CT.gov, openFDA, Orange Book landing in raw zone daily; bitemporal schema live; rate limits respected |
| **M2** | Entity resolution | ≥95% of universe companies mapped across all three sources; review queue tooling functional |
| **M3** | Feature store | All Block A–F features computing as-of any date; lookahead audit passing with stratified sampling |
| **M4** | Rule-based scoring | Phase 1 composite ranks the universe; dossiers generate with full source attribution |
| **M5** | Block G — process signals | 8-K phrase detection + 13D ingest + review queue live; tiering applied; verification gate enforced |
| **M6** | Labels + backtest | Deal dataset built, hand-verified, `deal_type` classified with reverse mergers excluded; walk-forward harness reports lift@k against all three baselines |
| **M7** | Return model | `E[return \| deal]` trained; `expected_value` becomes the default ranking |
| **M8** | Calibrated model | Only if M6 shows the rule-based version beats the naive-informed baseline |
| **M9** | Partnership structures | Second-pass manual enrichment on the top 200 by composite |
| **M10** | API + alerts | FastAPI read endpoints; idempotent change alerts |

M1–M5 are the minimum viable system — **Block G is now in the MVP**, because a screen that cannot see a disclosed sale process is missing the highest-precision signal in the domain and will be visibly wrong to any domain user on first inspection.

M6 determines whether the fundamental premise holds. If lift against the naive-informed baseline is at 1.0, stop and reconsider the feature set rather than proceeding. Note that Block G will inflate headline metrics without validating the fundamental model at all — **report backtest metrics separately for Tier 0 names and for all names.** Tier 0 performance is the honest measure of whether Blocks A–F work.

---

## 10. Known Risks

| Risk | Mitigation |
|---|---|
| Orange Book ↔ product-revenue join is manual-heavy | Scope explicitly; start with top 25 acquirers only, ~15 products each |
| CT.gov sponsor names are messy | Human-in-the-loop review queue from day one; never auto-accept fuzzy matches |
| Label set is small (~500 events) | Keep Phase 1 rule-based; heavy regularisation in Phase 2; resist deep models |
| Survivorship bias in market data | Vendor must supply delisted tickers; verify before signing |
| Base rates drift as FDA policy changes | `base_rates.yaml` versioned with citation dates; annual review task |
| Regime change makes history unrepresentative | Report backtest metrics by year, not pooled — a model that only worked in 2020–21 must be visible as such |
| Vendor data redistribution limits | Maskable export layer, config-gated |
| Block G phrase matching produces false positives at high weight | Verification gate: unverified rows never score. Review queue SLA of 1 business day |
| **Block G is confirmatory, not first-mover — by design** | See note below. Document in the dossier UI; do not let Tier 2/3 promotion be read as an entry signal |
| Tier 3 names are already priced; ranking looks impressive but is not actionable | Dual-target design (§5.3); report Tier 0 metrics separately (§9) |
| Reverse mergers invert the Block B signal if mislabelled | Mandatory `deal_type` classification with detection heuristics (§2.5); reviewer checklist |
| Partnership-structure extraction is manual and unbounded | Scoped to top 200 by composite as a second pass, not a full-universe ingest |
| Base-rate YAML incomplete for most therapeutic areas | `base_rate_fallback` flag surfaced in every run report; long fallback list blocks release |

---

**On Block G latency.** The verification gate (§4.7) plus a one-business-day review SLA (§3.4) means Block G will essentially never be first to react to its own triggering event. §5.2 notes the market reprices a disclosed process within minutes; a human confirming an 8-K the following morning is not competing on that axis and should not pretend to.

This is a deliberate tradeoff, not an oversight. At the weight these features carry, a single unreviewed false positive puts an arbitrary company at rank 1 and destroys the user's trust in the whole list. Latency is the cheaper failure.

The consequence to state plainly in the UI and the dossier: **Tier 2/3 promotion is a confirmatory and monitoring signal, not a source of alpha on the initial print.** Its uses are (a) validating that the fundamental model was already surfacing the name before the process became public — the single most informative feedback loop in the system, and (b) tracking a known process toward resolution. Anyone reading a tier promotion as "buy this now" has misunderstood the output, and the interface should make that hard to do.

**On CVR valuation — an open gap, and not purely M7-scoped.** §5.3 specifies including contingent value rights "at probability-weighted value," but nothing in the spec estimates that probability. This matters more than it looks: CVRs frequently pay zero or partial value, so treating `cvr_max_per_share` as a known input systematically overstates realised deal returns.

The reviewer's suggestion to defer this to M7 is half right. The **model** can wait; the **label convention cannot**, because M6 builds the return targets that M7 trains on. Whatever CVR assumption is implicit in the labels gets baked into every downstream estimate.

Therefore, decide at M6 and record it in `config/cvr_policy.yaml`:

- **v1 convention (recommended): value CVRs at zero** and store `upfront_per_share` as the return basis, with `cvr_max_per_share` retained but unused. This is conservative, unambiguous, and cannot silently inflate backtest returns.
- Persist `cvr_realized_per_share` on the `deal` table as outcomes become known, building the dataset a future sub-model needs.
- At M7, if that dataset supports it, replace the zero convention with an estimate. The natural approach reuses the Block B trial-outcome machinery, since most CVR milestones are clinical or regulatory events with the same base rates — but it needs its own calibration and its own section, not a line in §5.3.

Any change to `cvr_policy.yaml` invalidates existing labels and forces a full re-score. Wire it into `config_hash`.

---

## 11. Required Disclaimer

Every exported artifact — watchlist, dossier, alert, API response — must carry:

> This output is generated by an automated screening tool for informational and research purposes only. It is not investment advice, not a recommendation to buy or sell any security, and has not been prepared in accordance with any regulatory standard for investment research. Scores reflect statistical base rates and heuristics, not predictions. Acquisition outcomes are uncertain and most screened companies will not be acquired. Investing in clinical-stage biotechnology carries a substantial risk of total loss of capital. Consult a licensed financial advisor before making any investment decision.

Implement as a template constant, injected by the export layer, not by individual writers.
