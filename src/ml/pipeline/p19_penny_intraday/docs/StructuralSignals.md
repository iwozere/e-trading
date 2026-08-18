# P19 — Structural Signals Reference

Companion to `pipeline-specification-v2.md` §7. Defines every negative (N) and
positive (P) signal in Layer 0: what it measures, which filing carries it, how it is
computed, the threshold, its grade impact, and — most importantly — **how it fails**.

Scope: cap-structure integrity and dilution urgency. Not business quality, not
valuation.

---

## 1. Grading model (recap)

Two axes, never blended:

- **`structural_grade` ∈ {A, B, C, D}** — D suppresses alerts, C forces
  `FADE_CANDIDATE` on any momentum trigger, A/B are alertable.
- **`dilution_urgency` ∈ [0, 100]** — continuous, feeds alert copy and calibration.
- **`insider_conviction` ∈ [0, 100]** — continuous, lifts B → A.

Three rules that govern everything below:

1. **No positive signal overrides a D disqualifier.** Dilution mechanics are close to
   deterministic; conviction signals are probabilistic. The asymmetry is intentional.
2. **Unknown grades C, never A.** A signal that could not be resolved is not a signal
   that came back clean.
3. **Every signal carries a resolved/unresolved flag.** `coverage` is the fraction
   resolved, and it is surfaced in the alert. Alerts say *"no disqualifiers found
   (coverage 0.82)"* — never *"clean"*.

---

## 2. Coverage by issuer type — read this first

Coverage is not uniform, and the gaps are structural rather than random. **The least
transparent issuers are systematically the ones your signals cannot see**, which is
exactly the population most likely to appear on a penny-stock momentum watchlist.

| Issuer type | Form 4 | 8-K | XBRL facts | Realistic coverage |
|---|---|---|---|---|
| Domestic 10-K/10-Q filer | ✅ | ✅ | ✅ | 0.85–1.00 |
| **Foreign private issuer (20-F / 6-K)** | ❌ **exempt** | ❌ **files 6-K instead** | ⚠️ partial, IFRS tags | **0.30–0.55** |
| Recent IPO (< 8 quarters) | ✅ | ✅ | ⚠️ no 8q history | 0.55–0.75 |
| Smaller reporting company | ✅ | ✅ | ⚠️ reduced disclosure | 0.70–0.90 |

**The FPI gap is the single most important caveat in this document.** Foreign private
issuers are exempt from Section 16, so **no Form 4 exists** — every insider signal
(P1, P2, P10, N14) is permanently unresolvable, not merely missing. They file 6-K
instead of 8-K, so item-numbered triggers (N7, N13) do not exist either; the
equivalent disclosure is unstructured prose inside a 6-K. Your GRSD example is exactly
this shape: Jersey-domiciled, 6-K filer.

Consequences, all of which are already encoded in the v2 spec:

- FPIs will cluster at coverage ≈ 0.4 and therefore at grade C via N17. **This is
  correct behaviour, not a bug** — but it means the C bucket contains two very
  different populations (genuinely risky domestic filers, and opaque-by-structure
  FPIs). Track them separately in calibration or the grade-vs-`close_retention` test
  in §15 will be confounded.
- Consider a distinct `structural_grade = "U"` (unprofilable) if the FPI share of the
  watchlist exceeds ~20%. Treat it operationally like C, report it separately.
- Do not attempt to compensate by loosening thresholds for FPIs. The absence of
  insider data is information about verifiability, not evidence of cleanliness.

---

## 3. Negative signals

### N1 — Reverse split within 24 months · **Grade D**

**Measures:** the strongest single negative marker available. A company that
engineered its share price to hold a listing has demonstrated both the need and the
willingness; recurrence rates are high.

**Source:** corporate-action split history (ratio < 1.0). Corroborate with 8-K
**item 5.03** (amendments to articles) and the FINRA daily list.

**Computation:** any split with ratio < 1.0 dated within 24 months.

**Failure modes:**
- Vendor split histories are unreliable for recently uplisted and OTC-graduated names,
  and are sometimes silently backfilled. Cross-check 5.03 rather than trusting one
  source.
- **FPIs do not file 8-K 5.03**, so corroboration is vendor-only for them.
- A pending-but-not-yet-effective reverse split is invisible in split history and is
  arguably a *stronger* negative than a completed one. Catch it via 8-K 5.03 /
  DEF 14A proxy item text.

**Calibration note:** test the window. 24 months is a starting assumption; the
recurrence curve may justify 36.

---

### N2 — Any reverse split in listed history · **Grade C**

Same detection, no window. Weaker but persistent — a company that has done it once has
revealed something about its financing trajectory. Downgrade only, never disqualify.

---

### N3 / N4 — Share-count growth · **D at CAGR > 25% / C at 10–25%**

**Measures:** the actual dilution rate, as opposed to the *capacity* to dilute. This is
the denominator that silently kills most microcap theses — a business that triples
while share count triples returns nothing.

**Source:** XBRL cover-page shares outstanding (`dei:EntityCommonStockSharesOutstanding`),
one point per filing, from `companyfacts`.

**Computation:** CAGR over the trailing 8 quarters of the cover-page series.

**Failure modes — two serious ones:**

1. **⚠️ Reverse splits make the raw series discontinuous.** An unadjusted series shows
   share count *falling* across a reverse split, producing a negative CAGR — i.e. the
   most diluted names in the universe score as the cleanest, and can even pick up P3.
   **The series must be split-adjusted before the CAGR is computed.** This is the most
   dangerous single defect available in Layer 0, because it inverts the signal on
   precisely the names N1 exists to catch. Assert it in tests.
2. Cover-page counts are *as of a date near filing*, not period-end, and the gap is
   irregular. Fine for multi-quarter CAGR; do not use for precise point-in-time float.

Recent IPOs lack 8 quarters → null → contributes to low coverage → C via N17.

---

### N5 — Floating-rate / toxic convertible outstanding · **Grade D**

**Measures:** the death-spiral structure. Conversion priced at a discount to trailing
VWAP means share issuance scales with price weakness and the holder is economically
indifferent to — or served by — decline. A price rise converts mechanically into
supply. No momentum thesis survives this.

**Source:** full-text search scoped to CIK, **latest annual and interim filings only**.

**Phrase set (starting point, must be tuned):** *lowest VWAP*, *conversion price equal
to*, *discount to the market price*, *variable conversion price*, *lowest trading
price*, *percentage of the lowest*.

**Failure modes:**
- **Recency scoping is essential.** Filings routinely describe converts that have been
  retired or restructured. Unscoped matching produces false D grades on names that
  cleaned up their balance sheet — the most expensive error class here, since D is a
  hard suppression.
- Boilerplate risk-factor language describing convertibles *in general* matches the
  same phrases. Restrict to the debt/convertible footnote sections where possible.
- FPI equivalents in 20-F use different phrasing; the phrase set will under-fire.

**Calibration note:** open question 6 in the spec. Hand-label ~50 filings before
trusting this signal. Until precision is measured, consider firing N5 at grade C
rather than D.

---

### N6 — Going-concern qualification · **Grade D**

**Measures:** the auditor's own statement that survival is in doubt. Near-perfect
precision when present.

**Source:** full-text search, latest annual: *substantial doubt about its ability to
continue as a going concern*.

**Failure modes:**
- Appears both in the audit opinion and in a management footnote, sometimes with
  mitigating-plans language attached. Presence is sufficient; do not attempt to parse
  the mitigation.
- Watch for negation — *alleviated substantial doubt*, *no longer substantial doubt*.
  Require the affirmative form.
- No standard XBRL tag exists. Text search is the only route.
- Stale by up to 12 months by construction.

---

### N7 — Active exchange deficiency notice · **Grade D**

**Measures:** the precursor to N1. A bid-price or market-value compliance notice is the
mechanical cause of most reverse splits, typically 6–12 months ahead.

**Source:** 8-K **item 3.01**, trailing 12 months, with no subsequent
compliance-regained 8-K.

**Failure modes:**
- Compliance-regained disclosures are inconsistently filed; a stale-open notice may
  read as active when it is resolved. Cross-check against price — a name trading well
  above $1 for 30+ sessions has almost certainly cured a bid-price deficiency.
- Distinguish deficiency types: bid price, market value of listed securities,
  stockholders' equity, late filing. Late-filing deficiencies (Rule 5250(c)) are a
  different and arguably worse signal — they mean the rest of Layer 0 is running on
  stale data.
- **FPIs: no 8-K.** Unresolvable.

---

### N8 — Offering document filed within 5 trading days · **Grade C (D if within 2)**

**Measures:** active selling, right now. The most time-precise negative available.

**Source:** filing index, form types `424B5`, `424B3`, `424B4`, `S-1`, `S-1/A`,
`S-3` takedowns.

**Note:** `424B5` is the actual takedown off an existing shelf — the selling event
itself, not the authorisation. This is the form that matters most.

**Failure modes:**
- 424B filings also cover resale registrations for existing holders, which is dilution
  of a different character (no new shares, but new float). Both are supply; treat
  alike, but tag the distinction for calibration.
- Filed after the close, so a same-day pump can precede the visible filing by hours.
  The intraday EFTS poll (spec §9) exists for exactly this.

---

### N9 — Shelf capacity > 30% of market cap · **Grade C**

**Measures:** how much can be sold without further authorisation.

**Source and computation — see §5, which gives a text-parse-free method for
sub-$75M-float issuers.**

**Failure modes:** for issuers above $75M public float, remaining capacity genuinely
requires prospectus text parsing. Until then, fall back to the boolean *shelf exists
and was used within 8 quarters*.

---

### N10 — ATM used in either of last two quarters · **Grade C**

**Measures:** demonstrated willingness, which is what separates a dormant shelf from a
live financing program. An issuer that sold stock last quarter will sell into this
pump.

**Source:** cash-flow statement, `us-gaap:ProceedsFromIssuanceOfCommonStock`, from
`companyfacts`.

**Computation:** proceeds > 0 in either of the last two quarters, with **no discrete
offering 8-K in the same period** (which would indicate a marketed deal rather than
dribble-out ATM sales).

**Failure modes:**
- ⚠️ **The tag also captures option and warrant exercises**, which are not ATM sales.
  Threshold on magnitude — require proceeds above ~1–2% of market cap — or the signal
  fires on nearly every issuer with employees.
- ⚠️ **10-Q cash-flow figures are cumulative year-to-date for many filers.** A Q2 10-Q
  reports six months, Q3 reports nine. **The series must be de-cumulated into discrete
  quarters** before any quarter-over-quarter logic, or Q4 will look like a spike on
  every name in the universe. Same trap applies to N11's burn calculation.
- Tag variants exist (`ProceedsFromIssuanceOfCommonStockNetOfIssuanceCosts`, and
  ATM-specific extensions). Build a fallback chain.

---

### N11 — Cash runway < 3 quarters · **Grade C (D if < 1.5 with active shelf)**

**Measures:** need. The multiplier that converts capacity into probability.

**Source:** `us-gaap:CashAndCashEquivalentsAtCarryingValue` ÷ quarterly operating cash
burn.

**Computation:**
```
cash        = CashAndCashEquivalents (+ ShortTermInvestments where present)
burn        = |min(0, discrete quarterly NetCashProvidedByUsedInOperatingActivities)|
              averaged over trailing 4 discrete quarters
runway_q    = cash / burn          (null if burn ≤ 0 → see P5/P6)
```

**Failure modes:**
- The de-cumulation trap from N10 applies identically and matters more here.
- Tag fallback chain needed:
  `CashAndCashEquivalentsAtCarryingValue` →
  `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents` (includes restricted
  cash — overstates availability, flag it) → IFRS equivalents for FPIs.
- **Ignores undrawn credit facilities**, which for a leveraged name changes the picture
  entirely — in both directions. An issuer with a drawn revolver and covenant pressure
  faces *equity* issuance pressure that runway alone understates.
- Capex-heavy issuers: consider free-cash-flow burn rather than operating burn.
- Cash is as-of quarter-end, so up to ~4 months stale at the moment of use. A name at
  1.5 quarters of reported runway may already have raised — or already be out.

---

### N12 — Warrant overhang > 20% of float, strike within 50% of price · **Grade C**

**Measures:** latent supply that activates precisely on a price spike. Warrants struck
above market become exercisable during exactly the moves this pipeline detects.

**Source:** warrant tables in annual/interim filings. **No standard XBRL tag** — text
parsing required.

**Status:** defer to Phase 3. Nullable; contributes to reduced coverage until built.

---

### N13 — 8-K item 3.02 (unregistered equity sale) within 30 days · **Grade C**

**Measures:** PIPE or private placement — new shares issued, typically at a discount,
usually with registration rights that convert to float within months. Frequently
paired with warrants (feeding N12) and sometimes with the convertible structures N5
targets.

**Failure modes:** FPIs unresolvable. Some small issuances (advisor shares, settlement
of payables) are immaterial — threshold on disclosed size where available.

---

### N14 — Clustered insider sales / Form 144 within 90 days · **Grade C**

**Source:** Form 4 with `transactionCode = S`, `acquiredDisposedCode = D`,
non-derivative; plus Form 144 notices of proposed sale.

**Failure modes:**
- ⚠️ **Sales are far weaker evidence than purchases.** Insiders sell for diversification,
  tax, and liquidity reasons that carry no information. Weight accordingly — this is a
  minor C contributor, not a near-disqualifier.
- **Exclude code `F`** (shares withheld for tax on vesting) and code `M`+`S` pairs
  (option exercise-and-sell). These are mechanical and dominate raw sale counts. A
  naive `S` count will flag essentially every company with an equity comp plan.
- **10b5-1 plan sales are pre-scheduled** and carry near-zero information. Form 4 has
  carried a plan-adoption checkbox since the 2022 amendments — use it to exclude.
- FPIs: unresolvable.

---

### N15 — Recent IPO + micro float + FPI reporting · **Grade C**

**Measures:** a structural pattern — not a jurisdiction — that recurs across a
well-documented population of Nasdaq microcap pump-and-collapse episodes: listed
within 18 months, float under ~5M shares, reporting as a foreign private issuer, thin
sponsor and market-maker support.

**Computation:** conjunction of all three. Any one alone is not a signal.

**Failure modes:** genuine small IPOs match. This is a base-rate flag that lowers a
grade; it must never disqualify on its own. Its real function is forcing coverage
honesty — these names have poor filing coverage by construction and should not be
graded A.

---

### N16 — Auditor quality · **Grade C**

**Source:** audit opinion signature block; the PCAOB **Form AP** database maps auditors
to issuers and engagement partners, and PCAOB publishes inspection reports and
disciplinary orders.

**Computation:** auditor absent from a maintained whitelist, or subject to a recent
PCAOB sanction, or auditing an implausibly large number of microcap issuers relative
to firm size.

**Status:** Phase 3. High signal value, moderate maintenance cost (the whitelist needs
periodic review).

---

### N17 — Coverage < 0.4 · **Grade C**

**Measures:** unverifiability itself.

The signal that prevents the failure mode where the least transparent issuers grade
best because nothing fired against them. Given §2, this will fire on most FPIs. That is
the intended behaviour.

**Threshold note:** 0.4 is a starting assumption. Set it after measuring the actual
coverage distribution across the watchlist — if the median domestic filer resolves at
0.85 and the median FPI at 0.45, the threshold should sit between the modes rather
than at a round number.

---

## 4. Positive signals

Positive signals lift B → A and feed `insider_conviction`. They never override a D.

### P1 — Clustered insider open-market buying · **highest weight**

**Definition:** ≥ 3 distinct insiders, `transactionCode = P`,
`acquiredDisposedCode = A`, non-derivative, within a 30-day window.

**Why it dominates:** it is the only signal here where an informed party takes personal
financial risk with no alternative explanation. Diversification, tax, and liquidity
explain sales; nothing comparable explains coordinated open-market buying. Clustering
across multiple individuals is what distinguishes signal from one person's idiosyncratic
portfolio decision.

**Requirements:**
- **Code `P` only.** Exclude `A` (grants), `M` (option exercise), `F` (tax withholding),
  `G` (gifts). A count that includes grants measures compensation policy, not conviction.
- Exclude 10b5-1 pre-scheduled purchases via the plan checkbox.
- Weight by count of *distinct individuals*, not transaction count — one insider
  filing five times is one signal.
- Weight secondarily by size relative to that individual's disclosed compensation. A
  $15k purchase from a CEO on $2M is noise; $400k is not.

**Failure modes:** FPIs — permanently unresolvable. Also, buying is a *long-horizon*
signal; whether it predicts anything on an intraday spike horizon is an open empirical
question (spec §15, question 4). Test it and drop it if flat rather than assuming it
transfers.

---

### P2 — Any officer/director open-market buy in 90 days · **high**

Same mechanics, single-buyer, wider window. Weaker than P1 by roughly the degree that
one observation is weaker than three.

---

### P3 — Share count flat or declining over 8 quarters · **high**

**Measures:** revealed non-need. A microcap that has not issued stock in two years has
demonstrated it can fund itself — which in this universe is genuinely uncommon.

**⚠️ Depends entirely on the split-adjustment fix in N3.** Without it, this signal
fires on reverse-split names, which is the exact inversion described above. If you
implement one assertion in Layer 0's test suite, make it this one.

---

### P4 — Executed buyback · **high**

`us-gaap:PaymentsForRepurchaseOfCommonStock` > 0. **Authorised ≠ executed** — microcap
buyback authorisations are frequently announced and never used, and the announcement
alone is a promotional signal rather than a structural one. Require cash actually spent.

---

### P5 — Positive operating cash flow · **medium**

Any magnitude above zero. In this universe the sign matters more than the level: it
converts `runway_quarters` to undefined-in-a-good-way and removes the mechanical
compulsion to dilute.

---

### P6 — Runway > 8 quarters or net cash positive · **medium**

Derived from N11. Net cash positive (cash > total debt) is materially stronger than a
long runway and should be weighted separately.

---

### P7 — No dilution event in 24 months · **medium**

Absence of 424B*, S-1, S-3 takedowns, 8-K 3.02. Absence-of-evidence caveat: verify
coverage was adequate over the window before crediting it. A name with 0.3 coverage
has not demonstrated anything by not tripping N8.

---

### P8 — Institutional accumulation · **medium**

New or increased 13G/13D positions across two consecutive quarters. 13G indicates
passive accumulation above 5%; 13D indicates activist intent — different signals,
worth separate treatment. Full 13F reverse-lookup is a heavier quarterly job; defer to
Phase 3.

**Caveat:** at these market caps a 5% position is small in absolute dollars, and some
13G filers are market makers or index-adjacent rather than conviction holders.

---

### P9 — No debt maturity within 24 months · **medium**

Maturity schedule from annual filings; nullable and often unstructured. A near-dated
maturity in a company without cash is an equity-issuance event with a known date —
arguably this belongs on the negative side as a scheduled dilution catalyst, and is
worth testing in both directions.

---

### P10 — Insider ownership > 20% and stable · **low**

Aggregate from Forms 3/4. Alignment signal. Stability matters more than level — a high
but declining stake is a negative reading of the same number.

---

### P11 — High short interest, **conditional on grade A/B** · context only

**This is the conditionality that matters most in the whole model.**

High SI% with rising days-to-cover is squeeze fuel *on a defensible float* and
distribution fuel *on a dilution machine* — an issuer with an active ATM can supply
every share the shorts need, at prices they choose, and the "squeeze" becomes the
financing event. Identical raw number, opposite meaning.

v1 added squeeze score unconditionally into severity, which systematically
over-scored the worst names in the universe. Under v2, SI contributes to
`insider_conviction` only at grade A/B; at C/D it is displayed as context and, if
anything, **raises** `dilution_urgency`.

---

## 5. Shelf capacity without text parsing — the baby-shelf rule

This resolves spec open question 7 for the majority of the watchlist.

Form S-3 **General Instruction I.B.6** — the "baby shelf" limitation — restricts
issuers with public float below **$75M** to primary offerings of no more than
**one-third of public float in any trailing 12-month period**.

For a penny-stock watchlist, nearly every name sits below that threshold. So remaining
capacity is computable arithmetically:

```
if public_float_value < $75M:
    annual_capacity   = public_float_value / 3
    used_ttm          = Σ proceeds from primary offerings, trailing 12 months
    remaining         = max(0, annual_capacity - used_ttm)
    capacity_pct_mcap = remaining / market_cap
```

Properties worth noting:

- **Capacity scales with float value, so it moves with price.** A name that doubles
  intraday roughly doubles its legal capacity to sell — which is a direct mechanical
  reason spikes attract offerings, and a genuinely useful thing for the alert to say.
  A momentum trigger that *simultaneously increases dilution capacity* is a materially
  different object from one that does not.
- Float value is recomputed continuously from price; the $75M test is applied as-of
  specific measurement dates in practice, so treat proximity to the threshold as
  uncertain rather than binary.
- Above $75M float the limitation lifts and true capacity requires the prospectus —
  fall back to the boolean.
- Non-S-3-eligible issuers (delinquent filers, recent IPOs under the 12-month
  seasoning requirement) use S-1 instead, which has no percentage cap but is slower
  and more visible. Absence of an S-3 is not absence of dilution risk.

**Recommended sequencing:** fit `dilution_urgency` against the
`dilution_event_within_5d` label using this arithmetic estimate first. Only build the
prospectus parser if the estimate proves to be the limiting factor.

---

## 6. Derived metrics

### `dilution_urgency` ∈ [0, 100]

```
dilution_urgency = 100 × clip(
      w_r · runway_pressure          # 1 − clip(runway_quarters / 8, 0, 1)
    + w_c · shelf_capacity_pct_mcap  # clipped at 1.0, from §5
    + w_h · recent_usage_rate        # offerings in trailing 8q / 8
    + w_d · dilution_history         # split-adjusted share-count CAGR, clipped
    , 0, 1)
```

Bootstrap weights `w_r=0.35, w_c=0.25, w_h=0.25, w_d=0.15` — placeholders only. Fit
against `dilution_event_within_5d`, which is a hard binary label and makes this a
clean supervised problem rather than a judgement call.

**Design intent:** capacity alone is weak (most microcaps have a shelf). Need alone is
weak (a broke company with no registration statement cannot move quickly). The product
of capacity × need × demonstrated willingness is what predicts selling into strength.

### `insider_conviction` ∈ [0, 100]

Weighted sum of P1–P10, normalised. **Renormalise over resolved signals only** — an
FPI with no Form 4 data must not score low on conviction *because* the data is absent;
it should score null and route to C via coverage. Scoring absence as evidence conflates
two different things and will corrupt the calibration in §15 question 4.

### Grade assignment

```
D  ← any D-severity disqualifier
C  ← any C-severity disqualifier, or coverage < 0.4
B  ← no disqualifiers, insider_conviction < 40
A  ← no disqualifiers, insider_conviction ≥ 40, dilution_urgency < 30
```

Thresholds 40 and 30 are placeholders pending §15 calibration.

---

## 7. Consolidated trap list

The defects most likely to silently invert a signal, in descending order of damage:

1. **Unadjusted share-count series across reverse splits** — inverts N3 and P3, causing
   the most-diluted names to grade cleanest. Assert in tests.
2. **Cumulative YTD cash-flow figures used as discrete quarters** — corrupts N10 and
   N11 for every filer, with a systematic Q4 artefact.
3. **Unscoped full-text matching for N5** — false D grades on names that retired their
   converts. D is a hard suppression, so precision matters more than recall here.
4. **Treating FPI unresolvability as clean** — the §2 problem. Guarded by N17, but only
   if coverage is computed honestly per-signal rather than assumed.
5. **Counting Form 4 codes A/M/F as insider activity** — grants and tax withholding
   swamp genuine code-P buying by an order of magnitude.
6. **Option exercises counted as ATM usage in N10** — needs a magnitude threshold.
7. **Scoring absence of data as a low positive score** rather than null — corrupts
   `insider_conviction` and makes the calibration unreadable.
8. **Stale-open deficiency notices in N7** — compliance-regained filings are
   inconsistent; cross-check against price.

---

## 8. Implementation priority

Ordered by signal value ÷ implementation cost, for Phase 1.5 sequencing.

**Tier 1 — build first (deterministic, high coverage, cheap):**
N1, N3/N4, N11, N8, P1/P2, P3, P5 — plus the split-adjustment fix, which is a
prerequisite for two of them rather than a signal in its own right.

**Tier 2 — cheap, needs one lookup each:**
N7, N10, N13, N14, P4, P6, P7, and the §5 shelf-capacity arithmetic.

**Tier 3 — text parsing or external datasets, defer to Phase 3:**
N5 (needs the labelled precision study first), N6, N9 above the baby-shelf threshold,
N12, N16, P8, P9.

**Always on, from day one:** N17 and per-signal coverage accounting. Without them
every grade above C is unfalsifiable.

---

*Reference documentation for a monitoring tool. Signal grades are classifications of
filing evidence, not investment advice.*
