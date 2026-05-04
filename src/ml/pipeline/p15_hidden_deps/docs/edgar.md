# EDGAR Submissions — Download Strategy

No, downloading for all companies daily is definitely not needed. Let's break down the logic.

---

## Why 8-K filings are in the pipeline

The pipeline searches for **sector-level lag correlations**, not individual stock predictions. So 8-K filings are only needed in two cases:

1. **Macro-significant events** — earnings surprises from the largest companies that move the entire sector (JPM earnings → entire XLF)
2. **RAG context** — the agent needs to understand what was happening in a sector before generating a hypothesis

Monitoring all 10K+ companies daily is clear overkill.

---

## Real architecture: two tiers

### Tier 1 — Watchlist (daily, ~50–80 companies)

Only systemically important companies — those whose 8-K filings actually move the sector:

```python
TIER1_WATCHLIST = {

    # ── XLF: Financials ─────────────────────────────────────────────────────
    # BRK-B (11.75%), JPM (11.23%), V (7.20%), MA (5.76%), BAC (4.58%),
    # GS (3.59%), WFC (3.49%), C (2.81%), MS (2.80%), AXP (2.29%)
    # + key sub-sector representatives: BLK (asset mgmt), SPGI (ratings), CME (exchange)
    "XLF": [
        "BRK-B",  # Berkshire — systemic bellwether
        "JPM",    # largest US bank, earnings move entire sector
        "V",      # payments — rate-insensitive growth within financials
        "MA",     # payments duopoly
        "BAC",    # rate-sensitive commercial bank
        "GS",     # capital markets / investment banking cycle
        "WFC",    # consumer banking, mortgage exposure
        "C",      # global wholesale bank, EM exposure
        "MS",     # wealth management + capital markets
        "AXP",    # consumer credit, spending indicator
        "BLK",    # largest asset manager, AUM as market sentiment proxy
        "SPGI",   # S&P Global — credit ratings, financial data
        "CME",    # exchange — volatility → volume → revenue
        "USB",    # US Bancorp — large regional bank proxy
        "PGR",    # Progressive — insurance cycle indicator
    ],

    # ── XLE: Energy ──────────────────────────────────────────────────────────
    # XOM (22%), CVX (17%), COP (7%), WMB (4.4%), SLB (4.5%),
    # EOG (4%), PSX (3.8%), VLO (3.7%), KMI (3.7%), MPC (3.6%)
    # + oilfield services (HAL), E&P pure-play (OXY), midstream (OKE)
    "XLE": [
        "XOM",    # largest US oil major, global integrated
        "CVX",    # second major, Permian + international
        "COP",    # pure E&P, price-sensitive upstream
        "WMB",    # Williams Cos — natural gas midstream/pipeline
        "SLB",    # oilfield services leader, capex cycle proxy
        "EOG",    # Permian pure-play E&P, shale bellwether
        "PSX",    # Phillips 66 — refining margin indicator
        "VLO",    # Valero — crack spread proxy
        "KMI",    # Kinder Morgan — gas pipeline infrastructure
        "MPC",    # Marathon Petroleum — refining
        "OXY",    # Occidental — Buffett holding, Permian
        "HAL",    # Halliburton — oilfield services #2
        "BKR",    # Baker Hughes — services + LNG equipment
        "OKE",    # ONEOK — gas gathering/processing
        "FANG",   # Diamondback Energy — Permian pure-play
    ],

    # ── XLK: Technology ──────────────────────────────────────────────────────
    # NVDA (14.78%), AAPL (12.14%), MSFT (9.23%), AVGO (6.03%),
    # MU (4.32%) — weights shift frequently due to AI rally
    # + enterprise software (CRM, ORCL), semis (AMD, QCOM, TXN), infra (ACN)
    "XLK": [
        "NVDA",   # AI/GPU — dominant signal for AI capex cycle
        "AAPL",   # largest market cap, consumer hardware + services
        "MSFT",   # cloud (Azure) + enterprise software
        "AVGO",   # Broadcom — networking chips, AI accelerators
        "MU",     # Micron — memory cycle, leading indicator for semis
        "AMD",    # CPU/GPU competitor to Intel/NVDA
        "ORCL",   # Oracle — enterprise cloud, database
        "CRM",    # Salesforce — enterprise SaaS spending indicator
        "ACN",    # Accenture — IT services, consulting capex
        "QCOM",   # Qualcomm — mobile chips, handset cycle
        "TXN",    # Texas Instruments — analog semis, industrial demand
        "AMAT",   # Applied Materials — semiconductor equipment
        "PLTR",   # Palantir — government AI/defense tech
        "NOW",    # ServiceNow — enterprise workflow automation
        "PANW",   # Palo Alto Networks — cybersecurity cycle
    ],

    # ── XLV: Health Care ─────────────────────────────────────────────────────
    # LLY (14%), JNJ (10%), ABBV (7%), UNH (6%), MRK (5%),
    # AMGN (3.5%), TMO (3.5%), ISRG (3.2%), ABT (3.1%), GILD (3.1%)
    "XLV": [
        "LLY",    # Eli Lilly — GLP-1/obesity drugs, dominant weight
        "JNJ",    # Johnson & Johnson — diversified, defensive
        "ABBV",   # AbbVie — Humira/Skyrizi, income proxy
        "UNH",    # UnitedHealth — managed care, insurance cycle
        "MRK",    # Merck — Keytruda oncology, vaccines
        "AMGN",   # Amgen — biotech large-cap
        "TMO",    # Thermo Fisher — lab equipment, biotech capex proxy
        "ISRG",   # Intuitive Surgical — robotic surgery, procedure volume
        "ABT",    # Abbott Labs — devices + diagnostics
        "GILD",   # Gilead — HIV/oncology, cash flow story
        "BSX",    # Boston Scientific — cardiac devices
        "SYK",    # Stryker — orthopedic implants, elective surgery
        "BMY",    # Bristol-Myers Squibb — oncology pipeline
        "CVS",    # CVS Health — PBM + retail pharmacy + insurance
        "HCA",    # HCA Healthcare — hospital utilization indicator
    ],

    # ── XLI: Industrials ─────────────────────────────────────────────────────
    # CAT (7%), GE (6.6%), RTX (5%), GEV (4.4%), BA (3.3%),
    # UBER (3%), UNP (2.9%), DE (2.9%), HON (2.9%), ETN (2.6%)
    "XLI": [
        "CAT",    # Caterpillar — global construction/mining capex
        "GE",     # GE Aerospace — commercial aviation cycle
        "RTX",    # RTX Corp — defense + aircraft engines
        "GEV",    # GE Vernova — power generation, energy transition
        "BA",     # Boeing — commercial/defense aviation
        "UNP",    # Union Pacific — rail freight, economic activity
        "DE",     # Deere — agricultural capex, commodity cycle
        "HON",    # Honeywell — industrial conglomerate
        "ETN",    # Eaton — electrical components, data center power
        "LMT",    # Lockheed Martin — defense budget indicator
        "NOC",    # Northrop Grumman — defense
        "UPS",    # UPS — parcel volume, consumer + B2B indicator
        "EMR",    # Emerson Electric — automation/process control
        "CSX",    # CSX — eastern rail freight
        "PWR",    # Quanta Services — grid infrastructure buildout
    ],

    # ── XLB: Materials ───────────────────────────────────────────────────────
    # LIN (17%), NEM (7.3%), SHW (6.2%), FCX (5.3%), CRH (5%),
    # ECL (4.8%), APD (4.7%), CTVA (4.8%), MLM (4.4%), NUE (3.5%)
    "XLB": [
        "LIN",    # Linde — industrial gases, largest weight
        "NEM",    # Newmont — gold mining, safe-haven proxy
        "SHW",    # Sherwin-Williams — housing/construction indicator
        "FCX",    # Freeport-McMoRan — copper, China demand proxy
        "CRH",    # CRH — cement/construction materials
        "APD",    # Air Products — industrial gases, hydrogen
        "ECL",    # Ecolab — water treatment, specialty chemicals
        "CTVA",   # Corteva — agricultural chemicals/seeds
        "MLM",    # Martin Marietta — aggregates, construction
        "NUE",    # Nucor — steel, manufacturing demand
        "DOW",    # Dow Inc — commodity chemicals
        "PPG",    # PPG Industries — coatings, auto/industrial
        "VMC",    # Vulcan Materials — aggregates, infrastructure
        "ALB",    # Albemarle — lithium, EV battery supply chain
        "CF",     # CF Industries — nitrogen fertilizers, nat gas spread
    ],

    # ── XLU: Utilities ───────────────────────────────────────────────────────
    # NEE (13.8%), SO (7.3%), DUK (6.9%), CEG (6.5%), AEP (5.1%),
    # SRE (4.2%), VST (3.8%), D (3.7%), XEL (3.4%), EXC (3.4%)
    "XLU": [
        "NEE",    # NextEra — renewable energy leader, rate-sensitive
        "SO",     # Southern Company — regulated, nuclear
        "DUK",    # Duke Energy — large regulated utility
        "CEG",    # Constellation Energy — nuclear, AI power demand
        "AEP",    # American Electric Power — transmission grid
        "SRE",    # Sempra Energy — gas utility + LNG export
        "VST",    # Vistra — power generation, merchant energy
        "D",      # Dominion Energy — regulated, rate-sensitive
        "XEL",    # Xcel Energy — renewables transition
        "EXC",    # Exelon — nuclear + regulated distribution
        "PCG",    # PG&E — California utility, wildfire risk proxy
        "ED",     # Consolidated Edison — NYC utility, stable
        "EIX",    # Edison International — California utility
        "ETR",    # Entergy — nuclear + regulated South
        "FE",     # FirstEnergy — mid-Atlantic regulated
    ],

    # ── XLP: Consumer Staples ────────────────────────────────────────────────
    # WMT (11.9%), COST (9.4%), PG (7.4%), KO (6.5%), PM (5.5%),
    # CL (4.75%), PEP (4.7%), MO (4.7%), MDLZ (4.4%), MNST (3.7%)
    "XLP": [
        "WMT",    # Walmart — consumer spending bellwether
        "COST",   # Costco — membership model, affluent consumer
        "PG",     # Procter & Gamble — household staples pricing power
        "KO",     # Coca-Cola — global beverage, defensive
        "PM",     # Philip Morris — international tobacco, EM exposure
        "PEP",    # PepsiCo — beverages + snacks (Frito-Lay)
        "MO",     # Altria — domestic tobacco, high yield
        "CL",     # Colgate-Palmolive — oral/personal care
        "MDLZ",   # Mondelez — global snack foods
        "MNST",   # Monster Beverage — energy drinks growth story
        "TGT",    # Target — discretionary/staples overlap
        "KR",     # Kroger — grocery chain, food inflation proxy
        "GIS",    # General Mills — packaged food
        "KHC",    # Kraft Heinz — packaged food, pricing pressure
        "STZ",    # Constellation Brands — beer/wine/spirits
    ],

    # ── XLRE: Real Estate ────────────────────────────────────────────────────
    # WELL (10.3%), PLD (9.1%), EQIX (7.25%), AMT (5.8%), DLR (4.8%),
    # SPG (4.6%), CBRE (4.5%), VTR (4.4%), O (4.4%), PSA (3.5%)
    "XLRE": [
        "WELL",   # Welltower — healthcare REIT, aging demographics
        "PLD",    # Prologis — industrial/warehouse REIT, e-commerce
        "EQIX",   # Equinix — data center REIT, AI infrastructure
        "AMT",    # American Tower — cell tower REIT
        "DLR",    # Digital Realty — data center REIT
        "SPG",    # Simon Property Group — retail/mall REIT
        "CBRE",   # CBRE Group — CRE services, transaction volume
        "VTR",    # Ventas — senior housing + medical office
        "O",      # Realty Income — net lease, monthly dividend
        "PSA",    # Public Storage — self-storage
        "CCI",    # Crown Castle — cell tower infrastructure
        "EQR",    # Equity Residential — apartment REIT, rents
        "AVB",    # AvalonBay — apartment REIT, coastal markets
        "ARE",    # Alexandria RE — life science lab REIT
        "VICI",   # VICI Properties — gaming/entertainment REIT
    ],

    # ── XLY: Consumer Discretionary ─────────────────────────────────────────
    # AMZN (21-28% depending on date), TSLA (15-20%), HD (5-7%),
    # TJX (4%), MCD (4%), BKNG (4%), LOW (3%), SBUX (2.3%), ORLY (2.1%)
    "XLY": [
        "AMZN",   # Amazon — e-commerce + AWS, dominant weight
        "TSLA",   # Tesla — EV cycle, volatile weight
        "HD",     # Home Depot — housing/renovation indicator
        "MCD",    # McDonald's — consumer spending health
        "BKNG",   # Booking Holdings — travel demand
        "TJX",    # TJX Companies — value retail, trade-down indicator
        "LOW",    # Lowe's — housing/DIY, rate sensitivity
        "SBUX",   # Starbucks — discretionary spending indicator
        "ORLY",   # O'Reilly Auto — auto aftermarket, aging fleet
        "NKE",    # Nike — global consumer brand, China exposure
        "CMG",    # Chipotle — fast casual restaurant cycle
        "ABNB",   # Airbnb — short-term rental, travel
        "LVS",    # Las Vegas Sands — Macau/Singapore gaming
        "GM",     # General Motors — auto cycle, EV transition
        "F",      # Ford — auto + EV, labor cost indicator
    ],

    # ── XLC: Communication Services ──────────────────────────────────────────
    # META (14-23%), GOOGL (8.6%), GOOG (6.9%), DIS (4.6%), CMCSA (4.5%),
    # NFLX (5.9%), T (5%), VZ (4.7%), TMUS (4.6%), WBD (4.7%)
    "XLC": [
        "META",   # Meta — digital advertising, AI investment cycle
        "GOOGL",  # Alphabet A — search + cloud + YouTube
        "GOOG",   # Alphabet C — same economic exposure
        "NFLX",   # Netflix — streaming, subscriber growth
        "T",      # AT&T — telecom, high yield, capex
        "VZ",     # Verizon — wireless, dividend yield proxy
        "TMUS",   # T-Mobile — wireless subscriber growth
        "DIS",    # Disney — streaming + parks + content
        "CMCSA",  # Comcast — cable + NBC + streaming
        "WBD",    # Warner Bros Discovery — media/streaming
        "EA",     # Electronic Arts — gaming cycle
        "TTWO",   # Take-Two Interactive — GTA cycle
        "LYV",    # Live Nation — live events, concert demand
        "CHTR",   # Charter Communications — cable broadband
        "OMC",    # Omnicom — advertising industry indicator
    ],
}

# Flat list for the submissions API
ALL_TIER1_TICKERS = sorted(set(
    ticker
    for tickers in TIER1_WATCHLIST.values()
    for ticker in tickers
))

print(f"Total Tier-1 tickers: {len(ALL_TIER1_TICKERS)}")
# → Total Tier-1 tickers: 162 (accounting for GOOGL/GOOG duplicates — ~160 unique companies)
```

For these ~160 companies, downloading submissions.json **daily** means ~160 requests — takes seconds.

### Tier 2 — Full universe (on demand / quarterly)

Remaining companies are fetched only when the agent generates a hypothesis that involves them, or quarterly to refresh companyfacts.

---

## What to check in submissions.json daily

The file is ~50–200 KB (not 5 MB like companyfacts). Only `filings.recent` is needed, filtered by `form` and `filedAt`:

```python
import requests
import json
from datetime import date, timedelta
from pathlib import Path

HEADERS = {"User-Agent": "YourProject your@email.com"}

def check_new_8k(cik: str, since_days: int = 1) -> list[dict]:
    """
    Check whether a company filed an 8-K in the last N days.
    submissions.json contains the last ~1000 filings — sufficient for daily monitoring.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    data = r.json()

    recent = data["filings"]["recent"]
    cutoff = (date.today() - timedelta(days=since_days)).isoformat()

    results = []
    for i, form in enumerate(recent["form"]):
        filed = recent["filedAt"][i][:10]  # "2025-04-30T..."
        if form == "8-K" and filed >= cutoff:
            results.append({
                "cik":        cik,
                "ticker":     data.get("tickers", ["?"])[0],
                "filed_date": filed,
                "accession":  recent["accessionNumber"][i],
                "items":      recent["primaryDocument"][i],
            })
    return results

def daily_8k_scan(watchlist_ciks: dict) -> list[dict]:
    """Scan the Tier-1 watchlist for yesterday's filings."""
    all_events = []
    for sector, ciks in watchlist_ciks.items():
        for cik in ciks:
            events = check_new_8k(cik, since_days=1)
            for e in events:
                e["sector"] = sector
            all_events.extend(events)
            time.sleep(0.12)  # SEC 10 req/sec limit
    return all_events
```

Result: a small JSON of yesterday's events. If empty — do nothing. If non-empty — download only those specific filing documents.

---

## Which 8-K items actually matter

Not all 8-Ks are equally significant. SEC requires filers to declare the Item number:

```python
MATERIAL_ITEMS = {
    "2.02": "Results of Operations",         # EARNINGS — most important
    "7.01": "Regulation FD Disclosure",      # guidance, forward-looking statements
    "8.01": "Other Events",                  # often profit warnings
    "5.02": "Departure/Appointment CEO",     # management change
    "1.01": "Material Definitive Agreement", # M&A, major contracts
    "1.03": "Bankruptcy or Receivership",    # bankruptcy
}

# Skip low-significance items:
SKIP_ITEMS = {
    "5.07",  # Results of vote at shareholder meeting
    "9.01",  # Financial statements (just attachments)
    "4.02",  # Non-reliance on financial statements (technical)
}
```

```python
def is_material_8k(items_str: str) -> bool:
    """items_str arrives as '2.02,9.01' or simply '2.02'"""
    items = [i.strip() for i in items_str.split(",")]
    return any(item in MATERIAL_ITEMS for item in items)
```

---

## Final architecture

```
Daily (morning):
  submissions.json × ~160 Tier-1 companies     ← ~160 requests, ~20 sec
  → filter: only 8-K filed yesterday
  → filter: only material items (2.02, 7.01...)
  → download text of only those filings (~0–5 on a normal day, 50+ in earnings season)
  → FinBERT → write to edgar_events table

Quarterly:
  companyfacts.json × all companies in the universe
  → update financial metrics (EPS, revenue, margins)

On agent demand:
  submissions.json for any company over the needed period
  → RAG context for hypothesis generation
```

```sql
CREATE TABLE edgar_events (
    filed_date      DATE,
    ticker          VARCHAR,
    sector          VARCHAR,
    form_type       VARCHAR,   -- '8-K'
    items           VARCHAR,   -- '2.02,9.01'
    is_earnings     BOOLEAN,
    sentiment_score DOUBLE,    -- FinBERT if text was downloaded
    accession_no    VARCHAR,
    url             VARCHAR
);
```

**Conclusion:** check ~160 companies daily (~160 lightweight HTTP requests), and download + process filing text only for actual material events — on average 0–5 files per day outside earnings season and 20–40 during earnings season (4 times per year, ~3 weeks each).
