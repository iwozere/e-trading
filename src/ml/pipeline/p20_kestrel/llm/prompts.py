"""
P20 Kestrel — LLM prompt templates.

All prompt strings are module-level constants.
Input trimming happens before calls; never pass whole 10-Ks here.
"""

SYSTEM_ANALYST = (
    "You are a quantitative equity research analyst. "
    "Respond only with valid JSON matching the schema provided. "
    "Be concise and factual. Do not add commentary outside the JSON."
)

# ---------------------------------------------------------------------------
# 8-K / PR classification
# ---------------------------------------------------------------------------

CLASSIFY_8K = """\
Classify the following SEC 8-K filing excerpt for a company being monitored
as a potential investment candidate.

Ticker: {ticker}
Filing date: {filed_date}
Form type: {form_type}

--- FILING EXCERPT ---
{text}
--- END EXCERPT ---

Return a JSON object with exactly these fields:
{{
  "event_type": "<one of: earnings_beat, earnings_miss, earnings_inline, guidance_raise, guidance_cut, \
guidance_reiterated, buyback_auth, executive_change, m_and_a_announcement, activist_disclosure, \
product_approval, regulatory_risk, restructuring, debt_refinancing, other>",
  "materiality": "<high|medium|low>",
  "thesis_impact": "<invalidates|neutral|supports>",
  "one_liner": "<single sentence summarizing the filing>"
}}
"""

# ---------------------------------------------------------------------------
# Candidate dossier
# ---------------------------------------------------------------------------

DOSSIER = """\
Generate a concise investment dossier for the following potential turnaround candidate.

Ticker: {ticker}
Score: {score}/100 (interim mode: {interim_mode})
Drawdown from 2-year high: {drawdown:.1%}
Market cap: ${mcap_b:.1f}B
Sector: {sector}

--- RECENT FILINGS SUMMARY ---
{filings_summary}

--- RECENT FINANCIALS ---
{financials_summary}

--- SENTIMENT SIGNALS ---
{sentiment_summary}

Return a JSON object with exactly these fields:
{{
  "thesis": "<2-3 sentence bull case>",
  "bull_cases": ["<bull case 1>", "<bull case 2>", "<bull case 3>"],
  "bear_cases": ["<bear case 1>", "<bear case 2>", "<bear case 3>"],
  "red_flags": [
    {{"flag": "<description>", "source": "<filing or data source>"}}
  ],
  "catalysts_ahead": ["<catalyst 1>", "<catalyst 2>"],
  "verdict": "<advance|watch|reject>",
  "reject_reason": "<required if verdict=reject, else null>",
  "confidence": "<high|medium|low>",
  "invalidation_line": "<single sentence: what would break the thesis>",
  "sources_used": ["<source 1>", "<source 2>"]
}}
"""

# ---------------------------------------------------------------------------
# 10-K/Q risk diff
# ---------------------------------------------------------------------------

RISK_DIFF = """\
Compare the risk factor sections of two consecutive SEC filings for the same company.

Ticker: {ticker}
Current filing: {current_form} dated {current_date}
Previous filing: {prev_form} dated {prev_date}

--- CURRENT RISK FACTORS (EXTRACT) ---
{current_risks}

--- PREVIOUS RISK FACTORS (EXTRACT) ---
{prev_risks}

Return a JSON object:
{{
  "risks_added": ["<new risk 1>", "<new risk 2>"],
  "risks_removed": ["<removed risk 1>"],
  "risks_escalated": ["<risk that worsened materially>"],
  "red_flags": ["<specific concern for an investor>"],
  "overall_assessment": "<improved|neutral|deteriorated>"
}}
"""

# ---------------------------------------------------------------------------
# Guidance-delta proxy (Sleeve A interim mode, §4.2.1)
# ---------------------------------------------------------------------------

GUIDANCE_DELTA_PROXY = """\
Analyze the management guidance language in these two earnings call 8-K excerpts
for signs of improving or deteriorating forward outlook.

Ticker: {ticker}
Current quarter 8-K (more recent): {current_date}
{current_guidance}

Previous quarter 8-K: {prev_date}
{prev_guidance}

Return a JSON object:
{{
  "delta_score": <integer from -5 to +5, positive = improving guidance>,
  "confidence": "<high|medium|low>",
  "key_phrases_positive": ["<phrase indicating improvement>"],
  "key_phrases_negative": ["<phrase indicating deterioration>"],
  "one_liner": "<one sentence summary>"
}}
"""

# ---------------------------------------------------------------------------
# Form 10 (spin-off) dossier
# ---------------------------------------------------------------------------

FORM10_DOSSIER = """\
Summarize this SEC Form 10 or S-11 registration for a newly spun-off company.

Ticker: {ticker} (SpinCo)
Parent: {parent_ticker}
Filing date: {filed_date}

--- FORM 10 EXTRACT (Business, Risk Factors, MD&A) ---
{text}

Return a JSON object:
{{
  "business_summary": "<2 sentences on what the SpinCo does>",
  "competitive_position": "<1 sentence on moat or lack thereof>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "financial_snapshot": "<revenue, profitability, leverage if mentioned>",
  "independence_readiness": "<high|medium|low: how ready is the SpinCo to operate standalone>",
  "investment_angle": "<1 sentence on why it might be interesting post-spin>"
}}
"""
