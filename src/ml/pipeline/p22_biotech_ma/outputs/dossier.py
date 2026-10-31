"""
P22 — company dossier (spec §6.2, M4), 2026-10-30.

One Markdown file per top-N company of a score run. Sections follow the spec: identity and capital
structure; asset table with trial detail; cash and runway with the arithmetic shown; acquirer fit;
frictions; catalyst timeline; process signals. Every figure is read as of the run's `as_of` through
`FeatureContext`, so a dossier shows what the score saw.

**"No figure appears without a resolvable `source_url`":** a number is printed only beside a link.
Figures whose stored row has no `source_url` are left out of the tables and listed under "Omitted
(no source)" at the end, so a gap is visible instead of silently filled. Derived figures (market cap,
EV, runway) are shown as arithmetic over linked inputs. Scores and ranks are model outputs, not facts,
and carry no link. Prices link to the vendor's public history page for the ticker.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features import block_b, block_c
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.ingest.market_cap import compute_market_cap

DISCLAIMER = (
    "Research screen only. Not investment advice, not a price prediction and not a prediction of clinical "
    "outcomes (spec §11)."
)
_EDGAR_COMPANY_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
_CTGOV_URL = "https://clinicaltrials.gov/study/{nct_id}"
_PRICE_URL = "https://finance.yahoo.com/quote/{ticker}/history"
_RUNWAY_PERIODS = 4


@dataclass
class _Doc:
    """Markdown being assembled plus the figures left out for lack of a source."""

    lines: List[str] = field(default_factory=list)
    omitted: List[str] = field(default_factory=list)

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def heading(self, text: str) -> None:
        self.lines.extend(["", f"## {text}", ""])


def _money(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    v = abs(value)
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= scale:
            return f"{sign}${v / scale:,.1f}{unit}"
    return f"{sign}${v:,.0f}"


def _num(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _fact_cell(doc: _Doc, row: Optional[Dict[str, Any]], label: str, money: bool = True) -> Optional[str]:
    """`value ([source](url), period end)` or None; a row without a source_url goes to `omitted`."""
    if row is None or row.get("value") is None:
        return None
    if not row.get("source_url"):
        doc.omitted.append(f"{label} ({row.get('source_id')}, period end {row.get('period_end')})")
        return None
    value = float(row["value"])
    text = _money(value) if money else _num(value, 0)
    return f"{text} ([source]({row['source_url']}), period end {row.get('period_end')})"


def _identity(doc: _Doc, company: Dict[str, Any], repo: Any, ctx: FeatureContext) -> None:
    company_id = company["company_id"]
    cik = company.get("cik")
    doc.heading("Identity and capital structure")
    doc.add(f"- Name: {company.get('name')}",
            f"- Ticker / exchange: {company.get('ticker') or '-'} / {company.get('exchange') or '-'}",
            f"- CIK: [{cik}]({_EDGAR_COMPANY_URL.format(cik=cik)})" if cik else "- CIK: -",
            f"- SIC: {company.get('sic_code') or '-'}")
    shares_row = ctx.get_latest_fact_row(company_id, "shares_outstanding")
    shares = _fact_cell(doc, shares_row, "shares_outstanding", money=False)
    if shares:
        doc.add(f"- Shares outstanding: {shares}")
    result, reason = compute_market_cap(repo, company_id, ctx.as_of)
    ticker = company.get("ticker")
    if result is not None and shares and ticker:
        price = result.market_cap / result.shares_outstanding if result.shares_outstanding else None
        doc.add(f"- Market cap: {_money(result.market_cap)} = close {_num(price)} on {result.price_trade_date} "
                f"([prices]({_PRICE_URL.format(ticker=ticker)})) × {_num(result.shares_outstanding, 0)} shares"
                + (f" (split factor {result.split_factor:g} since the share count)" if result.split_factor != 1.0 else ""))
    elif result is None:
        doc.add(f"- Market cap: not computable as of {ctx.as_of} ({reason})")
    for metric in ("cash_and_equivalents", "short_term_investments", "total_debt"):
        cell = _fact_cell(doc, ctx.get_latest_fact_row(company_id, metric), metric)
        if cell:
            doc.add(f"- {metric.replace('_', ' ').capitalize()}: {cell}")
    ev = block_c.enterprise_value(company_id, ctx.as_of, ctx)
    if ev is not None:
        doc.add(f"- Enterprise value: {_money(ev)} = market cap − (cash + short-term investments) + debt")


def _assets(doc: _Doc, trials: Sequence[Dict[str, Any]], as_of: date, ctx: FeatureContext, company_id: int) -> None:
    doc.heading("Assets and trials")
    if not trials:
        doc.add("No linked trials (trials reach a company only through single-intervention linkage).")
        return
    lead = block_b.lead_asset(list(trials))
    if lead is not None:
        poa = block_b.lead_asset_poa(company_id, as_of, ctx)
        doc.add(f"Lead asset (furthest phase): asset {lead['asset_id']}, {lead['therapeutic_area']}, "
                f"{block_b.PHASE_RANK_LABELS[lead['phase_rank']]}; probability of approval "
                f"{_num(poa) if poa is not None else 'not computed (only for Phase I and earlier, decision 13)'}", "")
    doc.add("| Trial | Asset | Therapeutic area | Phase | Status | Primary completion |", "|---|---|---|---|---|---|")
    for t in sorted(trials, key=lambda t: (str(t.get("phase") or ""), str(t.get("nct_id"))), reverse=True):
        nct = t.get("nct_id")
        link = f"[{nct}]({_CTGOV_URL.format(nct_id=nct)})" if nct else "-"
        doc.add(f"| {link} | {t.get('asset_name') or t.get('asset_id')} | {t.get('therapeutic_area') or '-'} | "
                f"{t.get('phase') or '-'} | {t.get('status') or '-'} | {t.get('primary_completion_date') or '-'} |")


def _runway(doc: _Doc, company_id: int, ctx: FeatureContext) -> None:
    doc.heading("Cash and runway")
    cash_row = ctx.get_latest_fact_row(company_id, "cash_and_equivalents")
    sti_row = ctx.get_latest_fact_row(company_id, "short_term_investments")
    burn_rows = ctx.repo.get_financial_facts_as_of(company_id, "quarterly_opex_burn", ctx.as_of)[:_RUNWAY_PERIODS]
    cash = _fact_cell(doc, cash_row, "cash_and_equivalents")
    if cash is None or not burn_rows:
        doc.add("Runway not computable: cash or operating cash flow history missing.")
        return
    sti = _fact_cell(doc, sti_row, "short_term_investments")
    burn_cells = [c for c in (_fact_cell(doc, r, "quarterly_opex_burn") for r in burn_rows) if c]
    doc.add(f"- Cash: {cash}", f"- Short-term investments: {sti or '$0 (none on file)'}",
            f"- Operating cash flow, last {len(burn_cells)} quarters:")
    doc.add(*[f"  - {c}" for c in burn_cells])
    runway = block_c.cash_runway_months(company_id, ctx.as_of, ctx)
    if runway is None:
        doc.add("- Runway: not applicable (operating cash flow is not negative on average)")
    else:
        doc.add(f"- Runway: (cash + short-term investments) ÷ average quarterly burn × 3 = **{runway:.1f} months**")
    dilution = block_c.dilution_risk(company_id, ctx.as_of, ctx)
    doc.add(f"- Dilution risk (runway < 12 months with a readout inside it): "
            f"{'yes' if dilution == 1.0 else 'no' if dilution == 0.0 else 'unknown'}")


def _fit(doc: _Doc, score: Dict[str, Any]) -> None:
    doc.heading("Acquirer fit")
    likely = (score.get("subscores") or {}).get("likely_acquirer")
    if likely:
        doc.add(f"Likely acquirer: **{likely.get('name')}** (from {likely.get('source')})", "")
    pairs = (score.get("contributions") or {}).get("top_pairs") or []
    if not pairs:
        doc.add("No acquirer pair could be scored.")
        return
    doc.add("| Acquirer | Fit | Preference terms | Gates | Unknown gates |", "|---|---|---|---|---|")
    for p in pairs:
        terms = ", ".join(f"{k} {_num(v)}" for k, v in (p.get("terms") or {}).items() if v is not None) or "-"
        gates = ", ".join(f"{k} {_num(v)}" for k, v in (p.get("gates") or {}).items() if v is not None) or "-"
        doc.add(f"| {p.get('acquirer_name')} | {_num(p.get('fit'))} | {terms} | {gates} | {', '.join(p.get('unknown_gates') or []) or '-'} |")


def _frictions(doc: _Doc, score: Dict[str, Any]) -> None:
    doc.heading("Frictions")
    features = (score.get("subscores") or {}).get("features") or {}
    foreign = features.get("is_foreign_domiciled")
    if foreign == 1.0:
        doc.add("- Foreign domicile (foreign-investment screening and antitrust exposure)")
    else:
        doc.add("None detected. Staggered board, poison pill and the other governance frictions are not extracted yet.")


def _catalysts(doc: _Doc, trials: Sequence[Dict[str, Any]], as_of: date) -> None:
    doc.heading("Catalyst timeline")
    upcoming = sorted((t for t in trials if t.get("primary_completion_date") and t["primary_completion_date"] >= as_of),
                      key=lambda t: t["primary_completion_date"])
    if not upcoming:
        doc.add("No upcoming primary completion dates on file.")
        return
    for t in upcoming:
        nct = t.get("nct_id")
        doc.add(f"- {t['primary_completion_date']}: primary completion of [{nct}]({_CTGOV_URL.format(nct_id=nct)}) "
                f"({t.get('phase') or '-'}, {t.get('therapeutic_area') or '-'}), {(t['primary_completion_date'] - as_of).days} days out")


def _pct_of_class(value: Any) -> str:
    return "stake not parsed" if value is None else f"{float(value):.1f}% of class"


def _process(doc: _Doc, company_id: int, score: Dict[str, Any], ctx: FeatureContext) -> None:
    doc.heading("Process signals (Block G, verified only)")
    doc.add(f"Tier {score.get('tier')} ({score.get('tier_reason')})", "")
    rows = [("8-K process event", e.get("event_date"), f"{e.get('state')} / {e.get('scope')}", e.get("source_url"))
            for e in ctx.get_verified_process_events(company_id)]
    rows += [(f"{p.get('form_type')} by {p.get('filer_name')} ({p.get('filer_type') or 'unclassified'})", p.get("filed_date"),
              _pct_of_class(p.get("pct_of_class")), p.get("source_url"))
             for p in ctx.get_verified_activist_positions(company_id)]
    if not rows:
        doc.add("No verified process events or 13D positions.")
        return
    for what, when, detail, url in sorted(rows, key=lambda r: str(r[1]), reverse=True):
        if not url:
            doc.omitted.append(f"{what} on {when}")
            continue
        doc.add(f"- {when}: {what}, {detail} ([filing]({url}))")


def render_dossier(repo: Any, company: Dict[str, Any], score: Dict[str, Any], as_of: date, run_meta: Dict[str, Any]) -> str:
    """
    Build one company's dossier as Markdown.

    Args:
        repo: A `P22Repo`-shaped object (read-only).
        company: The `p22_company` row.
        score: The company's score row from `score_universe` (`composite`, `tier`, `subscores`, `contributions`).
        as_of: The score run date; every figure is read as of it.
        run_meta: `{"config_hash", "model_version", "run_id"}` for the header.
    """
    ctx = FeatureContext(as_of=as_of, repo=repo)
    company_id = company["company_id"]
    trials = repo.get_trials_for_company(company_id)
    doc = _Doc()
    subscores = score.get("subscores") or {}
    doc.add(f"# {company.get('name')} ({company.get('ticker') or '-'})", "",
            f"_{DISCLAIMER}_", "",
            f"As of {as_of}; score run {run_meta.get('run_id')}, model {run_meta.get('model_version')}, "
            f"config {run_meta.get('config_hash')}.", "",
            f"- Rank by composite: {score.get('rank_by_composite')} (composite {_num(score.get('composite'), 3)}, "
            f"tier {score.get('tier')}, fundamental {_num(subscores.get('fundamental'), 3)}, "
            f"weight coverage {_num(subscores.get('coverage'))})",
            "- Rank by expected value: not available until the M7 return model",
            "- Block scores: " + ", ".join(f"{k} {_num(v)}" for k, v in (subscores.get("blocks") or {}).items()))
    _identity(doc, company, repo, ctx)
    _assets(doc, trials, as_of, ctx, company_id)
    _runway(doc, company_id, ctx)
    _fit(doc, score)
    _frictions(doc, score)
    _catalysts(doc, trials, as_of)
    _process(doc, company_id, score, ctx)
    if doc.omitted:
        doc.heading("Omitted (no source)")
        doc.add(*(f"- {o}" for o in doc.omitted))
    return "\n".join(doc.lines) + "\n"


def write_dossiers(repo: Any, rows: Sequence[Dict[str, Any]], as_of: date, run_meta: Dict[str, Any], out_dir: Path,
                   top_n: int = 25) -> List[Path]:
    """Write dossiers for the first `top_n` ranked rows (rows are already sorted by rank). Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    companies = {c["company_id"]: c for c in repo.list_companies_full()}
    written: List[Path] = []
    for score in [r for r in rows if r.get("rank_by_composite") is not None][:top_n]:
        company = companies.get(score["company_id"])
        if company is None:
            continue
        name = f"{score['rank_by_composite']:03d}_{company.get('ticker') or company['company_id']}.md"
        path = out_dir / name
        path.write_text(render_dossier(repo, company, score, as_of, run_meta), encoding="utf-8")
        written.append(path)
    return written
