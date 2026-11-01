"""
P22 — company-name <-> ClinicalTrials.gov sponsor-name matching (spec §2.2, §3.3, Block B input).

Two jobs, one tokenization:

1. **`ctgov_query_name`** — the string sent as CT.gov's `query.spons`. CT.gov requires EVERY query
   word to appear in the sponsor name, so the raw SEC registrant name fails on its legal-form tail:
   live-verified 2026-11-01, `"maia biotechnology, inc."` returns 0 studies, `"maia biotechnology"`
   returns 3 (the sponsor is filed as plain "Maia Biotechnology"). In the 2026-10-31 scoring run 225
   of 497 targets got no studies at all for this reason; stripping legal forms recovered 139.
2. **`sponsor_matches_company`** — whether a study's lead sponsor is this company. `query.spons`
   also matches collaborators, so a raw hit list attributes CRO clients' trials to ICON/PPD, a
   partner's trials to the licensor, etc. Attribution requires every query token of the company to
   appear in the lead-sponsor name ("Spero Therapeutics" matches "Spero Therapeutics, Inc."; a trial
   led by "Hartford Hospital" does not).

Unlike `entity_resolution.normalize_company_name` (spec §3.3's alias-dedup key), descriptive words
such as "therapeutics"/"pharmaceuticals" are KEPT: dropping them turns "Compass Therapeutics" into
the query "compass", which also matches "Compass Pathways".

Token subsets cannot tell apart two unrelated companies that share a name, so known homonyms are
excluded explicitly (`_HOMONYM_SPONSOR_TOKENS`): Merck & Co.'s query "merck" also returns Merck KGaA's
trials (2026-11-01 run: 330 of 5,242 studies, led by "Merck KGaA, Darmstadt, Germany" and "Merck
Healthcare KGaA, ..."), which Merck & Co.'s own sponsor strings never contain.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, List

# Legal-form words only — never descriptive ones (see module docstring).
_LEGAL_FORM_TOKENS = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "llc", "lp",
    "holdings", "holding", "sa", "ag", "nv", "se", "bv", "ab", "asa", "spa", "the",
})
# EDGAR conformed names carry state/filer tags: "SEATTLE GENETICS INC /WA", "ICON PLC /ADR/",
# "VERTEX PHARMACEUTICALS INC / MA".
_EDGAR_TAG_RE = re.compile(r"/\s*[a-z]{2,5}\b/?")
# Dotted legal forms collapse to one token before punctuation is stripped ("n.v." -> "nv").
_DOTTED_FORMS = (("n.v.", "nv"), ("s.a.", "sa"), ("s.p.a.", "spa"), ("b.v.", "bv"))
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
# Company query tokens -> sponsor tokens that mark an unrelated same-name company (see module docstring).
_HOMONYM_SPONSOR_TOKENS: Dict[FrozenSet[str], FrozenSet[str]] = {
    frozenset({"merck"}): frozenset({"kgaa", "serono"}),
}


def _tokens(name: str) -> List[str]:
    """Lowercase alphanumeric tokens, EDGAR tags removed, every other character a separator."""
    lowered = name.lower()
    for dotted, plain in _DOTTED_FORMS:
        lowered = lowered.replace(dotted, plain)
    lowered = _EDGAR_TAG_RE.sub(" ", lowered)
    return _NON_WORD_RE.sub(" ", lowered).split()


def company_query_tokens(company_name: str) -> List[str]:
    """The distinctive tokens of a company name: legal forms and single characters removed, order kept."""
    return [t for t in _tokens(company_name) if t not in _LEGAL_FORM_TOKENS and len(t) > 1]


def ctgov_query_name(company_name: str) -> str:
    """
    CT.gov `query.spons` string for a company (see module docstring). Falls back to the lowercased
    name if stripping leaves nothing, so a name made only of legal-form words is still searched.
    """
    tokens = company_query_tokens(company_name)
    return " ".join(tokens) if tokens else company_name.lower()


def sponsor_matches_company(sponsor_name: str, company_name: str) -> bool:
    """True if every distinctive token of `company_name` appears in `sponsor_name` (see module docstring)."""
    wanted = set(company_query_tokens(company_name))
    if not wanted or not sponsor_name:
        return False
    sponsor_tokens = set(_tokens(sponsor_name))
    if sponsor_tokens & _HOMONYM_SPONSOR_TOKENS.get(frozenset(wanted), frozenset()):
        return False
    return wanted <= sponsor_tokens
