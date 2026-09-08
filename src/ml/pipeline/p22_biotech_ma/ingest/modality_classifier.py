"""
P22 — best-effort modality classifier from CT.gov intervention name/type (spec §3.5, M3), 2026-09-08.

Same role, same discipline, and the same caller as `therapeutic_area_classifier.py`
(`ingest/asset_normalization.resolve_or_create_asset`) — that module's docstring names this exact
gap: "`modality`... always `None`: ... modality classification is a separate, not-yet-built
decision." **Explicitly a best-effort heuristic, not a clinical/regulatory taxonomy mapping** — the
vocabulary itself (`config/pipeline/p22_modality.yaml`) is still an unreviewed DRAFT
(`docs/Tasks.md` item 4), so any output here is a candidate classification pending domain review,
not ground truth, same status as `classify_therapeutic_area`'s output.

Classifies from the intervention's free-text `name` (drug-naming conventions carry real signal —
INN suffixes like `-mab` for monoclonal antibodies are assigned by WHO specifically to be
identifiable, though matched here WITHOUT the literal hyphen — see the keyword table's own note on
why) with CT.gov's own `interventionType` (`DRUG`/`BIOLOGICAL`) as a weak prior, never a guessed
default — see the module docstring's discipline note on why `type == "DRUG"` alone is NOT treated
as `small_molecule` without a corroborating keyword (that would be exactly the kind of "guess the
common case" fabrication `p22_base_rates.yaml`'s orphan-modifier warning forbids applying
elsewhere in this codebase).

**Known limitations, disclosed rather than hidden** (same posture as `therapeutic_area_classifier.py`):
- Order-dependent keyword priority — antibody-drug-conjugate keywords are checked before
  monoclonal-antibody ones, so a name like "trastuzumab deruxtecan" (an ADC) doesn't fall through
  to `biologic_antibody` just because its base antibody name ends in the `mab` root.
- Substring matching, no word boundaries — same class of false-positive risk `mab`/`aso`-style
  short keywords always carry (e.g. a coincidental substring match), accepted here as it was there.
- `device_combination` and `protein_replacement` are reachable in principle but rare in practice —
  `resolve_or_create_asset` only ever classifies DRUG/BIOLOGICAL-type interventions (spec's own
  `_ASSET_INTERVENTION_TYPES` scope), and CT.gov intervention names rarely spell out device or
  replacement-protein framing explicitly.
- `UNCLASSIFIED` (not a guess) whenever no keyword matches — a plain small-molecule drug name with
  no distinguishing suffix (e.g. "Compound XYZ-123") is indistinguishable, from name text alone,
  from any other modality this classifier doesn't have a positive signal for.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

UNCLASSIFIED = "unclassified"

# (vocab key, keywords) — checked IN ORDER, first match wins. All keywords are plain substrings,
# NOT literal hyphenated suffixes — WHO's INN naming convention attaches a stem directly onto the
# generic name (e.g. "pembrolizumab" ends in "...umab", not "...u-mab"), so a keyword like "-mab"
# would never match a real drug name; every suffix keyword below is written without the hyphen and
# live-checked against a real approved drug name in that category (see test_modality_classifier.py).
#
# Antibody-drug conjugates and bispecifics are listed before the generic monoclonal-antibody
# keywords they would otherwise false-match on — an ADC's name still contains its antibody
# component's "mab" root (e.g. "trastuzumab deruxtecan").
_KEYWORD_RULES: List[Tuple[str, List[str]]] = [
    ("antibody_drug_conjugate", [
        "antibody-drug conjugate", "antibody drug conjugate", " adc ", "vedotin", "deruxtecan",
        "emtansine", "govitecan",
    ]),
    ("cell_therapy", [
        "car-t", "car t cell", "chimeric antigen receptor", "tcr-t", "til therapy",
        "tumor infiltrating lymphocyte", "autologous cell", "allogeneic cell", "cell therapy",
        "leucel",  # INN suffix for cellular therapies, e.g. "axicabtagene ciloleucel", "brexucabtagene autoleucel"
    ]),
    ("gene_therapy", [
        "gene therapy", "adeno-associated virus", "aav", "lentiviral", "lentivirus", "gene editing",
        "crispr", "parvovec",  # INN suffix for AAV-vector gene therapies, e.g. "voretigene neparvovec"
    ]),
    ("radioligand", [
        "radioligand", "radionuclide", "lutetium", "actinium", "theranostic", "177lu", "225ac",
    ]),
    # Checked BEFORE rna_therapeutic: an mRNA vaccine (e.g. "mRNA-1273") contains "mrna", but its
    # more useful categorical identity here is its USE (vaccine) rather than its delivery
    # mechanism — same order-dependent-ambiguity disclosure as therapeutic_area_classifier.py's
    # "Multiple Sclerosis" example. rna_therapeutic's OTHER keywords (siRNA, ASO) are never
    # vaccines, so this reordering only actually changes anything for "mrna"/"rnai" + "vaccine"
    # appearing together.
    ("vaccine", [
        "vaccine",
    ]),
    ("rna_therapeutic", [
        "sirna", "antisense oligonucleotide", "aso", "mrna", "rnai",
        "rsen",  # INN suffix for antisense oligonucleotides, e.g. "inotersen", "eteplirsen", "nusinersen"
    ]),
    ("biologic_antibody", [
        "monoclonal antibody", "bispecific", "nanobody", "immunoglobulin",
        "mab",  # INN suffix, e.g. "pembrolizumab" — checked after the more specific rules above
    ]),
    ("protein_replacement", [
        "enzyme replacement", "recombinant protein", "protein replacement therapy", "clotting factor",
    ]),
    ("peptide", [
        "peptide",
    ]),
    ("device_combination", [
        "device combination", "drug-device combination", "combination product",
    ]),
]


def classify_modality(intervention_type: Optional[str], intervention_name: str) -> str:
    """
    Classify one intervention's modality via keyword matching over its `name`, or `UNCLASSIFIED`
    if nothing matches. `intervention_type` (CT.gov's `DRUG`/`BIOLOGICAL`) is accepted for
    signature symmetry with the CT.gov data model and future use, but NOT currently used to guess
    a default when no keyword matches — see module docstring. Never raises, never returns `None`,
    same contract as `classify_therapeutic_area`.
    """
    del intervention_type
    text = f" {intervention_name.lower()} "
    for modality, keywords in _KEYWORD_RULES:
        if any(keyword in text for keyword in keywords):
            return modality
    return UNCLASSIFIED
