"""
P22 — best-effort therapeutic-area classifier from free-text conditions (spec §3.5, M3).

Maps a CT.gov study's `conditions` (free-text strings like "Non-Small Cell
Lung Cancer" or "Cystic Fibrosis") to a `p22_therapeutic_area.yaml` vocab
value, via keyword matching. **Explicitly a best-effort heuristic, not a
clinical taxonomy mapping** — user-approved 2026-08-31 as an acceptable
tradeoff for `ingest/asset_normalization.py`'s single-intervention-trial
asset creation, which needs SOME `therapeutic_area` value because
`p22_asset.therapeutic_area` is `NOT NULL` in the schema.

This solves a different problem than the mapping work already done in
`p22_base_rates.yaml`'s header/comments (spec §4.2's "record the mapping
decision" request, resolved 2026-08-31 by finding the actual BIO study) —
that work maps a VOCAB KEY to a base RATE; this module maps a trial's own
free-text CONDITIONS to a VOCAB KEY in the first place. Neither one implies
the other is done.

**Known limitations, disclosed rather than hidden:**
- Order-dependent keyword priority (e.g. hematologic-malignancy keywords are
  checked before generic "cancer", so "Acute Myeloid Leukemia" correctly
  lands in `oncology_heme` and not `oncology_solid`) — but any keyword list
  will miscategorize condition names it wasn't written with in mind.
- No confidence score — a match is a match, there is no "probably right"
  vs. "definitely right" distinction. `UNCLASSIFIED` is returned only when
  NO keyword matches, not when a match is uncertain.
- Ambiguous conditions get one hardcoded answer (e.g. "Multiple Sclerosis"
  is genuinely both a neurological and an autoimmune-adjacent disease; this
  module classifies it `neurology`, matching this vocab's convention of
  reserving `autoimmune` for classic rheumatologic/GI autoimmune disease).
- `rare_metabolic`, `cardiometabolic`, `gene_cell_therapy`, and
  `rare_orphan_disease` are never returned by this classifier — none of them
  is identifiable from condition text alone (rarity and platform/modality
  aren't properties of a condition's name), consistent with
  `p22_base_rates.yaml`'s finding that the BIO study itself can't populate
  these either.

Any asset this module's output feeds into should be treated as a candidate
classification, not a verified one, until `docs/Tasks.md`'s broader
"Decisions needed" items around therapeutic-area review are resolved.
"""

from __future__ import annotations

from typing import List, Tuple

UNCLASSIFIED = "unclassified"

# (vocab key, keywords) — checked IN ORDER, first match wins. More specific
# categories are listed before more general ones that would otherwise
# shadow them (e.g. hematologic-malignancy terms before generic "cancer").
_KEYWORD_RULES: List[Tuple[str, List[str]]] = [
    ("oncology_heme", [
        "leukemia", "leukaemia", "lymphoma", "myeloma", "hodgkin", "myelodysplastic",
    ]),
    ("hematology_nonmalig", [
        "hemophilia", "haemophilia", "sickle cell", "thalassemia", "von willebrand",
        "thrombocytopenia", "anemia", "anaemia",
    ]),
    ("oncology_solid", [
        "cancer", "carcinoma", "tumor", "tumour", "sarcoma", "melanoma", "neoplasm",
        "oncology", "adenocarcinoma", "glioblastoma", "glioma",
    ]),
    ("autoimmune", [
        "lupus", "rheumatoid arthritis", "psoriasis", "psoriatic", "crohn",
        "ulcerative colitis", "sjogren", "vasculitis", "ankylosing spondylitis", "autoimmune",
    ]),
    ("neurology", [
        "parkinson", "alzheimer", "epilepsy", "migraine", "amyotrophic lateral sclerosis",
        "multiple sclerosis", "stroke", "dementia", "neuropathy", "huntington",
    ]),
    ("psychiatry", [
        "depression", "depressive", "schizophrenia", "anxiety", "bipolar",
        "post-traumatic stress", "ptsd",
    ]),
    ("cardiovascular", [
        "heart failure", "atrial fibrillation", "hypertension", "coronary artery",
        "myocardial infarction", "atherosclerosis", "cardiovascular", "cardiomyopathy",
    ]),
    ("metabolic", [
        "diabetes", "obesity", "dyslipidemia", "metabolic syndrome", "nash", "nafld",
        "hyperlipidemia",
    ]),
    ("infectious_disease", [
        "infection", "hiv", "hepatitis", "covid", "influenza", "tuberculosis", "sepsis",
        "bacterial", "viral",
    ]),
    ("respiratory", [
        "asthma", "copd", "pulmonary fibrosis", "cystic fibrosis", "pneumonia", "respiratory",
    ]),
    ("ophthalmology", [
        "macular degeneration", "glaucoma", "retinopathy", "uveitis", "ocular",
    ]),
    ("dermatology", [
        "eczema", "atopic dermatitis", "acne", "vitiligo", "hidradenitis",
    ]),
    ("gastroenterology", [
        "irritable bowel", "gastroesophageal", "gastroparesis", "cirrhosis",
    ]),
    ("nephrology", [
        "chronic kidney disease", "renal failure", "nephropathy", "polycystic kidney",
    ]),
    ("musculoskeletal", [
        "osteoarthritis", "osteoporosis", "muscular dystrophy",
    ]),
]


def classify_therapeutic_area(conditions: List[str]) -> str:
    """
    Classify a trial's `conditions` list into a `p22_therapeutic_area.yaml`
    vocab value via keyword matching, or `UNCLASSIFIED` if nothing matches.
    Never raises, never returns `None` — always a valid vocab value, so
    callers writing `p22_asset.therapeutic_area` (`NOT NULL`) always have
    something safe to write.
    """
    text = " ".join(conditions).lower()
    for area, keywords in _KEYWORD_RULES:
        if any(keyword in text for keyword in keywords):
            return area
    return UNCLASSIFIED
