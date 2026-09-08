"""Tests for ingest/modality_classifier.py (spec §3.5)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.modality_classifier import UNCLASSIFIED, classify_modality


def test_classify_modality_monoclonal_antibody():
    assert classify_modality("BIOLOGICAL", "Pembrolizumab") == "biologic_antibody"


def test_classify_modality_antibody_drug_conjugate_checked_before_mab():
    """An ADC's name still ends in the antibody component's '-mab'-adjacent root — must not
    fall through to biologic_antibody."""
    assert classify_modality("BIOLOGICAL", "Trastuzumab deruxtecan") == "antibody_drug_conjugate"
    assert classify_modality("BIOLOGICAL", "Sacituzumab govitecan") == "antibody_drug_conjugate"


def test_classify_modality_cell_therapy():
    assert classify_modality("BIOLOGICAL", "Axicabtagene ciloleucel") == "cell_therapy"
    assert classify_modality("BIOLOGICAL", "CAR-T cell therapy targeting CD19") == "cell_therapy"


def test_classify_modality_gene_therapy():
    assert classify_modality("BIOLOGICAL", "AAV9 gene therapy vector") == "gene_therapy"
    assert classify_modality("DRUG", "Onasemnogene abeparvovec-xioi") == "gene_therapy"  # -parvovec suffix


def test_classify_modality_rna_therapeutic():
    assert classify_modality("DRUG", "VX-522 mRNA therapy") == "rna_therapeutic"
    assert classify_modality("DRUG", "Inotersen antisense oligonucleotide") == "rna_therapeutic"


def test_classify_modality_radioligand():
    assert classify_modality("DRUG", "Lutetium Lu 177 vipivotide tetraxetan") == "radioligand"


def test_classify_modality_vaccine_takes_precedence_over_mrna_delivery_mechanism():
    """An mRNA vaccine's more useful categorical identity here is its use (vaccine), not its
    delivery mechanism (rna_therapeutic) — see module docstring's order-dependent-ambiguity note."""
    assert classify_modality("BIOLOGICAL", "mRNA-1273 vaccine") == "vaccine"


def test_classify_modality_vaccine_takes_precedence_over_protein_replacement():
    assert classify_modality("BIOLOGICAL", "Recombinant protein vaccine") == "vaccine"


def test_classify_modality_peptide():
    assert classify_modality("DRUG", "Semaglutide peptide analog") == "peptide"


def test_classify_modality_protein_replacement():
    assert classify_modality("BIOLOGICAL", "Recombinant protein replacement therapy") == "protein_replacement"


def test_classify_modality_unclassified_when_no_keyword_matches():
    assert classify_modality("DRUG", "Compound XYZ-123") == UNCLASSIFIED


def test_classify_modality_never_guesses_small_molecule_from_type_alone():
    """type == 'DRUG' alone, with no corroborating keyword, must NOT default to small_molecule —
    see module docstring's discipline note."""
    assert classify_modality("DRUG", "ABC-999") == UNCLASSIFIED


def test_classify_modality_case_insensitive():
    assert classify_modality("BIOLOGICAL", "PEMBROLIZUMAB") == "biologic_antibody"


def test_classify_modality_none_type_does_not_raise():
    assert classify_modality(None, "Compound X") == UNCLASSIFIED
