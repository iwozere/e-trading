"""Tests for ingest/therapeutic_area_classifier.py."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.therapeutic_area_classifier import (
    UNCLASSIFIED,
    classify_therapeutic_area,
)


def test_classifies_solid_tumor_cancer():
    assert classify_therapeutic_area(["Non-Small Cell Lung Cancer"]) == "oncology_solid"


def test_classifies_hematologic_malignancy_not_solid():
    """Leukemia/lymphoma must NOT fall through to the generic 'cancer' oncology_solid rule."""
    assert classify_therapeutic_area(["Acute Myeloid Leukemia"]) == "oncology_heme"
    assert classify_therapeutic_area(["Diffuse Large B-Cell Lymphoma"]) == "oncology_heme"


def test_classifies_non_malignant_blood_disorder():
    assert classify_therapeutic_area(["Sickle Cell Disease"]) == "hematology_nonmalig"
    assert classify_therapeutic_area(["Hemophilia A"]) == "hematology_nonmalig"


def test_classifies_autoimmune():
    assert classify_therapeutic_area(["Rheumatoid Arthritis"]) == "autoimmune"
    assert classify_therapeutic_area(["Crohn's Disease"]) == "autoimmune"


def test_classifies_neurology():
    assert classify_therapeutic_area(["Parkinson's Disease"]) == "neurology"
    assert classify_therapeutic_area(["Multiple Sclerosis"]) == "neurology"  # documented convention choice


def test_classifies_cardiovascular():
    assert classify_therapeutic_area(["Heart Failure with Reduced Ejection Fraction"]) == "cardiovascular"


def test_classifies_metabolic():
    assert classify_therapeutic_area(["Type 2 Diabetes Mellitus"]) == "metabolic"


def test_classifies_infectious_disease():
    assert classify_therapeutic_area(["Chronic Hepatitis B Infection"]) == "infectious_disease"


def test_classifies_respiratory_cystic_fibrosis():
    assert classify_therapeutic_area(["Cystic Fibrosis"]) == "respiratory"


def test_no_keyword_match_returns_unclassified():
    assert classify_therapeutic_area(["Some Extremely Rare Novel Syndrome XYZ"]) == UNCLASSIFIED


def test_empty_conditions_returns_unclassified():
    assert classify_therapeutic_area([]) == UNCLASSIFIED


def test_case_insensitive_matching():
    assert classify_therapeutic_area(["LUNG CANCER"]) == "oncology_solid"


def test_multiple_conditions_first_keyword_match_wins():
    """Order-dependent: heme-malignancy keywords are checked before generic cancer keywords,
    regardless of which condition string in the list they appear in."""
    assert classify_therapeutic_area(["Solid Tumor", "Acute Lymphoblastic Leukemia"]) == "oncology_heme"


def test_never_returns_areas_with_no_keyword_rules():
    """rare_metabolic/cardiometabolic/gene_cell_therapy/rare_orphan_disease are never returned —
    see module docstring."""
    never_returned = {"rare_metabolic", "cardiometabolic", "gene_cell_therapy", "rare_orphan_disease"}
    sample_conditions = [
        ["Lung Cancer"], ["Leukemia"], ["Sickle Cell"], ["Lupus"], ["Parkinson's"],
        ["Depression"], ["Heart Failure"], ["Diabetes"], ["HIV"], ["Asthma"],
        ["Glaucoma"], ["Eczema"], ["IBS"], ["Chronic Kidney Disease"], ["Osteoarthritis"],
        ["Nonsense Condition"],
    ]
    for conditions in sample_conditions:
        assert classify_therapeutic_area(conditions) not in never_returned
