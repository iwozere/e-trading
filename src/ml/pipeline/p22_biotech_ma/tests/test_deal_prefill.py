"""Tests for ingest/deal_prefill.py (spec §2.5 step 2). No network or DB — filing text is synthetic,
modelled on real SC 14D9 / S-4 wording (Allos/Spectrum 2012, Salix/Valeant 2015)."""

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.deal_prefill import (
    cluster_filings,
    extract_agreement_date,
    extract_parties,
    extract_price,
    extract_terms,
    filing_text,
    name_key,
    pick_acquirer,
    prefill_cluster,
    suggest_deal_type,
)

_TENDER = (
    'This Schedule 14D-9 relates to the tender offer by Sapphire Acquisition Sub, Inc. ("Merger Sub"), a wholly '
    'owned subsidiary of Spectrum Pharmaceuticals, Inc. ("Parent"), to purchase all outstanding shares of Allos '
    'Therapeutics, Inc. ("Company") for (i) an amount net to the seller in cash equal to $1.82, without interest '
    '(the "Cash Portion") and (ii) one contingent value right to receive additional consideration of $0.11 per '
    'Share in cash (the "CVR"), pursuant to the Agreement and Plan of Merger, dated as of April 4, 2012, by and '
    'among Parent, Merger Sub and the Company. The Securities and Exchange Commission ("SEC") maintains a site. '
    'In September Parent proposed $2.20 per share, in cash and stock.'
)
_REVERSE_MERGER = (
    'Agreement and Plan of Merger, dated as of June 22, 2023, by and among Aeglea BioTherapeutics, Inc. ("Parent"), '
    'Aspen Merger Sub I, Inc. ("Merger Sub") and Spyre Therapeutics, Inc. ("Spyre"). Following the merger, '
    'Aeglea stockholders are expected to own approximately 16.4% of the combined company. The exchange ratio '
    'will be fixed. Centerview Partners LLC ("Centerview") acted as financial advisor.'
)


def test_filing_text_strips_tags_entities_and_curly_quotes():
    text = filing_text("<html><style>x{}</style><p>Pfizer&nbsp;Inc. (“Parent”)</p><b>ok</b></html>")
    assert text == 'Pfizer Inc. ("Parent") ok'


def test_name_key_ignores_case_punctuation_and_suffixes():
    assert name_key("SALIX PHARMACEUTICALS LTD") == name_key("Salix Pharmaceuticals, Ltd.") == "salix"
    assert name_key("The Medicines Company") == "medicines"


def test_extract_agreement_date_takes_earliest_plausible_date():
    text = (
        "Agreement and Plan of Merger, dated as of March 2, 2024. Amendment No. 1 to the Merger Agreement, "
        "dated as of April 10, 2024. A prior Merger Agreement dated as of May 1, 2015 was terminated."
    )
    assert extract_agreement_date(text, filed=date(2024, 4, 20)) == date(2024, 3, 2)


def test_extract_agreement_date_ignores_dates_after_filing():
    assert extract_agreement_date("Merger Agreement dated as of May 1, 2024", filed=date(2024, 4, 1)) is None


def test_extract_parties_keeps_only_legal_entities():
    parties = extract_parties(_TENDER)
    assert parties["Parent"] == "Spectrum Pharmaceuticals, Inc."
    assert parties["Merger Sub"] == "Sapphire Acquisition Sub, Inc."
    assert "SEC" not in parties and "Cash Portion" not in parties


def test_pick_acquirer_prefers_roster_name_over_parent_label():
    parties = {"Purchaser": "Bolt Merger Sub I, Inc.", "Parent": "Merck Sharp & Dohme LLC", "Merck": "Merck & Co., Inc."}
    assert pick_acquirer(parties, "Prometheus Biosciences", ["Merck & Co Inc"]) == "Merck Sharp & Dohme LLC"
    parties_ultimate_first = {"Merck": "Merck & Co., Inc.", "Parent": "MSD Holdings LLC"}
    assert pick_acquirer(parties_ultimate_first, "Prometheus Biosciences", ["Merck & Co Inc"]) == "Merck & Co., Inc."


def test_pick_acquirer_without_parent_label_needs_allow_any_party():
    parties = {"Parent": "Aeglea BioTherapeutics, Inc.", "Merger Sub": "Aspen Merger Sub I, Inc.",
               "Centerview": "Centerview Partners LLC", "Spyre": "Spyre Therapeutics, Inc."}
    assert pick_acquirer(parties, "Aeglea BioTherapeutics", allow_any_party=False) is None
    assert pick_acquirer(parties, "Aeglea BioTherapeutics", allow_any_party=True) == "Spyre Therapeutics, Inc."


def test_extract_price_prefers_net_to_seller_over_cvr_and_proposals():
    assert extract_price(_TENDER) == 1.82


def test_extract_price_generic_tier_skips_excluded_cvr_amounts():
    text = "holders receive $5.00 per share in cash and a CVR worth $1.00 per share in cash, $5.00 per share in cash"
    assert extract_price(text, exclude={1.0}) == 5.0


def test_extract_terms_cash_tender_with_cvr():
    terms = extract_terms(_TENDER, filed=date(2012, 4, 16), subject_name="ALLOS THERAPEUTICS INC")
    assert terms.agreement_date == date(2012, 4, 4)
    assert terms.acquirer_name == "Spectrum Pharmaceuticals, Inc."
    assert terms.upfront_per_share == 1.82
    assert terms.has_cvr is True
    assert terms.stock_consideration_pct == 0.0
    assert suggest_deal_type("SC 14D9", terms) == ("strategic_acquisition", None)


def test_extract_terms_reverse_merger_from_holders_stake():
    terms = extract_terms(_REVERSE_MERGER, filed=date(2023, 8, 1), subject_name="Aeglea BioTherapeutics Inc")
    assert terms.target_holders_pct is not None and abs(terms.target_holders_pct - 0.164) < 1e-9
    assert terms.acquirer_name == "Spyre Therapeutics, Inc."
    assert terms.stock_consideration_pct == 1.0
    assert suggest_deal_type("S-4", terms) == ("reverse_merger", None)


def test_suggest_deal_type_rejects_when_subject_holders_keep_majority():
    terms = SimpleNamespace(target_holders_pct=0.7, acquirer_name="X Inc", upfront_per_share=None)
    deal_type, note = suggest_deal_type("S-4", terms)  # type: ignore[arg-type]
    assert deal_type is None and note is not None and "acquirer" in note


def test_suggest_deal_type_shell_and_pe_buyers():
    shell = SimpleNamespace(target_holders_pct=None, acquirer_name="Concentra Biosciences, LLC", upfront_per_share=2.0)
    pe = SimpleNamespace(target_holders_pct=None, acquirer_name="Blackstone Life Sciences LLC", upfront_per_share=2.0)
    assert suggest_deal_type("SC 14D9", shell)[0] == "shell_transaction"  # type: ignore[arg-type]
    assert suggest_deal_type("SC 14D9", pe)[0] == "pe_take_private"  # type: ignore[arg-type]


def test_cluster_filings_splits_on_long_gap_and_per_company():
    rows = [
        {"item_id": 1, "company_id": 7, "filed_date": "2012-04-16"},
        {"item_id": 2, "company_id": 7, "filed_date": "2012-05-01"},
        {"item_id": 3, "company_id": 7, "filed_date": "2018-01-10"},
        {"item_id": 4, "company_id": 8, "filed_date": "2012-04-20"},
    ]
    clusters = cluster_filings(rows)
    assert sorted([r["item_id"] for r in c] for c in clusters) == [[1, 2], [3], [4]]


def _cluster():
    return [
        {"item_id": 10, "company_id": 7, "company_name": "ALLOS THERAPEUTICS INC", "form_type": "SC 14D9",
         "filed_date": "2012-04-16", "evidence_url": "u1"},
        {"item_id": 11, "company_id": 7, "company_name": "ALLOS THERAPEUTICS INC", "form_type": "SC 14D9/A",
         "filed_date": "2012-05-10", "evidence_url": "u2"},
    ]


_COMPLETION = "On May 9, 2012, Parent completed the Merger."


def _context(premium_1d=0.5, flags=()):
    return SimpleNamespace(premium_1d=premium_1d, premium_30d=0.6, flags_fired=list(flags))


def test_prefill_cluster_drafts_deal_and_links_with_completion():
    texts = {"u1": _TENDER, "u2": _COMPLETION}
    rows = prefill_cluster(_cluster(), lambda r: texts[r["evidence_url"]], context_fn=lambda cid, e: _context())

    head, tail = rows
    assert head["action"] == "deal" and head["prefill_confidence"] == "high"
    assert head["announcement_date"] == "2012-04-04"
    assert head["acquirer_name"] == "Spectrum Pharmaceuticals, Inc."
    assert head["upfront_per_share"] == "1.82"
    assert head["has_cvr"] == "yes"
    assert head["status"] == "completed" and head["completion_date"] == "2012-05-09"
    assert head["stock_consideration_pct"] == "0%"
    assert tail["action"] == "link" and not tail.get("link_deal_id")  # import attaches it to the deal above


def test_prefill_cluster_fills_roster_cik():
    rows = prefill_cluster(
        _cluster(), lambda r: _TENDER, roster={"Spectrum Pharmaceuticals": "0000831547"}, context_fn=None
    )
    assert rows[0]["acquirer_cik"] == "0000831547"


def test_prefill_cluster_leaves_action_blank_when_flags_fire():
    rows = prefill_cluster(_cluster(), lambda r: _TENDER, context_fn=lambda cid, e: _context(flags=["target_market_cap_lt_50m"]))
    assert rows[0]["prefill_confidence"] == "flags"
    assert "action" not in rows[0] and "action" not in rows[1]
    assert rows[0]["prefill_flags"] == "target_market_cap_lt_50m"


def test_prefill_cluster_leaves_action_blank_on_implausible_premium():
    rows = prefill_cluster(_cluster(), lambda r: _TENDER, context_fn=lambda cid, e: _context(premium_1d=-0.6))
    assert rows[0]["prefill_confidence"] == "check_price"
    assert "action" not in rows[0]


def test_prefill_cluster_low_confidence_when_document_missing():
    rows = prefill_cluster(_cluster(), lambda r: None)
    assert rows[0]["prefill_confidence"] == "low"
    assert "action" not in rows[0]


def test_prefill_cluster_links_to_existing_deal():
    existing = [{"deal_id": 42, "announcement_date": date(2012, 4, 4)}]
    rows = prefill_cluster(_cluster(), lambda r: _TENDER, existing_deals=existing)
    assert [r["action"] for r in rows] == ["link", "link"]
    assert all(r["link_deal_id"] == "42" for r in rows)


def test_prefill_cluster_reverse_merger_majority_holders_rejects_all_rows():
    text = _REVERSE_MERGER.replace("16.4%", "72%")
    cluster = [dict(r, company_name="Aeglea BioTherapeutics Inc", form_type="S-4") for r in _cluster()]
    rows = prefill_cluster(cluster, lambda r: text)
    assert [r["action"] for r in rows] == ["reject", "reject"]
    assert "acquirer" in rows[0]["note"]
