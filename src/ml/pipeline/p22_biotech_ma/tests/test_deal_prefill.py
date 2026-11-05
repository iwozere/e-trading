"""Tests for ingest/deal_prefill.py (spec §2.5 step 2). No network or DB — filing text is synthetic,
modelled on real SC 14D9 / S-4 wording (Allos/Spectrum 2012, Salix/Valeant 2015)."""

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.deal_prefill import (
    DealTerms,
    cluster_filings,
    extract_agreement_date,
    extract_parties,
    extract_price,
    extract_terms,
    filing_name,
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


def test_extract_agreement_date_breaks_ties_with_earliest_date():
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
    terms = DealTerms(target_holders_pct=0.7, acquirer_name="X Inc")
    deal_type, note = suggest_deal_type("S-4", terms)
    assert deal_type is None and note is not None and "acquirer" in note


def test_suggest_deal_type_shell_and_pe_buyers():
    shell = DealTerms(acquirer_name="Concentra Biosciences, LLC", upfront_per_share=2.0)
    pe = DealTerms(acquirer_name="Blackstone Life Sciences LLC", upfront_per_share=2.0)
    assert suggest_deal_type("SC 14D9", shell)[0] == "shell_transaction"
    assert suggest_deal_type("SC 14D9", pe)[0] == "pe_take_private"


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


def test_extract_agreement_date_prefers_most_repeated_date():
    text = (
        "Agreement and Plan of Merger, dated as of February 5, 2015. Centerview delivered its opinion on the "
        "merger agreement and in connection with the delivery of its opinion, dated February 4, 2015. "
        "The Agreement and Plan of Merger, dated as of February 5, 2015, provides for the merger."
    )
    assert extract_agreement_date(text, filed=date(2015, 4, 10)) == date(2015, 2, 5)


def test_extract_parties_reads_which_we_refer_to_as_definitions():
    text = (
        "Covance Inc., a Delaware corporation, which we refer to as Covance, and Laboratory Corporation of "
        "America Holdings, which we refer to as LabCorp, have entered into an Agreement and Plan of Merger."
    )
    parties = extract_parties(text)
    assert parties["Covance"] == "Covance Inc."
    assert parties["LabCorp"] == "Laboratory Corporation of America Holdings"


def test_extract_parties_trims_leading_prose_but_keeps_lowercase_suffixes():
    text = (
        'On behalf of the boards of directors of Bristol-Myers Squibb Company ("Bristol-Myers Squibb") we write. '
        'As previously announced, on November 16, 2014, Actavis plc ("Actavis") agreed to acquire Allergan.'
    )
    parties = extract_parties(text)
    assert parties["Bristol-Myers Squibb"] == "Bristol-Myers Squibb Company"
    assert parties["Actavis"] == "Actavis plc"


def test_pick_acquirer_falls_back_to_most_mentioned_party():
    parties = {"Forest": "Forest Laboratories, Inc.", "Actavis": "Actavis plc", "Greenhill": "Greenhill & Co., LLC"}
    mentions = {"Forest": 900, "Actavis": 700, "Greenhill": 150}
    assert pick_acquirer(parties, "FOREST LABORATORIES INC", mentions=mentions) == "Actavis plc"
    assert pick_acquirer(parties, "FOREST LABORATORIES INC", mentions={"Actavis": 5}) is None


def test_extract_price_reads_cash_and_stock_consideration():
    text = "Covance holders are entitled to receive 0.2686 shares of LabCorp common stock and $75.76 in cash, without interest."
    assert extract_price(text) == 75.76


def test_suggest_deal_type_rejects_non_acquisitions():
    assert suggest_deal_type("S-4", DealTerms(is_debt_exchange=True))[1] is not None
    assert suggest_deal_type("SC 14D9", DealTerms(is_mini_tender=True))[1] is not None
    note = suggest_deal_type("S-4", DealTerms(acquirer_name="Abraxis BioScience, Inc.", upfront_per_share=43.0))[1]
    assert note is not None and "acquirer" in note


def test_suggest_deal_type_minority_stake_with_cash_is_strategic():
    terms = DealTerms(target_holders_pct=0.31, acquirer_name="Bristol-Myers Squibb Company", upfront_per_share=50.0)
    assert suggest_deal_type("DEFM14A", terms) == ("strategic_acquisition", None)


def test_suggest_deal_type_all_stock_proxy_needs_reverse_merger_wording():
    terms = DealTerms(target_holders_pct=0.45, acquirer_name="Alumis Inc.")
    assert suggest_deal_type("DEFM14A", terms) == (None, None)
    terms.mentions_reverse_merger = True
    assert suggest_deal_type("DEFM14A", terms) == ("reverse_merger", None)


def test_extract_terms_flags_debt_exchange_offer():
    text = "Exchange offer for outstanding notes. " + "The old notes and the exchange notes. " * 60
    assert extract_terms(text, filed=date(2020, 5, 20), subject_name="Biogen Inc").is_debt_exchange is True
    assert extract_terms(_TENDER, filed=date(2012, 4, 16), subject_name="ALLOS THERAPEUTICS INC").is_debt_exchange is False


def test_prefill_cluster_competing_acquirers_left_for_human():
    hostile = _TENDER.replace("Spectrum Pharmaceuticals, Inc.", "Valeant Pharmaceuticals International, Inc.")
    texts = {"u1": hostile, "u2": _TENDER}
    rows = prefill_cluster(_cluster(), lambda r: texts[r["evidence_url"]])
    assert rows[0]["prefill_confidence"] == "low"
    assert "acquirer differs" in rows[0]["prefill_evidence"]
    assert "action" not in rows[0]


def test_extract_parties_handles_spaced_quotes_and_inner_parenthetical():
    text = (
        'the Agreement and Plan of Merger among Novartis AG, a company limited by shares ( Aktiengesellschaft ) '
        'incorporated under the laws of Switzerland (" Novartis "), and BioMarin Pharmaceutical Inc., a Delaware '
        'corporation (which we refer to as "BioMarin"), and LDH XV Corporation ("Purchaser") and Eli Lilly and Company ("Parent")'
    )
    parties = extract_parties(text)
    assert parties["Novartis"] == "Novartis AG"
    assert parties["BioMarin"] == "BioMarin Pharmaceutical Inc."
    assert parties["Parent"] == "Eli Lilly and Company"


def test_extract_price_prefers_ads_amount():
    text = '$13.375 in cash per ordinary share. "ADS Consideration" means an amount per ADS equal to $107 in cash.'
    assert extract_price(text) == 107.0


def test_pick_acquirer_ignores_roster_company_named_in_passing():
    parties = {"EIP": "EIP Pharma, Inc.", "Vertex": "Vertex Pharmaceuticals Incorporated", "Parent": "GSK plc"}
    assert pick_acquirer(parties, "Diffusion Pharmaceuticals", ["Vertex Pharmaceuticals Inc"], mentions={"Vertex": 3}) == "GSK plc"
    ultimate = {"Parent": "GlaxoSmithKline LLC", "Ultimate Parent": "GSK plc"}
    assert pick_acquirer(ultimate, "RAPT Therapeutics", ["GSK plc"], mentions={"Ultimate Parent": 2}) == "GSK plc"


def test_filing_name_uses_name_at_filing_time():
    row = {"company_name": "LENZ THERAPEUTICS, INC.", "names_in_filing": "Graphite Bio, Inc.  (LENZ)  (CIK 0001815776)"}
    assert filing_name(row) == "Graphite Bio, Inc."
    assert filing_name({"company_name": "Pfizer Inc", "names_in_filing": ""}) == "Pfizer Inc"


def test_extract_terms_holders_stake_skips_past_tense_voting_agreements():
    text = (
        "certain Graphite stockholders who in the aggregate owned approximately 52% signed support agreements. "
        "Following the merger, pre-merger Graphite stockholders will own approximately 35% of the combined company."
    )
    terms = extract_terms(text, filed=date(2023, 12, 6), subject_name="Graphite Bio, Inc.")
    assert terms.target_holders_pct is not None and abs(terms.target_holders_pct - 0.35) < 1e-9


def test_suggest_deal_type_despac_asset_sale_and_pe_without_cash():
    assert "SPAC" in (suggest_deal_type("S-4", DealTerms(is_despac=True))[1] or "")
    assert suggest_deal_type("DEFM14A", DealTerms(is_asset_sale=True, acquirer_name="Gilead Sciences, Inc.")) == ("asset_sale", None)
    assert suggest_deal_type("DEFM14A", DealTerms(acquirer_name="KKR JP LLC")) == (None, None)


def test_extract_terms_detects_despac_and_asset_sale():
    spac = "Data Knights is a blank check company. " * 5
    sale = "The Asset Purchase Agreement with Gilead. " * 5
    assert extract_terms(spac, filed=None, subject_name="Data Knights Acquisition Corp.").is_despac is True
    assert extract_terms(sale, filed=None, subject_name="HOOKIPA Pharma Inc.").is_asset_sale is True


def test_prefill_cluster_reject_clears_entry_fields():
    cluster = [dict(r, form_type="S-4") for r in _cluster()]
    rows = prefill_cluster(cluster, lambda r: _TENDER)
    assert rows[0]["action"] == "reject" and "acquirer" in rows[0]["note"]
    assert rows[0]["acquirer_name"] == "" and rows[0]["upfront_per_share"] == ""
