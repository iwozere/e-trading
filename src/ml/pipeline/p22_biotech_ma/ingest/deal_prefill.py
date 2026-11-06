"""
P22 — draft deal labels for the `deal_entry_cli` spreadsheet round trip (spec §2.5 step 2), 2026-11-05.

Spec §2.5 asks for two steps: "parse announcement date, acquirer, consideration per share, CVR
presence, premium to prior close", then "hand-verify ... automated extraction alone will be too
noisy for labels." `deal_entry_cli export` only covers the second step — every entry column starts
blank, so ~700 deals would be typed by hand. This module does the parse step and writes its guesses
into the same columns, so the reviewer corrects a draft instead of transcribing filings. Nothing here
writes to the database; the reviewed CSV still goes through `deal_entry_cli import --dry-run`.

How a draft is built:

1. **Group filings into deals.** One deal produces several candidates (SC 14D9 + amendments, a
   DEFM14A, an S-4/A chain). A company's filings are split into clusters wherever two consecutive
   filings are more than `CLUSTER_GAP_DAYS` apart. The earliest filing of a cluster becomes the `deal`
   row; the rest become `link` rows (blank `link_deal_id` — import attaches them to the deal just
   entered, `deal_entry.find_link_target`).
2. **Read the first filing** (and the last one, for completion): the merger-agreement date, the
   parties defined as `X ("Parent")` etc., the cash price per share, CVR terms, and — for stock
   deals — what share of the combined company the subject's holders keep.
3. **Suggest a `deal_type`** from form and terms: a pre-merger holders' stake under 50% is a reverse
   merger; 50% or more means the subject company is the buyer, so the row is a `reject`; known
   private-equity and cash-shell buyers get `pe_take_private` / `shell_transaction`; anything else
   with a cash price is `strategic_acquisition`.
4. **Check against prices** (optional `context_fn`, i.e. `deal_entry.gather_context`): a 1-day premium
   outside `PREMIUM_SANE_RANGE` almost always means a wrong price or date, and a positive label with
   reverse-merger flags needs `--ack-flags` — both leave `action` blank for a human.

`action` is filled only when every required field was found and nothing above fired; the
`prefill_confidence` column says why a row was left blank (`low`, `check_price`, `flags`).
"""

from __future__ import annotations

import html
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.deal_entry import DealEntryError, entry_from_mapping
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

CLUSTER_GAP_DAYS = 365
AGREEMENT_LOOKBACK_DAYS = 730
MAX_TEXT_CHARS = 600_000  # summary term sheets and the merger-agreement header sit at the front
PREMIUM_SANE_RANGE = (-0.10, 3.0)
REVERSE_MERGER_MAX_HOLDERS_PCT = 0.5

# Buyers whose deals are not strategic acquisitions (spec §2.5 table). Matched as lowercase substrings.
PE_BUYERS = (
    "blackstone", "kkr", "kohlberg kravis", "eqt ", "bain capital", "carlyle", "tpg ", "patient square",
    "nordic capital", "apollo", "advent international", "hellman", "warburg pincus", "permira",
    "thoma bravo", "gtcr", "water street", "avista", "novo holdings", "searchlight", "hillhouse",
)
SHELL_BUYERS = ("concentra biosciences", "tang capital", "xoma royalty", "xoma corporation")

# Defined parties that are never the counterparty: exchanges, agents, banks and advisors.
_NOT_COUNTERPARTY = (
    "nasdaq", "stock exchange", "depositary", "computershare", "american stock transfer", "broadridge",
    "georgeson", "innisfree", "mackenzie", "okapi", "d.f. king", "trust company", "bank", "securities",
    "sachs", "morgan", "lazard", "centerview", "evercore", "jefferies", "leerink", "cowen", "guggenheim",
    "piper", "moelis", "perella", "citigroup", "barclays", "merrill", "credit suisse", "ubs", "wells fargo",
    "stifel", "raymond james", "wedbush", "oppenheimer", "wainwright", "ladenburg", "partners",
    "greenhill", "rothschild", "houlihan", "qatalyst", "pjt", "allen & company", "deutsche", "hsbc", "bofa",
    "insurance corporation", "board of", "appraisal", "advisory", "william blair", "sodali", "alliance advisors",
    # law firms
    " llp", "p.c.", "sonsini", "nautadutilh", "de brauw", "stibbe", "loyens", "houthoff", "freshfields",
    "linklaters", "clifford chance", "slaughter and may", "allen & overy", "skadden", "wachtell",
)
_PARENT_LABELS = ("Parent", "Acquirer", "Acquiror", "Buyer", "Offeror", "Purchaser Parent")
_SUB_LABELS = ("Purchaser", "Merger Sub", "Merger Subsidiary", "Sub", "Acquisition Sub", "Offeror Sub")
_SUBJECT_LABELS = ("Company",)
_NAME_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "llc", "lp",
    "sa", "ag", "nv", "se", "the",
}

_CURLY_QUOTES = str.maketrans("“”‘’", "\"\"''")
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = rf"(?:{_MONTHS})\s+\d{{1,2}},\s*\d{{4}}"
_NUM = r"(\d{1,4}(?:,\d{3})*(?:\.\d{1,4})?)"

_SIDE_AGREEMENT = (
    r"(?:confidentiality|letter|support|advisory|exclusivity|voting|warrant purchase|note purchase|stock purchase"
    r"|securities purchase|subscription|CVR|rights|employment|license|services|standstill|commitment|loan|credit)"
    r"\s+agreement|offer to purchase"
)
_AGREEMENT_DATE_PATTERNS = (
    re.compile(
        rf"(?:Agreement and Plan of Merger|Merger Agreement|Transaction Agreement|Arrangement Agreement|Asset Purchase Agreement)"
        # no side agreement in between: "Prior to signing the Merger Agreement, Decibel and Regeneron
        # entered into a customary confidentiality agreement, dated June 30, 2023"
        rf"(?:(?!{_SIDE_AGREEMENT})[^.;]){{0,120}}?dated\s+(?:as\s+of\s+)?({_DATE})",
        re.I,
    ),
    re.compile(
        rf"On\s+({_DATE}),[^.]{{0,300}}?entered\s+into\s+(?:an?|the)\s+"
        rf"(?:Agreement and Plan of Merger|Merger Agreement|Transaction Agreement|Arrangement Agreement)",
        re.I,
    ),
)
# `Name ("Label")`, also `(which we refer to as "Label")` and `(" Label ")` (spaces left by stripped tags).
_PARTY_DEF = re.compile(
    r'([A-Z][^()"]{1,160}?)\s*\(\s*(?:the\s+|collectively,\s+|(?:which\s+)?we\s+refer\s+to\s+as\s+(?:the\s+)?)?'
    r'"\s*([A-Z][A-Za-z&\-. ]{0,40}?)\s*"\s*\)'
)
_NAME_JOINERS = {"of", "&", "de", "du", "la", "und"}
# Proxy-statement style: `Laboratory Corporation of America Holdings, which we refer to as LabCorp,`.
_PARTY_REFER = re.compile(
    r'([A-Z][^()";]{1,160}?),\s+which we refer to as\s+"?([A-Z][A-Za-z&\-]{0,30}\.?(?: [A-Z][A-Za-z&\-]{1,30}){0,2})"?(?=[,.;)\s])'
)
_SENTENCE_BREAK = re.compile(r"(?<!\bSt)(?<!\bCo)(?<!\bNo)(?<!\bInc)\.\s+(?=[A-Z])")
_PARTY_CUTS = (
    " by and among ", " by and between ", " among ", " between ", " with ", " by ",
    " subsidiary of ", " subsidiaries of ", " affiliate of ", " shares of ", " stock of ", " securities of ",
    " holders of ",
)
MIN_ACQUIRER_MENTIONS = 20  # a counterparty named without a Parent label must dominate the text
# Most specific wording first: the first tier with any match decides the price. Generic "$X per share
# in cash" also hits earlier proposals in the background section and CVR amounts, so it is a last resort.
# Cash-only terms: "Merger Consideration" of a cash-and-stock deal is the total (Pharmacyclics: $261.25 of
# which $152.25 cash), and "Purchase Price" is often an aggregate amount.
_PRICE_TERM = (
    r"(?:Offer Price|Cash Consideration|Closing Cash Consideration|Per Share Cash Consideration"
    r"|Cash Merger Consideration|Cash Amount|Closing Amount)"
)
_PRICE_TIERS = (
    # ADS-listed companies (Verona, Centessa): prices in our data are per ADS, so the ADS amount wins.
    (
        re.compile(rf"\$\s?{_NUM}\s+(?:in cash\s+)?per\s+(?:ADS|American Depositary Share)\b", re.I),
        re.compile(rf"per\s+ADS\s+equal\s+to\s+\$\s?{_NUM}", re.I),
    ),
    (
        re.compile(rf"\$\s?{_NUM}[^.$]{{0,40}}?per\s+share,?\s+net to the (?:seller|holder|stockholder|shareholder)s? in cash", re.I),
        re.compile(rf"net to the (?:seller|holder|stockholder|shareholder)s? in cash(?:,)? equal to\s+\$\s?{_NUM}", re.I),
        # "$42.00 per Share, to be paid to the holder in cash" (Albireo/Ipsen)
        re.compile(rf"\$\s?{_NUM}\s+per\s+share,?\s+to be paid to the (?:holder|stockholder|shareholder)s? in cash", re.I),
        # the defined price term: "a per Share offer price of $23.50 (the "Offer Price")" (Assertio)
        re.compile(rf"per\s+(?:share|ADS)\b[^.$\"]{{0,40}}?\$\s?{_NUM}[^.$\"]{{0,80}}?\(the\s+\"\s*{_PRICE_TERM}\s*\"\)", re.I),
        re.compile(rf"\$\s?{_NUM}[^.$\"]{{0,40}}?per\s+(?:share|ADS)\b[^.$\"]{{0,80}}?\(the\s+\"\s*{_PRICE_TERM}\s*\"\)", re.I),
        # "a price per Share equal to $6.00 net to the holder in cash" (Dimension/Ultragenyx)
        re.compile(rf"\$\s?{_NUM}\s+net to the (?:seller|holder|stockholder|shareholder)s? in cash", re.I),
    ),
    (
        re.compile(rf"right to receive\s+(?:an amount in cash equal to\s+)?\$\s?{_NUM}\s*(?:per share\s+)?in cash", re.I),
        re.compile(rf"right to receive\s+\$\s?{_NUM}\s+in cash", re.I),
        # enumerated merger consideration: "right to receive (i) $2.15, payable to the holder thereof in
        # cash" (Paratek), "purchase all of the outstanding Shares in exchange for (i) $7.00 in cash per Share"
        re.compile(rf"(?:right to receive|in exchange for)\s+\((?:i|a|x|1)\)\s+(?:an amount in cash equal to\s+)?\$\s?{_NUM}", re.I),
        # cash-and-stock merger consideration: "receive 0.2686 shares of LabCorp common stock and $75.76 in cash"
        re.compile(rf"(?:entitled to|right to|will) receive\s+(?:[^.$]|\.\d){{0,120}}?\$\s?{_NUM}\s+in cash", re.I),
    ),
    (
        re.compile(rf"\$\s?{_NUM}\s+per\s+share[^.]{{0,60}}?in cash", re.I),
        re.compile(rf"to (?:purchase|acquire) all[^.]{{0,200}}?for\s+\$\s?{_NUM}\s+per\s+share", re.I),
    ),
)
_LEGAL_SUFFIX = re.compile(
    r"(?:\b(?:Inc|Incorporated|Corp|Corporation|Company|Co|Ltd|Limited|plc|PLC|LLC|L\.L\.C|LP|L\.P|SA|S\.A|AG|NV|N\.V"
    r"|SE|GmbH|S\.p\.A|B\.V|BV|KGaA|SAS|S\.A\.S|ULC|AB|A/S|ASA|Oyj|K\.K)\.?)$"
)
_CVR = re.compile(r"contingent value right", re.I)
# Any mention, abbreviation included: the sentences whose amounts are CVR payments, not the upfront price.
_CVR_MENTION = re.compile(r"contingent value right|\bCVRs?\b")
_CVR_AMOUNTS = (
    re.compile(rf"up to\s+(?:an aggregate (?:amount )?of\s+)?\$\s?{_NUM}", re.I),
    re.compile(rf"\$\s?{_NUM}\s+(?:in cash\s+)?per\s+CVR", re.I),
)
_CVR_SENTENCE_AMOUNT = re.compile(
    rf"(?:right to receive|contingent (?:cash )?payments? of)\s+(?:a contingent (?:cash )?payment of\s+)?(?:up to\s+)?\$\s?{_NUM}",
    re.I,
)
_EXCHANGE_RATIO = re.compile(r"exchange ratio", re.I)
_HOLDERS_PCT = re.compile(
    r"([A-Z][A-Za-z0-9\-]+)(?:'s)?\s+(?:pre-merger\s+|existing\s+|current\s+|legacy\s+|former\s+)?"
    r"(?:stock|share)holders[^.%]{0,150}?\b(?:own|hold)\b[^.%]{0,40}?approximately\s+(\d{1,2}(?:\.\d+)?)\s?%"
)
_MERGER_AGREEMENT = re.compile(
    r"agreement and plan of merger|merger agreement|business combination agreement|arrangement agreement"
    r"|implementation agreement|transaction agreement",
    re.I,
)
_NOTES = re.compile(r"\bnotes\b", re.I)
_MINI_TENDER = re.compile(r"mini-tender", re.I)
_SPAC = re.compile(r"blank check company|special purpose acquisition", re.I)
_ASSET_SALE = re.compile(r"asset purchase agreement|asset sale proposal|sale of substantially all", re.I)
DESPAC_MIN_MENTIONS = 5
STOCK_DEAL_MIN_EXCHANGE_RATIOS = 10
MAX_PRICE_PER_SHARE = 1000.0  # above this it is an aggregate, a per-100-shares example or a preferred share
# Worked examples and preferred-stock terms: "if you own 1,000 shares ... you will receive $31,260 in cash".
_ILLUSTRATION = re.compile(r"if you (?:own|hold)|for example|preferred stock", re.I)
ASSET_SALE_MIN_MENTIONS = 5
_REVERSE_MERGER_WORDS = re.compile(r"reverse merger", re.I)
DEBT_EXCHANGE_MIN_NOTES = 100  # a registered debt exchange offer says "notes" hundreds of times ...
DEBT_EXCHANGE_MAX_AGREEMENTS = 2  # ... and never describes a merger agreement
_COMPLETED_ON = re.compile(
    rf"On\s+({_DATE}),?[^.]{{0,250}}?(?:completed|consummated|closed)\s+(?:the|its)\s+(?:previously announced\s+)?"
    rf"(?:Merger|acquisition|Offer|tender offer|Transaction)",
    re.I,
)


@dataclass
class DealTerms:
    """What could be read off a deal's filings. Every field is optional — absent means not found."""

    agreement_date: Optional[date] = None
    acquirer_name: Optional[str] = None
    parties: Dict[str, str] = field(default_factory=dict)
    upfront_per_share: Optional[float] = None
    has_cvr: Optional[bool] = None
    cvr_max_per_share: Optional[float] = None
    stock_consideration_pct: Optional[float] = None
    target_holders_pct: Optional[float] = None
    completion_date: Optional[date] = None
    completed: bool = False
    is_debt_exchange: bool = False  # registered exchange offer for the company's own notes, not M&A
    is_mini_tender: bool = False  # unsolicited offer for <5% of the shares, not M&A
    is_despac: bool = False  # merger with a blank-check SPAC: the company was private before it
    is_asset_sale: bool = False  # the company sells assets, its shares are not acquired
    mentions_reverse_merger: bool = False


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def filing_text(document: str) -> str:
    """Plain text of an HTML/TXT filing: tags and scripts dropped, entities and curly quotes
    normalized, whitespace collapsed, truncated to `MAX_TEXT_CHARS`."""
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", document)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = text.translate(_CURLY_QUOTES)
    return re.sub(r"\s+", " ", text).strip()[:MAX_TEXT_CHARS]


def parse_long_date(value: str) -> Optional[date]:
    """`"March 2, 2024"` -> `date(2024, 3, 2)`; None if it does not parse."""
    try:
        return datetime.strptime(re.sub(r"\s+", " ", value.replace(" ,", ",")).strip(), "%B %d, %Y").date()
    except ValueError:
        return None


def name_key(name: str) -> str:
    """The first significant word of a company name, lowercased — enough to tell the subject company
    from its counterparty ("Salix Pharmaceuticals, Ltd." and "SALIX PHARMACEUTICALS LTD" -> "salix")."""
    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in _NAME_SUFFIXES]
    return words[0] if words else ""


def _is_name_word(word: str) -> bool:
    """A word that can sit inside a company name: capitalized, a number, a joiner, or a suffix like "plc"."""
    return word[:1].isupper() or word[:1].isdigit() or word in _NAME_JOINERS or word.strip(".,").lower() in _NAME_SUFFIXES


def _is_name_phrase(name: str) -> bool:
    """Every word can sit inside a company name ("and" only before a capitalized word: "Eli Lilly and
    Company", "King Kee Appraisal and Advisory Limited"), and no sentence break."""
    words = name.split(" ")
    if any(w.endswith(".") and len(w.strip(".")) > 3 and w.strip(".,").lower() not in _NAME_SUFFIXES for w in words[:-1]):
        return False  # "Forecasts." ends a sentence; "F." / "Co." / "S.A." do not
    if any(w.lower() == "the" for w in words[1:]):
        return False  # "The Medicines Company", but not "Term Sheet On January 21, 2026, the Company"
    return all(
        _is_name_word(w) or (w == "and" and 0 < i < len(words) - 1 and words[i + 1][:1].isupper())
        for i, w in enumerate(words)
    )


def _trim_to_label(name: str, label: str) -> str:
    """Drop a lead-in joined by "of" before the label's word: label "Novogen", name "Extraordinary General
    Meeting of Novogen Limited" -> "Novogen Limited". "F. Hoffmann-La Roche Ltd" (label "Roche") stays."""
    first = label.split()[0]
    words = name.split(" ")
    for i in range(1, len(words)):
        if words[i] == first and words[i - 1] == "of":
            return " ".join(words[i:])
    return name


def _clean_party_name(raw: str, extra_cuts: Tuple[str, ...] = ()) -> str:
    """The company name at the end of the text preceding a `("Label")` / `which we refer to as` definition."""
    text = " " + raw.strip()
    cuts = _PARTY_CUTS + extra_cuts
    cut_at = max((text.rfind(sep) + len(sep) for sep in cuts if text.rfind(sep) >= 0), default=0)
    name = text[cut_at:].strip()
    name = re.split(r",\s+an?\s+", name, maxsplit=1)[0]
    # Keep only the trailing comma-separated segments that make up the name: "As previously announced,
    # on November 16, 2014, Actavis plc" -> "Actavis plc"; "Allos Therapeutics, Inc." stays whole.
    segments = [s.strip() for s in name.split(", ")]
    kept = [segments.pop()]
    while segments and re.fullmatch(_LEGAL_SUFFIX.pattern, kept[0]):
        kept.insert(0, segments.pop())
    # Then the trailing run of capitalized words, allowing "of"/"&" inside: "the boards of directors of
    # Bristol-Myers Squibb Company" -> "Bristol-Myers Squibb Company", while "Laboratory Corporation of
    # America Holdings" stays whole.
    words = ", ".join(kept).split(" ")
    start = len(words)
    while start > 0 and (_is_name_word(words[start - 1]) or (words[start - 1] == "and" and start < len(words) and words[start] == "Company")):
        start -= 1  # "Eli Lilly and Company"
    while start < len(words) and words[start] in _NAME_JOINERS:
        start += 1
    return " ".join(words[start:]).strip(" ,;")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_agreement_date(text: str, filed: Optional[date]) -> Optional[date]:
    """The merger-agreement date stated most often in the text, among dates plausible for a filing made
    on `filed` (on or before it, and at most `AGREEMENT_LOOKBACK_DAYS` earlier); ties go to the earliest,
    since amendments are dated after the original signing. Most often, because a proxy repeats the real
    date dozens of times while near-misses ("the merger agreement and ... its opinion, dated February 4")
    appear once."""
    found: List[date] = []
    for pattern in _AGREEMENT_DATE_PATTERNS:
        for match in pattern.finditer(text):
            parsed = parse_long_date(match.group(1))
            if parsed is None:
                continue
            if filed is not None and not (filed - timedelta(days=AGREEMENT_LOOKBACK_DAYS) <= parsed <= filed):
                continue
            found.append(parsed)
    if not found:
        return None
    counts = Counter(found)
    return min(counts, key=lambda d: (-counts[d], d))


def _name_before(text: str, end: int, label: str) -> Optional[str]:
    """
    The legal name starting with the label's first word in the 300 characters before `end` — for a
    definition whose name holds its own parenthetical: `Novartis AG, a company limited by shares
    ( Aktiengesellschaft ) incorporated under the laws of Switzerland (" Novartis ")`.
    """
    first = label.split()[0]
    window = text[max(0, end - 300):end]
    found = None
    for match in re.finditer(rf"\b{re.escape(first)}\b[A-Za-z0-9&.,\- ]{{0,60}}", window):
        words = match.group(0).split(" ")
        for n in range(len(words), 0, -1):
            candidate = " ".join(words[:n]).strip(" ,")
            # Prose that happens to end in a suffix word is no name: "Merger Agreement, and in accordance
            # with the General Corporation [Law]", "Financial Forecasts. Historically, the Company".
            if _LEGAL_SUFFIX.search(candidate) and _is_name_phrase(candidate):
                found = candidate
                break
    return found


def extract_parties(text: str) -> Dict[str, str]:
    """`{label: name}` for every party defined as `Name ("Label")`, first definition of a label wins.
    Only names ending in a legal-entity suffix count — filings define plenty of non-parties the same
    way (`Securities and Exchange Commission ("SEC")`, `Merger Agreement`, `Offer Price`)."""
    parties: Dict[str, str] = {}
    for match in _PARTY_DEF.finditer(text):
        label = match.group(2).strip()
        name = _clean_party_name(match.group(1))
        if label and name_key(label) not in name.lower():
            name = _name_before(text, match.start(2), label) or name
        name = _trim_to_label(name, label) if label else name
        if (
            label not in parties and 2 <= len(name) <= 120 and _LEGAL_SUFFIX.search(name)
            and name[0].isupper() and ":" not in name and name_key(name)
            and _is_name_phrase(name)  # not "Term Sheet On January 21, 2026, the Company"
        ):
            parties[label] = name
    # "X, which we refer to as Label" is used for parties only, so no legal suffix is required.
    for match in _PARTY_REFER.finditer(text):
        label = match.group(2).strip()
        name = _clean_party_name(_SENTENCE_BREAK.split(match.group(1))[-1], extra_cuts=(" and ",))
        if label not in parties and 2 <= len(name) <= 120 and name[0].isupper() and len(name.split()) >= 2 \
                and ":" not in name:
            parties[label] = name
    return parties


def pick_acquirer(
    parties: Dict[str, str],
    subject_name: str,
    roster_names: Iterable[str] = (),
    *,
    allow_any_party: bool = False,
    mentions: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    """
    The counterparty to the subject company among the defined parties.

    Order: a party whose name is on the acquirer roster (so an ultimate parent wins over the
    intermediate holding company a tender offer is often made through), then a `Parent`-like label,
    then the eligible party whose label the text uses most often, if at least `MIN_ACQUIRER_MENTIONS`
    times (`mentions`: label -> count; proxies call the buyer "Actavis" or "LabCorp", not "Parent"),
    then — only with `allow_any_party`, used for reverse mergers, where the subject itself is often
    defined as "Parent" — the first eligible party. Eligible: neither the subject, a merger subsidiary,
    nor an exchange/agent/advisor (`_NOT_COUNTERPARTY`).
    """
    subject = name_key(subject_name)
    others = {
        label: name for label, name in parties.items()
        if name_key(name) != subject and name_key(label) != subject and label not in _SUBJECT_LABELS
        and subject not in re.findall(r"[a-z0-9]+", name.lower())  # "Liquidation of Footstar, Inc."
        and label not in _SUB_LABELS
        and "merger sub" not in name.lower()
    }
    roster = {name_key(n) for n in roster_names}
    for label, name in others.items():
        # A roster company named in passing (Vertex as a licensor in a reverse-merger S-4) is not the buyer.
        named_as_buyer = mentions is None or "Parent" in label or mentions.get(label, 0) >= MIN_ACQUIRER_MENTIONS
        if name_key(name) in roster and named_as_buyer:
            return name
    for label in _PARENT_LABELS:
        if label in others:
            return others[label]
    eligible = {label: n for label, n in others.items() if not any(w in n.lower() for w in _NOT_COUNTERPARTY)}
    if mentions:
        ranked = sorted(eligible, key=lambda label: -mentions.get(label, 0))
        if ranked and mentions.get(ranked[0], 0) >= MIN_ACQUIRER_MENTIONS:
            return eligible[ranked[0]]
    if not allow_any_party:
        return None
    return next(iter(eligible.values()), None)


def extract_terms(text: str, *, filed: Optional[date], subject_name: str, roster_names: Iterable[str] = ()) -> DealTerms:
    """Every term `DealTerms` holds that this text states."""
    terms = DealTerms()
    terms.agreement_date = extract_agreement_date(text, filed)
    terms.parties = extract_parties(text)

    cvr_matches = list(_CVR.finditer(text))
    terms.has_cvr = bool(cvr_matches)
    cvr_amounts: List[float] = []
    cvr_spans: List[Tuple[int, int]] = []
    for match in cvr_matches[:20]:
        window = text[match.start():match.start() + 600]
        cvr_amounts.extend(_amount(m) for p in _CVR_AMOUNTS for m in p.finditer(window))
        # Within the sentence that introduces the CVR, any "right to receive $X" is the CVR's amount:
        # "one contingent value right (a "CVR"), which represents the right to receive $2.63 in cash".
        cvr_amounts.extend(_amount(m) for m in _CVR_SENTENCE_AMOUNT.finditer(_sentence_from(window)))
    if cvr_matches:
        for match in _CVR_MENTION.finditer(text):
            cvr_spans.append((match.start(), match.start() + len(_sentence_from(text[match.start():match.start() + 600]))))
    if cvr_amounts:
        terms.cvr_max_per_share = max(cvr_amounts)
    has_exchange_ratio = bool(_EXCHANGE_RATIO.search(text))
    terms.upfront_per_share = extract_price(
        text, exclude=set(cvr_amounts), skip_spans=cvr_spans,
        # a stock deal repeats its exchange ratio dozens of times; a cash deal may name one for options
        generic_ok=len(_EXCHANGE_RATIO.findall(text)) < STOCK_DEAL_MIN_EXCHANGE_RATIOS,
    )
    if terms.upfront_per_share is not None and not has_exchange_ratio:
        terms.stock_consideration_pct = 0.0
    elif terms.upfront_per_share is None and has_exchange_ratio:
        terms.stock_consideration_pct = 1.0

    subject = name_key(subject_name)
    for match in _HOLDERS_PCT.finditer(text):
        if name_key(match.group(1)) == subject:
            terms.target_holders_pct = float(match.group(2)) / 100.0
            break
    mentions = {label: len(re.findall(rf"\b{re.escape(label)}\b", text)) for label in terms.parties}
    terms.acquirer_name = pick_acquirer(
        terms.parties, subject_name, roster_names, allow_any_party=terms.target_holders_pct is not None,
        mentions=mentions,
    )

    terms.completion_date, terms.completed = extract_completion(text)
    agreements = len(_MERGER_AGREEMENT.findall(text))
    terms.is_debt_exchange = agreements <= DEBT_EXCHANGE_MAX_AGREEMENTS and len(_NOTES.findall(text)) >= DEBT_EXCHANGE_MIN_NOTES
    terms.is_mini_tender = bool(_MINI_TENDER.search(text))
    terms.is_despac = len(_SPAC.findall(text)) >= DESPAC_MIN_MENTIONS
    asset_sales = len(_ASSET_SALE.findall(text))
    terms.is_asset_sale = asset_sales >= ASSET_SALE_MIN_MENTIONS and asset_sales > agreements
    terms.mentions_reverse_merger = bool(_REVERSE_MERGER_WORDS.search(text))
    return terms


def _sentence_from(window: str) -> str:
    """`window` up to the end of its first sentence."""
    return re.split(r"\.\s+(?=[A-Z])", window, maxsplit=1)[0]


def _amount(match: "re.Match[str]") -> float:
    return float(match.group(1).replace(",", ""))


def extract_price(
    text: str, exclude: Optional[set] = None, skip_spans: Iterable[Tuple[int, int]] = (), *, generic_ok: bool = True,
) -> Optional[float]:
    """
    Upfront cash per share: the most frequent amount in the first `_PRICE_TIERS` tier that matches.

    Amounts inside `skip_spans` (the sentences introducing a CVR: "one contingent value right, which
    represents the right to receive $2.63 in cash") are dropped from every tier; `exclude` (CVR amounts)
    only from the generic last tier — dropping them everywhere would lose an upfront price that happens
    to equal the CVR (Regulus/Novartis: $7.00 plus a $7.00 CVR). `generic_ok=False` skips the generic
    last tier — for a stock deal, whose only "$X per share in cash" is an earlier proposal (Aratana/Elanco).
    """
    spans = list(skip_spans)
    for tier_index, tier in enumerate(_PRICE_TIERS):
        amounts = [
            _amount(m) for pattern in tier for m in pattern.finditer(text)
            if not any(start <= m.start(1) < end for start, end in spans)
            and not _ILLUSTRATION.search(text, max(0, m.start(1) - 150), m.start(1))
        ]
        if tier_index == len(_PRICE_TIERS) - 1 and not generic_ok:
            break
        if tier_index == len(_PRICE_TIERS) - 1 and exclude:
            amounts = [a for a in amounts if a not in exclude]
        amounts = [a for a in amounts if 0 < a < MAX_PRICE_PER_SHARE]
        if amounts:
            return Counter(amounts).most_common(1)[0][0]
    return None


def extract_completion(text: str) -> Tuple[Optional[date], bool]:
    """`(completion_date, completed)` from an "On <date>, ... completed the Merger" sentence."""
    for match in _COMPLETED_ON.finditer(text):
        parsed = parse_long_date(match.group(1))
        if parsed is not None:
            return parsed, True
    return None, False


def merge_terms(primary: DealTerms, fallback: DealTerms) -> DealTerms:
    """`primary` with each missing field taken from `fallback` (the cluster's last filing); completion
    always comes from the later filing, since the first one predates closing."""
    merged = DealTerms(**{k: v for k, v in vars(primary).items()})
    for name in ("agreement_date", "acquirer_name", "upfront_per_share", "cvr_max_per_share",
                 "stock_consideration_pct", "target_holders_pct"):
        if getattr(merged, name) is None:
            setattr(merged, name, getattr(fallback, name))
    merged.has_cvr = bool(primary.has_cvr or fallback.has_cvr)
    if fallback.completed:
        merged.completed, merged.completion_date = True, fallback.completion_date
    return merged


def suggest_deal_type(form_type: str, terms: DealTerms) -> Tuple[Optional[str], Optional[str]]:
    """
    `(deal_type, reject_note)`: a spec §2.5 `deal_type` guess, or a note explaining why the row should
    be rejected instead (not an acquisition of the subject company). `(None, None)` when nothing fits.

    An S-4 registers the securities its filer issues, so the subject company filing one is the buyer
    (or is exchanging its own notes) — unless its holders end up with a minority of the combined
    company, the reverse-merger case. A minority stake after a deal that also pays cash (LabCorp/Covance,
    BMS/Celgene) is an ordinary cash-and-stock acquisition, not a reverse merger; an all-stock merger
    whose proxy never says "reverse merger" (Alumis/ACELYRIN, both public) is left for a human.
    """
    acquirer = (terms.acquirer_name or "").lower()
    is_s4 = form_type.upper().startswith("S-4")
    if terms.is_mini_tender:
        return None, "mini-tender offer for a small stake, not an acquisition"
    if terms.is_debt_exchange:
        return None, "debt exchange offer for the company's notes, not an acquisition"
    if terms.is_despac:
        return None, "SPAC merger (de-SPAC): the company was not public before, not an acquisition"
    if terms.is_asset_sale:
        return ("asset_sale", None) if acquirer else (None, None)
    if terms.target_holders_pct is not None:
        if terms.target_holders_pct >= REVERSE_MERGER_MAX_HOLDERS_PCT:
            return None, f"company is the acquirer (its holders keep {terms.target_holders_pct:.0%})"
        if is_s4 or (terms.upfront_per_share is None and terms.mentions_reverse_merger):
            return "reverse_merger", None
        if terms.upfront_per_share is None:
            return None, None  # all-stock merger: a reverse merger only if the buyer was private — a human call
    elif is_s4:
        if terms.mentions_reverse_merger:
            return None, None  # the ownership split was not found: could be either side
        return None, "company filed the S-4 to issue its own stock, so it is the acquirer"
    if terms.upfront_per_share is not None and any(b in acquirer for b in SHELL_BUYERS):
        return "shell_transaction", None
    if terms.upfront_per_share is not None and any(b in acquirer for b in PE_BUYERS):
        return "pe_take_private", None  # a take-private pays cash; KKR as a selling holder (Jazz/Azur) is not one
    if terms.upfront_per_share is not None and acquirer:
        return "strategic_acquisition", None
    return None, None


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def filing_name(row: Dict[str, Any]) -> str:
    """The company's name when it filed (first entry of `names_in_filing`), else its current name. A
    reverse-merger shell is renamed after its private partner (Graphite Bio -> LENZ), so the current name
    would mistake the partner's holders' stake for the subject's."""
    names = str(row.get("names_in_filing") or "")
    first_name = re.split(r"\s{2,}\(|\s\(CIK|\s\|\s", names, maxsplit=1)[0].strip()
    return first_name or str(row.get("company_name") or "")


def _filed(row: Dict[str, Any]) -> Optional[date]:
    try:
        return date.fromisoformat(str(row.get("filed_date") or "")[:10])
    except ValueError:
        return None


def cluster_filings(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group export rows into deals: per company, by filed date, a new cluster after a
    `CLUSTER_GAP_DAYS` gap. Rows without a filed date form their own cluster."""
    by_company: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_company.setdefault(str(row["company_id"]), []).append(row)
    clusters: List[List[Dict[str, Any]]] = []
    for company_rows in by_company.values():
        dated: List[Tuple[date, str, Dict[str, Any]]] = []
        for row in company_rows:
            filed = _filed(row)
            if filed is not None:
                dated.append((filed, str(row["item_id"]), row))
        dated.sort(key=lambda t: (t[0], t[1]))
        current: List[Dict[str, Any]] = []
        previous: Optional[date] = None
        for filed, _, row in dated:
            if current and previous is not None and (filed - previous).days > CLUSTER_GAP_DAYS:
                clusters.append(current)
                current = []
            current.append(row)
            previous = filed
        if current:
            clusters.append(current)
        clusters.extend([r] for r in company_rows if _filed(r) is None)
    return clusters


def roster_cik(acquirer_name: str, roster: Dict[str, Optional[str]]) -> Optional[str]:
    """The roster CIK of the acquirer named `acquirer_name`, matched on `name_key`."""
    key = name_key(acquirer_name)
    return next((cik for name, cik in roster.items() if cik and name_key(name) == key), None) if key else None


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:g}"


def _entry_fields(form_type: str, terms: DealTerms, announcement: Optional[date]) -> Dict[str, str]:
    deal_type, _ = suggest_deal_type(form_type, terms)
    completed = terms.completed and terms.completion_date is not None and announcement is not None \
        and announcement <= terms.completion_date <= date.today()
    return {
        "announcement_date": announcement.isoformat() if announcement else "",
        "deal_type": deal_type or "",
        "acquirer_name": terms.acquirer_name or "",
        "status": "completed" if completed else "",
        "completion_date": terms.completion_date.isoformat() if completed and terms.completion_date else "",
        "upfront_per_share": _fmt(terms.upfront_per_share),
        "has_cvr": "" if terms.has_cvr is None else ("yes" if terms.has_cvr else "no"),
        "cvr_max_per_share": _fmt(terms.cvr_max_per_share) if terms.has_cvr else "",
        "stock_consideration_pct": "" if terms.stock_consideration_pct is None else f"{terms.stock_consideration_pct:.0%}",
        "target_holders_pct": "" if terms.target_holders_pct is None else f"{terms.target_holders_pct:.1%}",
    }


def prefill_cluster(
    cluster: List[Dict[str, Any]],
    fetch_text: Callable[[Dict[str, Any]], Optional[str]],
    *,
    roster: Optional[Dict[str, Optional[str]]] = None,
    context_fn: Optional[Callable[[int, Any], Any]] = None,
    existing_deals: Optional[List[Dict[str, Any]]] = None,
    cluster_id: int = 0,
) -> List[Dict[str, Any]]:
    """
    Draft entry columns for one deal's rows.

    Args:
        cluster: Export rows of one company, earliest filing first (`cluster_filings`).
        fetch_text: Row -> plain filing text (`filing_text` of the document), or None if unavailable.
        roster: Acquirer roster `{name: cik}` — preferred when choosing among defined parties, and the
            source of `acquirer_cik`.
        context_fn: `(company_id, DealEntry) -> DealContext` (`deal_entry.gather_context`); when given,
            premiums and reverse-merger flags are checked.
        existing_deals: The company's deals already in `p22_deal`; a cluster whose first filing is
            within `CLUSTER_GAP_DAYS` after one of them is linked to it instead of entered again.
        cluster_id: Written to `prefill_cluster` so the reviewer can see which rows belong together.

    Returns:
        The rows, each with entry columns and `prefill_*` columns filled in.
    """
    first, last = cluster[0], cluster[-1]
    roster = roster or {}
    roster_names = list(roster)
    out = [dict(row, prefill_cluster=cluster_id) for row in cluster]
    first_filed = _filed(first)

    for deal in existing_deals or []:
        announced = deal["announcement_date"]
        if first_filed is not None and timedelta(0) <= first_filed - announced <= timedelta(days=CLUSTER_GAP_DAYS):
            for row in out:
                row.update(action="link", link_deal_id=str(deal["deal_id"]), prefill_confidence="existing_deal")
            return out

    first_text = fetch_text(first)
    if first_text is None:
        out[0].update(prefill_confidence="low", prefill_evidence="document not fetched")
        return out
    company_name = filing_name(first)
    terms = extract_terms(first_text, filed=first_filed, subject_name=company_name, roster_names=roster_names)
    competing: Optional[str] = None
    if last is not first:
        last_text = fetch_text(last)
        if last_text is not None:
            last_terms = extract_terms(last_text, filed=_filed(last), subject_name=company_name, roster_names=roster_names)
            first_buyer, last_buyer = terms.acquirer_name, last_terms.acquirer_name
            if first_buyer and last_buyer and name_key(first_buyer) != name_key(last_buyer):
                # e.g. Valeant's hostile bid for Allergan, then Actavis as white knight: two deals in one
                # cluster, which only a human can split.
                competing = f"acquirer differs: {first_buyer} (first filing) vs {last_buyer} (last filing)"
            else:
                terms = merge_terms(terms, last_terms)

    announcement = terms.agreement_date
    fields = _entry_fields(str(first.get("form_type") or ""), terms, announcement)
    deal_type, reject_note = suggest_deal_type(str(first.get("form_type") or ""), terms)
    evidence = [
        f"agreement_date={'found' if terms.agreement_date else 'missing'}",
        f"parties={'; '.join(f'{k}={v}' for k, v in list(terms.parties.items())[:4]) or 'none'}",
    ]
    head = out[0]
    head.update(fields)
    head["acquirer_cik"] = roster_cik(terms.acquirer_name or "", roster) or ""

    if reject_note is not None:
        for name in fields:
            head[name] = ""  # what was read describes another company's deal; keep only the reason
        head["acquirer_cik"] = ""
        for row in out:
            row.update(action="reject", note=reject_note, prefill_confidence="medium")
        head["prefill_evidence"] = " | ".join(evidence)
        return out

    confidence = "high" if (announcement and terms.acquirer_name and deal_type and terms.upfront_per_share) else "medium"
    if not (announcement and terms.acquirer_name and deal_type):
        confidence = "low"
    if competing is not None:
        confidence = "low"
        evidence.insert(0, competing)

    if confidence != "low" and context_fn is not None:
        try:
            entry = entry_from_mapping(fields)
            context = context_fn(int(first["company_id"]), entry)
        except DealEntryError as exc:
            context = None
            confidence = "low"
            evidence.append(f"invalid entry: {exc}")
        if context is not None:
            head["prefill_premium_1d"] = _fmt(context.premium_1d)
            head["prefill_premium_30d"] = _fmt(context.premium_30d)
            head["prefill_flags"] = ",".join(context.flags_fired)
            low, high = PREMIUM_SANE_RANGE
            if deal_type == "strategic_acquisition" and context.flags_fired:
                confidence = "flags"
            elif deal_type != "reverse_merger" and context.premium_1d is not None and not low <= context.premium_1d <= high:
                confidence = "check_price"

    head["prefill_confidence"] = confidence
    head["prefill_evidence"] = " | ".join(evidence)
    if confidence in ("high", "medium"):
        head["action"] = "deal"
        for row in out[1:]:
            row.update(action="link", prefill_confidence=confidence)
    return out


PREFILL_COLUMNS = [
    "prefill_confidence", "prefill_cluster", "prefill_premium_1d", "prefill_premium_30d", "prefill_flags",
    "prefill_evidence",
]
