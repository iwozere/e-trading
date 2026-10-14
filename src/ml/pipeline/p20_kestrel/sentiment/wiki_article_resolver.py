"""
P20 Kestrel — resolve tickers to English Wikipedia article titles.

Resolution chain (first hit wins), cached on disk so steady-state runs make no
resolution requests at all:

1. **Wikidata** — items listed on NYSE/Nasdaq/NYSE American under that ticker
   symbol (P414 + P249), with an enwiki sitelink. Covers roughly half the
   watchlist; Wikidata ticker tagging is sparse for small/mid caps (even
   PYPL/PINS/RDDT lack it) and keeps stale listings, so each hit is cross-checked
   against the article's own infobox and dropped if it contradicts.
2. **Verified search** — full-text search by ticker and by the company's legal
   name; a candidate is accepted only if its infobox ``traded_as`` field lists the
   exact ticker on a US exchange (``{{nasdaq|MNDY}}``, ``{{ubl|{{NYSE|BOOT}}}}``;
   not Almirall's ``{{BMAD|...|ALM}}``).
3. **Name-title match** — a name-search hit whose normalized title equals the
   normalized legal name (``Energy Fuels Inc.`` ↔ ``Energy Fuels Inc``) and whose
   infobox doesn't contradict; catches articles that omit ``traded_as``.

Unresolved tickers are cached too (as ``None``) and retried after
``_CACHE_TTL_DAYS``, so a ticker with no article costs nothing day to day.
Tickers whose lookup *failed* (network, 429, 5xx) are never cached as unresolved:
they are retried on the next run.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.ml.pipeline.p20_kestrel.config import DATA_CACHE_PATH
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Wikimedia's User-Agent policy requires a contact address; generic UAs get throttled.
USER_AGENT = "e-trading-research akossyrev@gmail.com"

_WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
_WIKI_API = "https://en.wikipedia.org/w/api.php"
_CACHE_PATH = DATA_CACHE_PATH / "wikipedia" / "ticker_articles.json"
_CACHE_TTL_DAYS = 30
_WIKIDATA_BATCH = 100
_WIKITEXT_BATCH = 50  # MediaWiki API max titles per query
_SEARCH_LIMIT = 10
_REQUEST_DELAY_S = 0.2
_RETRY_BACKOFF_S = 5.0
_TIMEOUT_S = 30

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|sa|nv|ag|se|holdings?|group|lp|llc)\b\.?",
    re.IGNORECASE,
)
# Field value runs until the next ``| field =`` line or the infobox's closing braces — NOT the next bare
# ``|``: multi-line values (``{{ubl`` / ``  | {{NASDAQ|PYPL}}`` / ...) continue with pipe-led lines.
_TRADED_AS = re.compile(r"\|\s*traded_as\s*=(.*?)(?=\n\s*\|\s*[\w ]+=|\n\}\})", re.IGNORECASE | re.DOTALL)
# US listing templates only: a foreign listing sharing the ticker (Almirall is {{BMAD|...|ALM}}) must not match.
_US_EXCHANGE_TEMPLATE = r"\{\{\s*(?:NYSE(?:\s*(?:American|Arca|MKT))?|NASDAQ|AMEX)\s*\|"
# Wikidata items: NYSE, Nasdaq, NYSE American.
_US_EXCHANGE_QIDS = "wd:Q13677 wd:Q82059 wd:Q846626"


class _LookupFailed(Exception):
    """A resolution request failed transiently; the ticker must stay uncached so the next run retries it."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the ticker → {title, resolved_at} cache; a missing/corrupt file is an empty cache."""
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        _logger.warning("Wikipedia article cache unreadable at %s; rebuilding", path)
        return {}


def _save_cache(path: Path, cache: Mapping[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _is_fresh(entry: Mapping[str, Any], now: datetime) -> bool:
    try:
        resolved_at = datetime.fromisoformat(str(entry["resolved_at"]))
    except (KeyError, ValueError):
        return False
    return now - resolved_at < timedelta(days=_CACHE_TTL_DAYS)


def _normalize_name(name: str) -> str:
    """Lowercase, strip legal suffixes and punctuation: 'Energy Fuels Inc.' → 'energy fuels'."""
    stripped = _LEGAL_SUFFIXES.sub(" ", name.replace("_", " "))
    stripped = re.sub(r"\(.*?\)", " ", stripped)
    return " ".join(re.sub(r"[^a-z0-9& ]", " ", stripped.lower()).split())


def _traded_as_has_ticker(wikitext: str, ticker: str) -> bool:
    """True if the infobox ``traded_as`` field lists exactly this ticker on a US exchange."""
    match = _TRADED_AS.search(wikitext)
    if not match:
        return False
    pattern = _US_EXCHANGE_TEMPLATE + r"\s*(?:symbol\s*=\s*)?%s\s*[|}]" % re.escape(ticker)
    return re.search(pattern, match.group(1), re.IGNORECASE) is not None


def _not_contradicted(wikitext: str, ticker: str) -> bool:
    """
    True if the article lists this US ticker, or has no ``traded_as`` field at all.

    Used for the weaker evidence paths (Wikidata, name match): Wikidata keeps stale
    listings (Pandora's delisted ``P``; ``DV`` after DoubleVerify became Covista/CVSA),
    and those articles' own infoboxes show the ticker is no longer theirs.
    """
    return _traded_as_has_ticker(wikitext, ticker) or _TRADED_AS.search(wikitext) is None


def _get_json(session: requests.Session, url: str, params: Dict[str, str], timeout: float) -> Any:
    """GET with one retry on network errors, 429 and 5xx. Raises _LookupFailed if both attempts fail."""
    reason = ""
    for attempt in range(2):
        if attempt:
            time.sleep(_RETRY_BACKOFF_S)
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                reason = f"HTTP {resp.status_code}"
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            reason = str(exc)
        finally:
            time.sleep(_REQUEST_DELAY_S)
    raise _LookupFailed(f"{url}: {reason}")


def _fetch_wikitexts(session: requests.Session, titles: List[str]) -> Dict[str, str]:
    """
    Return {requested title: wikitext} for up to ``_WIKITEXT_BATCH`` titles, following redirects.

    Keyed by the *requested* title (normalization/redirect targets are mapped back) so
    callers can look up what they asked for: "Sellas Life Sciences Group" redirects to
    "Galena Biopharma" after the merger.

    Raises:
        _LookupFailed: On a transient request failure.
    """
    if not titles:
        return {}
    data = _get_json(
        session,
        _WIKI_API,
        {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
         "titles": "|".join(titles[:_WIKITEXT_BATCH]), "redirects": "1", "format": "json", "formatversion": "2"},
        _TIMEOUT_S,
    )
    query = data.get("query", {})
    texts = {p["title"]: p["revisions"][0]["slots"]["main"]["content"] for p in query.get("pages", []) if p.get("revisions")}
    final = {t: t for t in titles[:_WIKITEXT_BATCH]}
    for hop in query.get("normalized", []) + query.get("redirects", []):
        for requested, current in final.items():
            if current == hop["from"]:
                final[requested] = hop["to"]
    return {requested: texts[target] for requested, target in final.items() if target in texts}


def _resolve_via_wikidata(session: requests.Session, tickers: List[str]) -> tuple[Dict[str, str], set[str]]:
    """
    Batch-resolve tickers through Wikidata US-listing ticker statements.

    Returns:
        (hits, failed): hits are unambiguous and not contradicted by the article's own
        infobox; failed are tickers whose lookup errored and must stay uncached.
    """
    found: Dict[str, set[str]] = {}
    failed: set[str] = set()
    for i in range(0, len(tickers), _WIKIDATA_BATCH):
        batch = tickers[i : i + _WIKIDATA_BATCH]
        query = (
            "SELECT ?ticker ?article WHERE { VALUES ?ticker { %s } VALUES ?exch { %s } "
            # No end-date (P582) filter: it turns a ~1s query into a 504 timeout. Stale listings are
            # caught by the infobox cross-check below instead.
            "{ ?item p:P414 ?st . ?st ps:P414 ?exch ; pq:P249 ?ticker . } "
            "UNION { ?item wdt:P249 ?ticker ; wdt:P414 ?exch . } "
            "?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }"
        ) % (" ".join(json.dumps(t) for t in batch), _US_EXCHANGE_QIDS)
        try:
            bindings = _get_json(session, _WIKIDATA_SPARQL, {"query": query, "format": "json"}, _TIMEOUT_S * 2)[
                "results"
            ]["bindings"]
        except (_LookupFailed, KeyError, TypeError):
            _logger.warning("Wikidata ticker lookup failed for %d tickers (%s..%s)", len(batch), batch[0], batch[-1])
            failed.update(batch)
            continue
        for b in bindings:
            title = unquote(b["article"]["value"].rsplit("/", 1)[-1]).replace("_", " ")
            found.setdefault(b["ticker"]["value"], set()).add(title)

    # Ambiguous (e.g. SE → Sea Ltd + Spectra Energy) falls through to verified search.
    unique = {t: next(iter(titles)) for t, titles in found.items() if len(titles) == 1}

    # Cross-check each hit against the article itself; contradicted hits fall through to search.
    hits: Dict[str, str] = {}
    titles = sorted(set(unique.values()))
    for i in range(0, len(titles), _WIKITEXT_BATCH):
        chunk = set(titles[i : i + _WIKITEXT_BATCH])
        try:
            wikitexts = _fetch_wikitexts(session, sorted(chunk))
        except _LookupFailed:
            failed.update(t for t, title in unique.items() if title in chunk)
            continue
        hits.update(
            (t, title) for t, title in unique.items()
            if title in wikitexts and _not_contradicted(wikitexts[title], t)
        )
    return hits, failed


def _search_titles(session: requests.Session, query: str) -> List[str]:
    """Full-text search article titles. Raises _LookupFailed on a transient failure."""
    data = _get_json(
        session,
        _WIKI_API,
        {"action": "query", "list": "search", "srsearch": query, "srlimit": str(_SEARCH_LIMIT),
         "srnamespace": "0", "format": "json"},
        _TIMEOUT_S,
    )
    return [h["title"] for h in data.get("query", {}).get("search", [])]


def _resolve_via_search(session: requests.Session, ticker: str, name: Optional[str]) -> Optional[str]:
    """
    Search by ticker and legal name; accept an infobox-verified hit, else an exact name-title match.

    Raises:
        _LookupFailed: If any request failed — "no article" must not be concluded from a failed lookup.
    """
    queries = [f'"{ticker}" hastemplate:"Infobox company"', f'"NASDAQ: {ticker}" OR "NYSE: {ticker}"']
    if name:
        queries.append(name)

    candidates: List[str] = []
    for query in queries:
        candidates.extend(t for t in _search_titles(session, query) if t not in candidates)
    if not candidates:
        return None

    wikitexts = _fetch_wikitexts(session, candidates)
    verified = [title for title, text in wikitexts.items() if _traded_as_has_ticker(text, ticker)]
    if len(verified) == 1:
        return verified[0]
    if len(verified) > 1:
        _logger.debug("Ticker %s verified on several articles %s; skipping", ticker, verified)
        return None

    if name:
        target = _normalize_name(name)
        exact = [t for t in candidates if target and _normalize_name(t) == target]
        if len(exact) == 1 and exact[0] in wikitexts and _not_contradicted(wikitexts[exact[0]], ticker):
            return exact[0]
    return None


def resolve_articles(
    tickers: Iterable[str],
    names: Optional[Mapping[str, str]] = None,
    cache_path: Path = _CACHE_PATH,
) -> Dict[str, Optional[str]]:
    """
    Map tickers to English Wikipedia article titles, using and refreshing the on-disk cache.

    Args:
        tickers: Ticker symbols to resolve.
        names: Optional ticker → company legal name, used by the search fallbacks.
        cache_path: Location of the JSON cache.

    Returns:
        Dict ticker → article title, or None when no article could be resolved. A ticker
        whose lookup failed this run keeps its previous (stale) cached title, if any.
    """
    names = names or {}
    now = _now()
    cache = _load_cache(cache_path)
    wanted = sorted({t.upper() for t in tickers})
    pending = [t for t in wanted if t not in cache or not _is_fresh(cache[t], now)]

    if pending:
        _logger.info("Resolving Wikipedia articles for %d tickers (%d cached)", len(pending), len(wanted) - len(pending))
        with requests.Session() as session:
            session.headers.update({"User-Agent": USER_AGENT})
            hits, wikidata_failed = _resolve_via_wikidata(session, pending)
            resolved: Dict[str, Optional[str]] = dict(hits)
            failed: set[str] = set()
            for ticker in pending:
                if ticker in resolved:
                    continue
                try:
                    resolved[ticker] = _resolve_via_search(session, ticker, names.get(ticker))
                except _LookupFailed as exc:
                    _logger.debug("Search lookup failed for %s: %s", ticker, exc)
                    failed.add(ticker)
                    continue
                # Search found nothing, but Wikidata wasn't asked: not a trustworthy miss.
                if resolved[ticker] is None and ticker in wikidata_failed:
                    failed.add(ticker)
        if failed:
            _logger.warning("Wikipedia resolution incomplete for %d tickers; will retry next run", len(failed))
        stamp = now.isoformat()
        for ticker in pending:
            if ticker not in failed:
                cache[ticker] = {"title": resolved.get(ticker), "resolved_at": stamp}
        _save_cache(cache_path, cache)

    result = {t: cache[t].get("title") if t in cache else None for t in wanted}
    _logger.info("Wikipedia articles: %d/%d tickers resolved", sum(1 for v in result.values() if v), len(result))
    return result
