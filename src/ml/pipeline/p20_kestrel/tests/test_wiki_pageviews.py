"""Tests for the Wikipedia pageviews poll and its ticker → article resolver (§7.3 replacement)."""

import json
from datetime import date, datetime, timedelta, timezone

from typing import Any, cast

import pytest
import requests

from src.ml.pipeline.p20_kestrel.sentiment import wiki_article_resolver as resolver
from src.ml.pipeline.p20_kestrel.sentiment import wiki_pageviews_poll as poll


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise resolver.requests.HTTPError(f"HTTP {self.status_code}")


class _FakeSession:
    """Routes GETs to a handler(url, params) → _FakeResponse; records calls."""

    def __init__(self, handler):
        self._handler = handler
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self._handler(url, params or {})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _as_session(fake: _FakeSession) -> requests.Session:
    return cast(requests.Session, cast(Any, fake))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(resolver.time, "sleep", lambda s: None)
    monkeypatch.setattr(poll.time, "sleep", lambda s: None)


# ---------------------------------------------------------------------------
# Resolver helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wikitext, ticker, expected",
    [
        ("{{Infobox company\n| traded_as = {{nasdaq|MNDY}}\n| founded = 2012\n}}", "MNDY", True),
        ("{{Infobox company\n| traded_as = {{ubl|{{NYSE|BOOT}}|[[S&P 600]]}}\n}}", "BOOT", True),
        ("{{Infobox company\n| traded_as = {{NYSE|BOOTX}}\n}}", "BOOT", False),
        ("{{Infobox company\n| founded = 2012\n}} Ticker {{nasdaq|MNDY}} in prose", "MNDY", False),
        # multi-line value with pipe-led continuation lines (real PayPal infobox shape)
        ("{{Infobox company\n| traded_as = {{ubl\n  | {{NASDAQ|PYPL}}\n  | [[S&P 500]] component\n}}\n| founded = 1998\n}}",
         "PYPL", True),
        # foreign listing sharing the ticker (real Almirall infobox shape)
        ("{{Infobox company\n| traded_as        = {{BMAD|ES0157097017|ALM}}\n| ISIN = x\n}}", "ALM", False),
        # template alias (real Ubiquiti infobox shape)
        ("{{Infobox company\n| traded_as = {{Unbulleted list|class=nowrap|{{New York Stock Exchange|UI}}|"
         "[[Russell 1000]] component}}\n| founded = 2003\n}}", "UI", True),
        # former listing is not a current one
        ("{{Infobox company\n| traded_as = {{NYSE was|P}} (2011-2019)\n}}", "P", False),
    ],
)
def test_traded_as_has_ticker(wikitext, ticker, expected):
    assert resolver._traded_as_has_ticker(wikitext, ticker) is expected


@pytest.mark.parametrize(
    "wikitext, expected",
    [
        # foreign issuer listing only its home exchange (real Novo Nordisk shape) → no evidence against NVO
        ("{{Infobox company\n| traded_as = {{ubl|class=nowrap|{{OMX|CSE1158|NOVO B}}|[[OMX Copenhagen 25]]}}\n}}", True),
        ("{{Infobox company\n| founded = 1923\n}}", True),
        ("{{Infobox company\n| traded_as = {{NYSE|NVO}}\n}}", True),
        ("{{Infobox company\n| traded_as = {{NASDAQ was|GALE}}\n}}", False),  # only a former US listing
        ("{{Infobox company\n| traded_as = {{NYSE|NOVX}}\n}}", False),  # a different current US ticker
    ],
)
def test_not_contradicted(wikitext, expected):
    assert resolver._not_contradicted(wikitext, "NVO") is expected


def test_normalize_name_strips_legal_suffixes():
    assert resolver._normalize_name("Energy Fuels Inc.") == resolver._normalize_name("Energy Fuels Inc")
    assert resolver._normalize_name("On (company)") == "on"
    assert resolver._normalize_name("NOVO NORDISK A/S") == resolver._normalize_name("Novo Nordisk")
    assert resolver._normalize_name("Sea Limited") == resolver._normalize_name("Sea Ltd")


def _wikidata_handler(bindings, wikitexts):
    def handler(url, params):
        if url == resolver._WIKIDATA_SPARQL:
            return _FakeResponse(200, {"results": {"bindings": bindings}})
        pages = [{"title": t, "revisions": [{"slots": {"main": {"content": c}}}]} for t, c in wikitexts.items()]
        return _FakeResponse(200, {"query": {"pages": pages}})

    return handler


def _binding(ticker, slug):
    return {"ticker": {"value": ticker}, "article": {"value": f"https://en.wikipedia.org/wiki/{slug}"}}


def test_wikidata_skips_contradicted_tickers_but_accepts_foreign_only_listing():
    bindings = [
        _binding("AAPL", "Apple_Inc."),
        _binding("T", "AT%26T"),
        _binding("P", "Pandora_(service)"),
        _binding("DV", "Covista"),
        _binding("NVO", "Novo_(company)"),
    ]
    wikitexts = {
        "Apple Inc.": "{{Infobox company\n| traded_as = {{NASDAQ|AAPL}}\n}}",
        "AT&T": "{{Infobox company\n| founded = 1983\n}}",  # no traded_as → not contradicted
        "Pandora (service)": "{{Infobox company\n| traded_as = {{NYSE was|P}} (2011-2019)\n}}",  # stale listing
        "Covista": "{{Infobox company\n| traded_as = {{ubl|{{NYSE|CVSA}}|[[S&P 600]] component}}\n}}",  # renamed
        # ADR issuer listing only its home market (real Novo infobox shape)
        "Novo (company)": "{{Infobox company\n| traded_as = {{ubl|class=nowrap|{{OMX|CSE1158|NOVO B}}}}\n}}",
    }

    result = resolver._resolve_via_wikidata(_as_session(_FakeSession(_wikidata_handler(bindings, wikitexts))),
                                            ["AAPL", "T", "P", "DV", "NVO"])

    assert result == ({"AAPL": "Apple Inc.", "T": "AT&T", "NVO": "Novo (company)"}, set())


def test_wikidata_ambiguous_ticker_needs_exactly_one_current_listing():
    bindings = [
        _binding("SE", "Sea_Ltd"), _binding("SE", "Spectra_Energy"),
        _binding("XX", "Alpha_Corp"), _binding("XX", "Beta_Corp"),
    ]
    wikitexts = {
        "Sea Ltd": "{{Infobox company\n| traded_as = {{NYSE|SE}}\n}}",
        "Spectra Energy": "{{Infobox company\n| traded_as = {{NYSE was|SE}}\n}}",
        "Alpha Corp": "{{Infobox company\n| founded = 1990\n}}",  # not contradicted, but not proof either
        "Beta Corp": "{{Infobox company\n| founded = 1991\n}}",
    }

    result = resolver._resolve_via_wikidata(_as_session(_FakeSession(_wikidata_handler(bindings, wikitexts))),
                                            ["SE", "XX"])

    assert result == ({"SE": "Sea Ltd"}, set())


def test_search_fallback_accepts_only_infobox_verified_candidate():
    pages = {
        "Henry Boot plc": "{{Infobox company\n| traded_as = {{LSE|BOOT}}\n}}",
        "Boot Barn": "{{Infobox company\n| traded_as = {{NYSE|BOOT}}\n}}",
    }

    def handler(url, params):
        if params.get("list") == "search":
            return _FakeResponse(200, {"query": {"search": [{"title": t} for t in pages]}})
        return _FakeResponse(
            200,
            {"query": {"pages": [{"title": t, "revisions": [{"slots": {"main": {"content": c}}}]} for t, c in pages.items()]}},
        )

    # Henry Boot plc is LSE:BOOT — a foreign listing sharing the ticker must not verify.
    assert resolver._resolve_via_search(_as_session(_FakeSession(handler)), "BOOT", None) == "Boot Barn"

    # Two US-verified hits are rejected rather than guessed.
    pages["Boot Barn Holdings"] = "{{Infobox company\n| traded_as = {{NYSE|BOOT}}\n}}"
    assert resolver._resolve_via_search(_as_session(_FakeSession(handler)), "BOOT", None) is None


def test_search_fallback_uses_exact_name_title_when_infobox_lacks_ticker():
    def handler(url, params):
        if params.get("list") == "search":
            return _FakeResponse(200, {"query": {"search": [{"title": "Energy Fuels Inc."}, {"title": "Uranium mining"}]}})
        return _FakeResponse(200, {"query": {"pages": [
            {"title": "Energy Fuels Inc.", "revisions": [{"slots": {"main": {"content": "{{Infobox company\n}}"}}}]},
        ]}})

    assert resolver._resolve_via_search(_as_session(_FakeSession(handler)), "UUUU", "Energy Fuels Inc") == "Energy Fuels Inc."
    assert resolver._resolve_via_search(_as_session(_FakeSession(handler)), "UUUU", None) is None


def _search_handler(pages):
    def handler(url, params):
        if params.get("list") == "search":
            return _FakeResponse(200, {"query": {"search": [{"title": t} for t in pages]}})
        return _FakeResponse(
            200,
            {"query": {"pages": [{"title": t, "revisions": [{"slots": {"main": {"content": c}}}]} for t, c in pages.items()]}},
        )

    return handler


def test_search_fallback_name_settles_parent_and_subsidiary():
    pages = {
        "Garena": "{{Infobox company\n| traded_as = {{NYSE|SE}}\n}}",
        "Sea Ltd": "{{Infobox company\n| traded_as = {{NYSE|SE}}\n}}",
    }
    session = _as_session(_FakeSession(_search_handler(pages)))

    assert resolver._resolve_via_search(session, "SE", "Sea Limited") == "Sea Ltd"
    assert resolver._resolve_via_search(session, "SE", None) is None


def test_search_fallback_name_match_accepts_foreign_issuer_without_us_listing():
    pages = {
        "Novo Nordisk": "{{Infobox company\n| traded_as = {{ubl|{{OMX|CSE1158|NOVO B}}}}\n}}",
        "Novo Nordisk Foundation": "{{Infobox organization\n| founded = 1989\n}}",
    }
    session = _as_session(_FakeSession(_search_handler(pages)))

    assert resolver._resolve_via_search(session, "NVO", "NOVO NORDISK A/S") == "Novo Nordisk"

    # A name match whose infobox shows a *different* current US ticker is still rejected.
    pages["Novo Nordisk"] = "{{Infobox company\n| traded_as = {{NYSE|NOVX}}\n}}"
    assert resolver._resolve_via_search(session, "NVO", "NOVO NORDISK A/S") is None


def test_resolve_articles_retries_misses_from_older_resolver_version(tmp_path, monkeypatch):
    cache_path = tmp_path / "ticker_articles.json"
    now = datetime.now(timezone.utc).isoformat()
    cache_path.write_text(json.dumps({
        "UI": {"title": None, "resolved_at": now},  # pre-versioning miss
        "AAPL": {"title": "Apple Inc.", "resolved_at": now},  # hits stay valid across versions
        "ZZZZ": {"title": None, "resolved_at": now, "version": resolver._RESOLVER_VERSION},
    }))
    asked = []

    def wikidata(session, tickers):
        asked.extend(tickers)
        return {"UI": "Ubiquiti"}, set()

    monkeypatch.setattr(resolver, "_resolve_via_wikidata", wikidata)
    monkeypatch.setattr(resolver.requests, "Session", lambda: _FakeSession(lambda u, p: _FakeResponse()))

    result = resolver.resolve_articles(["UI", "AAPL", "ZZZZ"], cache_path=cache_path)

    assert asked == ["UI"]
    assert result == {"UI": "Ubiquiti", "AAPL": "Apple Inc.", "ZZZZ": None}
    assert json.loads(cache_path.read_text())["UI"]["version"] == resolver._RESOLVER_VERSION


def test_resolve_articles_uses_fresh_cache_without_network(tmp_path, monkeypatch):
    cache_path = tmp_path / "ticker_articles.json"
    now = datetime.now(timezone.utc).isoformat()
    cache_path.write_text(json.dumps({
        "AAPL": {"title": "Apple Inc.", "resolved_at": now},
        "ZZZZ": {"title": None, "resolved_at": now, "version": resolver._RESOLVER_VERSION},
    }))
    monkeypatch.setattr(resolver.requests, "Session", lambda: pytest.fail("network used despite fresh cache"))

    assert resolver.resolve_articles(["aapl", "ZZZZ"], cache_path=cache_path) == {"AAPL": "Apple Inc.", "ZZZZ": None}


def test_resolve_articles_refreshes_stale_and_caches_misses(tmp_path, monkeypatch):
    cache_path = tmp_path / "ticker_articles.json"
    stale = (datetime.now(timezone.utc) - timedelta(days=resolver._CACHE_TTL_DAYS + 1)).isoformat()
    cache_path.write_text(json.dumps({"AAPL": {"title": "Old Title", "resolved_at": stale}}))
    monkeypatch.setattr(resolver, "_resolve_via_wikidata", lambda session, tickers: ({"AAPL": "Apple Inc."}, set()))
    monkeypatch.setattr(resolver, "_resolve_via_search", lambda session, ticker, name: None)
    monkeypatch.setattr(resolver.requests, "Session", lambda: _FakeSession(lambda u, p: _FakeResponse()))

    result = resolver.resolve_articles(["AAPL", "NOPE"], cache_path=cache_path)

    assert result == {"AAPL": "Apple Inc.", "NOPE": None}
    saved = json.loads(cache_path.read_text())
    assert saved["AAPL"]["title"] == "Apple Inc." and saved["NOPE"]["title"] is None


def test_resolve_articles_never_caches_failed_lookups(tmp_path, monkeypatch):
    """A Wikidata 502 or search outage must not pin a ticker as 'no article' for _CACHE_TTL_DAYS."""
    cache_path = tmp_path / "ticker_articles.json"
    stale = (datetime.now(timezone.utc) - timedelta(days=resolver._CACHE_TTL_DAYS + 1)).isoformat()
    cache_path.write_text(json.dumps({"ON": {"title": "Onsemi", "resolved_at": stale}}))

    def search(session, ticker, name):
        raise resolver._LookupFailed("HTTP 503")

    monkeypatch.setattr(resolver, "_resolve_via_wikidata", lambda session, tickers: ({}, {"ON"}))
    monkeypatch.setattr(resolver, "_resolve_via_search", search)
    monkeypatch.setattr(resolver.requests, "Session", lambda: _FakeSession(lambda u, p: _FakeResponse()))

    result = resolver.resolve_articles(["ON", "IT"], cache_path=cache_path)

    assert result == {"ON": "Onsemi", "IT": None}  # stale title kept for this run
    saved = json.loads(cache_path.read_text())
    assert saved == {"ON": {"title": "Onsemi", "resolved_at": stale}}  # neither re-stamped nor cached as a miss


def test_resolve_articles_wikidata_outage_still_uses_search_but_caches_only_hits(tmp_path, monkeypatch):
    cache_path = tmp_path / "ticker_articles.json"
    monkeypatch.setattr(resolver, "_resolve_via_wikidata", lambda session, tickers: ({}, set(tickers)))
    monkeypatch.setattr(resolver, "_resolve_via_search", lambda session, ticker, name: "PayPal" if ticker == "PYPL" else None)
    monkeypatch.setattr(resolver.requests, "Session", lambda: _FakeSession(lambda u, p: _FakeResponse()))

    assert resolver.resolve_articles(["PYPL", "GLW"], cache_path=cache_path) == {"PYPL": "PayPal", "GLW": None}
    assert set(json.loads(cache_path.read_text())) == {"PYPL"}  # GLW's miss is untrustworthy → retried next run


def test_get_json_retries_once_then_raises(monkeypatch):
    monkeypatch.setattr(resolver, "_RETRY_BACKOFF_S", 0)
    session = _FakeSession(lambda u, p: _FakeResponse(502))

    with pytest.raises(resolver._LookupFailed):
        resolver._get_json(_as_session(session), "https://x", {}, 1)
    assert len(session.calls) == 2

    responses = iter([_FakeResponse(502), _FakeResponse(200, {"ok": True})])
    assert resolver._get_json(_as_session(_FakeSession(lambda u, p: next(responses))), "https://x", {}, 1) == {"ok": True}


def test_fetch_wikitexts_maps_redirects_back_to_requested_title():
    data = {"query": {
        "redirects": [{"from": "Sellas Life Sciences Group", "to": "Galena Biopharma"}],
        "pages": [{"title": "Galena Biopharma", "revisions": [{"slots": {"main": {"content": "TEXT"}}}]}],
    }}
    session = _FakeSession(lambda u, p: _FakeResponse(200, data))

    assert resolver._fetch_wikitexts(_as_session(session), ["Sellas Life Sciences Group"]) == {
        "Sellas Life Sciences Group": "TEXT"
    }


# ---------------------------------------------------------------------------
# Pageviews fetch
# ---------------------------------------------------------------------------


def _items(days_views):
    return {"items": [{"timestamp": d.strftime("%Y%m%d00"), "views": v} for d, v in days_views]}


def test_fetch_daily_views_fills_interior_gaps_with_zero():
    d0 = date(2026, 10, 1)
    session = _FakeSession(lambda u, p: _FakeResponse(200, _items([(d0, 10), (d0 + timedelta(days=3), 40)])))

    views = poll._fetch_daily_views(_as_session(session), "Apple Inc.", d0, d0 + timedelta(days=6))

    assert views == {d0: 10, d0 + timedelta(days=1): 0, d0 + timedelta(days=2): 0, d0 + timedelta(days=3): 40}
    assert "Apple_Inc." in session.calls[0][0]


def test_fetch_daily_views_404_means_no_data():
    session = _FakeSession(lambda u, p: _FakeResponse(404))

    assert poll._fetch_daily_views(_as_session(session), "Nothing", date(2026, 10, 1), date(2026, 10, 2)) == {}


def test_fetch_daily_views_raises_on_transient_error():
    session = _FakeSession(lambda u, p: _FakeResponse(503))

    with pytest.raises(poll._PageviewsUnavailable):
        poll._fetch_daily_views(_as_session(session), "Apple Inc.", date(2026, 10, 1), date(2026, 10, 2))


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.fixture
def _db(monkeypatch):
    state = {"upserted": [], "finished": []}
    monkeypatch.setattr(poll, "start_job_run", lambda job, d: None)
    monkeypatch.setattr(poll, "finish_job_run", lambda job, d, **kw: state["finished"].append(kw))
    monkeypatch.setattr(poll, "get_watchlist_tickers", lambda: ["AAPL", "NOPE"])
    monkeypatch.setattr(poll, "get_open_positions", lambda: [{"ticker": "msft"}])
    monkeypatch.setattr(poll, "get_all_aliases", lambda: [{"ticker": "AAPL", "alias": "Apple Inc", "alias_type": "legal_name"}])
    monkeypatch.setattr(poll, "upsert_sentiment", lambda rows: state["upserted"].extend(rows) or len(rows))
    monkeypatch.setattr(
        poll, "resolve_articles", lambda tickers, names: {"AAPL": "Apple Inc.", "MSFT": "Microsoft", "NOPE": None}
    )
    return state


def test_run_upserts_daily_rows_for_resolved_tickers(_db, monkeypatch):
    as_of = date(2026, 10, 14)
    yesterday = as_of - timedelta(days=1)
    session = _FakeSession(lambda u, p: _FakeResponse(200, _items([(yesterday - timedelta(days=1), 5), (yesterday, 7)])))
    monkeypatch.setattr(poll.requests, "Session", lambda: session)

    result = poll.run(as_of)

    assert result == {"tickers": 3, "resolved": 2, "rows_upserted": 4, "aborted": False}
    assert {(r["ticker"], r["date"], r["mentions"]) for r in _db["upserted"]} == {
        ("AAPL", yesterday - timedelta(days=1), 5.0), ("AAPL", yesterday, 7.0),
        ("MSFT", yesterday - timedelta(days=1), 5.0), ("MSFT", yesterday, 7.0),
    }
    assert all(r["source"] == "wikipedia" for r in _db["upserted"])
    assert _db["finished"][-1]["status"] == "ok"
    assert session.calls[0][0].endswith(f"/{(yesterday - timedelta(days=poll._LOOKBACK_DAYS - 1)):%Y%m%d}/{yesterday:%Y%m%d}")


def test_run_aborts_after_consecutive_failures_and_marks_failed(_db, monkeypatch):
    monkeypatch.setattr(poll, "_FAILURE_ABORT_THRESHOLD", 2)
    monkeypatch.setattr(poll.requests, "Session", lambda: _FakeSession(lambda u, p: _FakeResponse(503)))

    result = poll.run(date(2026, 10, 14))

    assert result["aborted"] is True and result["rows_upserted"] == 0
    assert _db["finished"][-1]["status"] == "failed"


def test_run_skips_when_no_tickers(_db, monkeypatch):
    monkeypatch.setattr(poll, "get_watchlist_tickers", lambda: [])
    monkeypatch.setattr(poll, "get_open_positions", lambda: [])

    assert poll.run(date(2026, 10, 14)) == {"tickers": 0, "resolved": 0, "rows_upserted": 0}
    assert _db["finished"][-1]["status"] == "skipped"
