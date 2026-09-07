"""
Real-Postgres integration test for P22Repo's bitemporal restatement path.

Opt-in like the rest of the repo's DB-touching tests: requires
ALEMBIC_DB_URL/ETRADING_TEST_DB_URL, never production. See tests/db/conftest.py.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import text

from src.data.db.repos.repo_p22_biotech_ma import P22Repo


def test_restatement_closes_prior_row_and_inserts_new(db_session) -> None:
    """
    Writing a second value for the same (company, metric) must close the
    first row's valid_to and insert a new row — never UPDATE the value in
    place (spec §2.4, §3.1).
    """
    repo = P22Repo(db_session)
    company_id = repo.upsert_company(cik="0000000001", name="Test Biotech Inc", role="target")

    first_known_from = datetime(2024, 1, 10, tzinfo=timezone.utc)
    repo.upsert_financial_fact_bitemporal(
        company_id=company_id,
        metric="cash",
        value=100_000_000,
        known_from=first_known_from,
        source_id="test_vendor",
        valid_from=date(2024, 1, 10),
    )

    second_known_from = datetime(2024, 4, 15, tzinfo=timezone.utc)
    repo.upsert_financial_fact_bitemporal(
        company_id=company_id,
        metric="cash",
        value=95_000_000,  # a restatement, e.g. a vendor revision
        known_from=second_known_from,
        source_id="test_vendor",
        valid_from=date(2024, 4, 15),
    )

    all_facts = repo.get_financial_facts_as_of(company_id, "cash", as_of_date=date(2024, 12, 31))
    assert len(all_facts) == 2

    open_rows = [f for f in all_facts if f["valid_to"] is None]
    closed_rows = [f for f in all_facts if f["valid_to"] is not None]
    assert len(open_rows) == 1
    assert len(closed_rows) == 1
    assert open_rows[0]["value"] == 95_000_000
    assert closed_rows[0]["value"] == 100_000_000
    assert closed_rows[0]["valid_to"] == date(2024, 4, 15)


def test_lookahead_filter_excludes_future_known_from(db_session) -> None:
    """A fact known_from after as_of_date must not be visible (spec §3.1)."""
    repo = P22Repo(db_session)
    company_id = repo.upsert_company(cik="0000000002", name="Future Fact Inc", role="target")

    repo.upsert_financial_fact_bitemporal(
        company_id=company_id,
        metric="cash",
        value=50_000_000,
        known_from=datetime(2024, 6, 1, tzinfo=timezone.utc),
        source_id="test_vendor",
        valid_from=date(2024, 6, 1),
    )

    # A backtest as of before known_from must not see this fact.
    visible = repo.get_financial_facts_as_of(company_id, "cash", as_of_date=date(2024, 5, 1))
    assert visible == []

    visible_after = repo.get_financial_facts_as_of(company_id, "cash", as_of_date=date(2024, 6, 2))
    assert len(visible_after) == 1


def test_lookahead_filter_uses_utc_not_db_session_timezone(db_session) -> None:
    """
    Regression test (fixed 2026-09-05): `get_financial_facts_as_of`'s as-of
    upper bound must be built with `tzinfo=timezone.utc` explicitly, not a
    naive `datetime.combine(as_of_date, datetime.max.time())` — `known_from`
    is `TIMESTAMPTZ` and always written UTC-aware, so a naive bound would be
    interpreted in the DB session's own timezone instead. Proven here by
    actually setting the session's timezone away from UTC: if the bound were
    naive, `2024-01-01 23:59:59.999999` would be read as 23:59:59.999999
    America/New_York (== 2024-01-02 04:59:59.999999 UTC), wrongly admitting a
    fact known at 02:00 UTC on Jan 2 as "known as of 2024-01-01."
    """
    db_session.execute(text("SET TIME ZONE 'America/New_York'"))

    repo = P22Repo(db_session)
    company_id = repo.upsert_company(cik="0000000029", name="TZ Guard Inc", role="target")

    repo.upsert_financial_fact_bitemporal(
        company_id=company_id,
        metric="cash",
        value=1.0,
        known_from=datetime(2024, 1, 2, 2, 0, tzinfo=timezone.utc),
        source_id="test_vendor",
        valid_from=date(2024, 1, 2),
    )

    visible = repo.get_financial_facts_as_of(company_id, "cash", as_of_date=date(2024, 1, 1))
    assert visible == []


def test_price_daily_upsert_never_rewrites_existing_row(db_session) -> None:
    """
    Raw prices are "as traded, never rewritten" (spec §2.0.7) — a second
    upsert for the same (company, trade_date, vendor) must be a no-op, not
    an overwrite, even if the caller passes a different value.
    """
    repo = P22Repo(db_session)
    company_id = repo.upsert_company(cik="0000000003", name="Price Archive Inc", role="target")

    repo.upsert_price_daily(company_id=company_id, trade_date=date(2020, 3, 2), vendor="ibkr", close_raw=10.0)
    repo.upsert_price_daily(company_id=company_id, trade_date=date(2020, 3, 2), vendor="ibkr", close_raw=999.0)

    assert repo.get_raw_close(company_id, date(2020, 3, 2), vendor="ibkr") == 10.0


def test_get_adjusted_close_applies_only_actions_known_by_as_of(db_session) -> None:
    """Real-DB round trip of the spec §2.0.7 lookahead guard through the repo layer."""
    repo = P22Repo(db_session)
    company_id = repo.upsert_company(cik="0000000004", name="Reverse Split Inc", role="target")

    repo.upsert_price_daily(company_id=company_id, trade_date=date(2019, 6, 1), vendor="ibkr", close_raw=100.0)
    repo.upsert_corporate_action(
        company_id=company_id,
        ex_date=date(2023, 3, 1),
        action_type="reverse_split",
        ratio=0.05,  # 1-for-20
        source="sec_8k",
        known_from=datetime(2023, 3, 1, tzinfo=timezone.utc),
    )

    # Before the split existed at all: unadjusted.
    assert repo.get_adjusted_close(company_id, date(2019, 6, 1), as_of=date(2020, 1, 1)) == 100.0
    # After the split is known: adjusted.
    assert repo.get_adjusted_close(company_id, date(2019, 6, 1), as_of=date(2023, 6, 1)) == 100.0 / 0.05


def test_get_latest_raw_close_as_of_returns_most_recent_qualifying_row(db_session) -> None:
    """`ingest/market_cap.py`'s price leg: the most recent RAW close on or before `as_of`, known
    on or before `as_of` — never the split-adjusted `get_adjusted_close` (spec §2.0.7's raw-on-raw
    pairing requirement for market_cap)."""
    repo = P22Repo(db_session)
    company_id = repo.upsert_company(cik="0000000030", name="Market Cap Inc", role="target")

    repo.upsert_price_daily(
        company_id=company_id, trade_date=date(2026, 9, 2), vendor="yfinance", close_raw=10.0,
        known_from=datetime(2026, 9, 2, 21, tzinfo=timezone.utc),
    )
    repo.upsert_price_daily(
        company_id=company_id, trade_date=date(2026, 9, 3), vendor="yfinance", close_raw=11.0,
        known_from=datetime(2026, 9, 3, 21, tzinfo=timezone.utc),
    )

    latest = repo.get_latest_raw_close_as_of(company_id, date(2026, 9, 5))
    assert latest == {"trade_date": date(2026, 9, 3), "close_raw": 11.0, "vendor": "yfinance"}

    # A row not yet known by as_of must not be visible, even though its trade_date qualifies.
    as_of_before_known = repo.get_latest_raw_close_as_of(company_id, date(2026, 9, 2))
    assert as_of_before_known == {"trade_date": date(2026, 9, 2), "close_raw": 10.0, "vendor": "yfinance"}

    none_yet = repo.get_latest_raw_close_as_of(company_id, date(2026, 9, 1))
    assert none_yet is None


def test_get_latest_raw_close_as_of_ignores_null_known_from(db_session) -> None:
    """A row with no `known_from` on file is treated as not-yet-known, not always-visible —
    mirroring `get_adjusted_close`'s `known_from_date=date.max` convention for the same gap."""
    repo = P22Repo(db_session)
    company_id = repo.upsert_company(cik="0000000031", name="No Known From Inc", role="target")

    repo.upsert_price_daily(company_id=company_id, trade_date=date(2026, 9, 2), vendor="yfinance", close_raw=10.0)

    assert repo.get_latest_raw_close_as_of(company_id, date(2026, 9, 5)) is None


def test_list_companies_returns_id_to_name_map(db_session) -> None:
    """`list_companies` is the match target `alias_matching.resolve_aliases` reads (spec §3.3)."""
    repo = P22Repo(db_session)
    id1 = repo.upsert_company(cik="0000000005", name="Acme Therapeutics Inc", role="target")
    id2 = repo.upsert_company(cik="0000000006", name="Beta Pharmaceuticals Corp", role="target")

    companies = repo.list_companies()

    assert companies[id1] == "Acme Therapeutics Inc"
    assert companies[id2] == "Beta Pharmaceuticals Corp"


def test_review_item_confirm_round_trip_through_repo(db_session) -> None:
    """Real-DB round trip of add_review_item -> get_pending_review_items -> resolve_review_item,
    including the created_at column added in migration 005 (spec §3.4)."""
    repo = P22Repo(db_session)

    item_id = repo.add_review_item(
        item_type="entity_match",
        payload={"reason": "spac_name_heuristic", "cik": "0000000007", "name": "Maybe A SPAC Inc"},
        priority=1,
    )

    pending = repo.get_pending_review_items(item_type="entity_match")
    assert any(i["item_id"] == item_id for i in pending)
    matched = next(i for i in pending if i["item_id"] == item_id)
    assert matched["created_at"] is not None

    repo.resolve_review_item(item_id=item_id, status="confirmed", reviewed_by="alex", note="looks right")

    resolved = repo.get_review_item(item_id)
    assert resolved is not None
    assert resolved["status"] == "confirmed"
    assert resolved["reviewed_by"] == "alex"
    assert resolved["note"] == "looks right"
    assert resolved["reviewed_at"] is not None

    still_pending = repo.get_pending_review_items(item_type="entity_match")
    assert not any(i["item_id"] == item_id for i in still_pending)


def test_get_companies_without_verified_alias(db_session) -> None:
    """spec §8.2: 'No company row without at least one verified alias' — the query feeding
    features/quality.assert_every_company_has_a_verified_alias."""
    repo = P22Repo(db_session)

    aliased_id = repo.upsert_company(cik="0000000010", name="Aliased Biotech Inc", role="target")
    unaliased_id = repo.upsert_company(cik="0000000011", name="Unaliased Biotech Inc", role="target")
    unverified_only_id = repo.upsert_company(cik="0000000012", name="Unverified Only Inc", role="target")

    repo.add_company_alias(company_id=aliased_id, alias="Aliased Biotech", source="clinicaltrials", is_verified=True)
    repo.add_company_alias(
        company_id=unverified_only_id, alias="Unverified Only", source="clinicaltrials", is_verified=False
    )

    missing = repo.get_companies_without_verified_alias()

    assert aliased_id not in missing
    assert unaliased_id in missing
    assert unverified_only_id in missing


def test_upsert_trial_round_trip_and_second_call_updates_in_place(db_session) -> None:
    """`upsert_trial` is keyed on nct_id — a second call for the same trial (e.g. a re-fetch
    after a status change) must update the existing row, not insert a duplicate (spec §3.2)."""
    repo = P22Repo(db_session)

    repo.upsert_trial(
        nct_id="NCT05668741",
        phase="PHASE1/PHASE2",
        status="RECRUITING",
        enrollment=26,
        primary_completion_date=date(2026, 4, 21),
        is_randomized=None,
        countries=["United States"],
        known_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    trial = repo.get_trial("NCT05668741")
    assert trial is not None
    assert trial["status"] == "RECRUITING"
    assert trial["asset_id"] is None
    assert trial["countries"] == ["United States"]

    # Re-fetch later: status changed, no new row.
    repo.upsert_trial(
        nct_id="NCT05668741",
        phase="PHASE1/PHASE2",
        status="ACTIVE_NOT_RECRUITING",
        enrollment=26,
        primary_completion_date=date(2026, 4, 21),
        countries=["United States"],
        known_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    updated = repo.get_trial("NCT05668741")
    assert updated is not None
    assert updated["status"] == "ACTIVE_NOT_RECRUITING"


def test_upsert_acquirer_company_creates_new_row_when_no_ticker_match(db_session) -> None:
    """A brand-new acquirer (not already in p22_company under any identity) gets a fresh
    cik=None, role='acquirer' row (spec §2.0.4)."""
    repo = P22Repo(db_session)

    company_id = repo.upsert_acquirer_company(name="Acme Pharma Inc", ticker="ACME")

    companies = repo.list_companies()
    assert companies[company_id] == "Acme Pharma Inc"


def test_upsert_acquirer_company_merges_into_existing_target_row_by_ticker(db_session) -> None:
    """An acquirer that's already a resolved DERA target (real cik) must be merged into
    (role -> 'both'), never duplicated into a second cik-less identity."""
    repo = P22Repo(db_session)
    target_id = repo.upsert_company(cik="0000000099", name="Dual Role Biotech Inc", ticker="DUAL", role="target")

    merged_id = repo.upsert_acquirer_company(name="Dual Role Biotech Inc", ticker="DUAL")

    assert merged_id == target_id  # same identity, not a new row
    merged_company = repo.get_company_by_cik("0000000099")
    assert merged_company is not None
    assert merged_company["role"] == "both"


def test_upsert_acquirer_company_finds_existing_row_by_cik_when_ticker_differs(db_session) -> None:
    """Real bug, live-caught 2026-08-31: a DERA-resolved row can have a stale/unrelated ticker on
    file for its CIK (e.g. SEC's snapshot mapped Bristol-Myers Squibb's CIK to "CELG-RI", a leftover
    Celgene CVR security ticker, not "BMY"). Looking up by ticker alone misses the existing row and
    tries to INSERT a duplicate with the same (unique) cik, which crashes. Must find it by cik first."""
    repo = P22Repo(db_session)
    target_id = repo.upsert_company(cik="0000014272", name="BRISTOL MYERS SQUIBB CO", ticker="CELG-RI", role="target")

    merged_id = repo.upsert_acquirer_company(name="Bristol-Myers Squibb Company", ticker="BMY", cik="0000014272")

    assert merged_id == target_id  # same identity, not a crash or a duplicate
    merged_company = repo.get_company_by_cik("0000014272")
    assert merged_company is not None
    assert merged_company["role"] == "both"
    # Curated config values win over the stale snapshot ticker/name for an identity match found by cik.
    assert merged_company["ticker"] == "BMY"
    assert merged_company["name"] == "Bristol-Myers Squibb Company"


def test_upsert_acquirer_company_is_idempotent_on_repeated_calls(db_session) -> None:
    """Re-running the loader job (e.g. every quarterly schedule tick) must not create duplicates."""
    repo = P22Repo(db_session)

    first_id = repo.upsert_acquirer_company(name="Repeat Pharma Inc", ticker="RPT")
    second_id = repo.upsert_acquirer_company(name="Repeat Pharma Inc", ticker="RPT")

    assert first_id == second_id


def test_upsert_asset_and_get_by_company_and_name_round_trip(db_session) -> None:
    repo = P22Repo(db_session)
    company_id = repo.upsert_company(cik="0000000020", name="Asset Owner Inc", role="target")

    asset_id = repo.upsert_asset(
        company_id=company_id,
        name="VX-522 mRNA therapy",
        therapeutic_area="respiratory",
        indication="Cystic Fibrosis",
    )

    found = repo.get_asset_by_company_and_name(company_id, "VX-522 mRNA therapy")
    assert found is not None
    assert found["asset_id"] == asset_id
    assert found["therapeutic_area"] == "respiratory"
    assert found["modality"] is None

    assert repo.get_asset_by_company_and_name(company_id, "Some Other Drug") is None


def test_upsert_patent_expiry_is_idempotent_on_repeated_calls(db_session) -> None:
    """Re-processing the same landed Orange Book snapshot must not create duplicate rows
    (spec §2.3/§3.2 gives patent_expiry no natural unique key — enforced app-side)."""
    repo = P22Repo(db_session)
    acquirer_id = repo.upsert_acquirer_company(name="Patent Holder Inc", ticker="PHLD")

    first_id = repo.upsert_patent_expiry(
        acquirer_id=acquirer_id,
        application_no="205552",
        loe_date=date(2031, 1, 5),
        source="orange_book",
        product_name="IMBRUVICA",
        exclusivity_type="patent",
    )
    second_id = repo.upsert_patent_expiry(
        acquirer_id=acquirer_id,
        application_no="205552",
        loe_date=date(2031, 1, 5),
        source="orange_book",
        product_name="IMBRUVICA",
        exclusivity_type="patent",
    )

    assert first_id == second_id
    rows = repo.get_patent_expiries_for_acquirer(acquirer_id)
    assert len(rows) == 1
    assert rows[0]["therapeutic_area"] is None
    assert rows[0]["ttm_revenue_usd"] is None
