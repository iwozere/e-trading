"""
Repository layer for all p22_* tables (P22 Biotech M&A pipeline).

Accepts a SQLAlchemy Session in __init__ — never opens its own session.
All writes call self.session.flush() to materialise generated keys/counts
within the current Unit-of-Work; the surrounding service commits on success.

`upsert_financial_fact_bitemporal` is the one generic restatement-safe write
path (spec §2.4, §3.1): it closes the prior open row's `valid_to` and inserts
a new row, and is never bypassed with an in-place `UPDATE ... SET value` —
see docs/Design.md for why that invariant matters.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.db.models.model_p22_biotech_ma import (
    P22Asset,
    P22Company,
    P22CompanyAlias,
    P22CorporateAction,
    P22FetchFailure,
    P22FinancialFact,
    P22PatentExpiry,
    P22PriceDaily,
    P22ReviewItem,
    P22Trial,
)

# `ingest/price_archive.py`'s pure adjustment math is imported lazily, inside
# `get_adjusted_close` below, rather than at module scope — mirroring the
# existing precedent for src/data/db -> src/ml/pipeline references
# (model_short_squeeze.py), which keeps the DB layer importable on its own
# and avoids a module-load-order dependency on a specific pipeline package.


class P22Repo:
    """All repository operations for p22_* tables, bound to a single Session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Company
    # ------------------------------------------------------------------

    def upsert_company(
        self,
        *,
        cik: Optional[str],
        name: str,
        ticker: Optional[str] = None,
        exchange: Optional[str] = None,
        sic_code: Optional[str] = None,
        is_active: Optional[bool] = None,
        delisted_date: Optional[date] = None,
        role: Optional[str] = None,
    ) -> int:
        """
        Upsert a company keyed on `cik` (spec §2.0's universe is CIK-keyed).
        Returns the company_id.
        """
        values: Dict[str, Any] = {
            "cik": cik,
            "name": name,
            "ticker": ticker,
            "exchange": exchange,
            "sic_code": sic_code,
            "is_active": is_active,
            "delisted_date": delisted_date,
            "role": role,
        }
        insert_stmt = pg_insert(P22Company).values(**values)
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["cik"],
            set_={k: insert_stmt.excluded[k] for k in values if k != "cik"},
        ).returning(P22Company.company_id)
        result = self.session.execute(upsert_stmt)
        self.session.flush()
        return result.scalar_one()

    def get_company_by_cik(self, cik: str) -> Optional[Dict[str, Any]]:
        row = self.session.execute(select(P22Company).where(P22Company.cik == cik)).scalars().first()
        if row is None:
            return None
        return {c.key: getattr(row, c.key) for c in P22Company.__table__.columns}

    def list_companies(self) -> Dict[int, str]:
        """`company_id -> name` for the whole resolved roster — the match target for alias resolution."""
        rows = self.session.execute(select(P22Company.company_id, P22Company.name)).all()
        return {row.company_id: row.name for row in rows}

    def upsert_acquirer_company(
        self, *, name: str, ticker: str, cik: Optional[str] = None
    ) -> int:
        """
        Upsert one acquirer-roster company (spec §2.0.4). If `cik` is known,
        this delegates to the stronger `upsert_company` (cik-keyed upsert).
        Otherwise — the common case today, since `p22_acquirers.yaml`'s CIKs
        are all unverified/null (see docs/Tasks.md "Decisions needed" item 3)
        — it merges on `ticker` instead, because `p22_company.ticker` has no
        DB-level uniqueness to upsert against via `ON CONFLICT`.

        Critically, this looks up by `ticker` regardless of whether the
        matching row already has a `cik` — an acquirer that's also SIC-coded
        into the DERA target universe (large-cap pharma routinely is) may
        already exist as a `role='target'` row with a real `cik`. Without
        this check, a naive ticker-less insert would create a second,
        duplicate identity for the same real-world company. When a match is
        found, `role` is merged (`target` + `acquirer` -> `both`) rather than
        overwritten, and the `cik`-bearing row is kept intact.
        """
        existing = self.session.execute(select(P22Company).where(P22Company.ticker == ticker)).scalars().first()

        if existing is not None:
            new_role = "both" if existing.role in ("target", "both") else "acquirer"
            self.session.execute(
                update(P22Company).where(P22Company.company_id == existing.company_id).values(role=new_role)
            )
            self.session.flush()
            return existing.company_id

        row = P22Company(cik=cik, name=name, ticker=ticker, role="acquirer")
        self.session.add(row)
        self.session.flush()
        return row.company_id

    def list_acquirer_companies(self) -> Dict[int, str]:
        """`company_id -> name` for every company with `role in ('acquirer','both')` — the match
        target for resolving Orange Book `Applicant_Full_Name` strings (spec §2.3, §4.1)."""
        rows = self.session.execute(
            select(P22Company.company_id, P22Company.name).where(P22Company.role.in_(("acquirer", "both")))
        ).all()
        return {row.company_id: row.name for row in rows}

    def get_companies_without_verified_alias(self) -> List[int]:
        """
        `company_id`s with no `p22_company_alias` row where `is_verified = TRUE` — spec §8.2's
        data-quality invariant ("No `company` row without at least one verified alias"), consumed
        by `features/quality.assert_every_company_has_a_verified_alias`.
        """
        verified_ids = select(P22CompanyAlias.company_id).where(P22CompanyAlias.is_verified.is_(True)).distinct()
        rows = self.session.execute(
            select(P22Company.company_id).where(P22Company.company_id.not_in(verified_ids))
        ).all()
        return [row.company_id for row in rows]

    # ------------------------------------------------------------------
    # Company alias
    # ------------------------------------------------------------------

    def add_company_alias(
        self,
        *,
        company_id: int,
        alias: str,
        source: str,
        is_verified: bool = False,
        known_from: Optional[datetime] = None,
    ) -> None:
        """
        Upsert an alias (company_id, alias, source). `known_from` must be the
        underlying filing/document date, never the review date (spec §3.4).
        """
        stmt = pg_insert(P22CompanyAlias).values(
            company_id=company_id,
            alias=alias,
            source=source,
            is_verified=is_verified,
            known_from=known_from,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["company_id", "alias", "source"],
            set_={"is_verified": stmt.excluded.is_verified, "known_from": stmt.excluded.known_from},
        )
        self.session.execute(stmt)
        self.session.flush()

    # ------------------------------------------------------------------
    # Financial fact — bitemporal restatement-safe write (spec §2.4, §3.1)
    # ------------------------------------------------------------------

    def upsert_financial_fact_bitemporal(
        self,
        *,
        company_id: int,
        metric: str,
        value: Optional[float],
        known_from: datetime,
        source_id: str,
        unit: str = "USD",
        period_end: Optional[date] = None,
        valid_from: Optional[date] = None,
        source_url: Optional[str] = None,
    ) -> int:
        """
        Write a financial fact, honoring the bitemporal restatement rule:
        never UPDATE an existing row's value in place. If an open row
        (`valid_to IS NULL`) exists for this `(company_id, metric)`, its
        `valid_to` is closed to `valid_from` of the new row (or `known_from`'s
        date if `valid_from` is not given) and a new row is inserted.

        This is the ONLY code path that should write p22_financial_fact.
        Every other call site — vendor ingest, SEC XBRL ingest, anything
        added later — must funnel through this method, not hand-roll the
        close-and-insert logic itself (spec §3.1 is the single most
        important correctness property in the system; one implementation
        of the invariant, not several slightly different ones).

        Returns:
            The new fact_id.
        """
        new_valid_from = valid_from or known_from.date()

        open_row = self.session.execute(
            select(P22FinancialFact).where(
                P22FinancialFact.company_id == company_id,
                P22FinancialFact.metric == metric,
                P22FinancialFact.valid_to.is_(None),
            )
        ).scalars().first()

        if open_row is not None:
            self.session.execute(
                update(P22FinancialFact)
                .where(P22FinancialFact.fact_id == open_row.fact_id)
                .values(valid_to=new_valid_from)
            )

        new_row = P22FinancialFact(
            company_id=company_id,
            metric=metric,
            value=value,
            unit=unit,
            period_end=period_end,
            valid_from=new_valid_from,
            valid_to=None,
            known_from=known_from,
            source_id=source_id,
            source_url=source_url,
        )
        self.session.add(new_row)
        self.session.flush()
        return new_row.fact_id

    def get_financial_facts_as_of(
        self, company_id: int, metric: str, as_of_date: date
    ) -> List[Dict[str, Any]]:
        """
        Every fact for (company_id, metric) known as of `as_of_date` — i.e.
        `known_from <= as_of_date`. Backtests must never see a fact whose
        `known_from` is after the as-of date (spec §3.1).
        """
        rows = self.session.execute(
            select(P22FinancialFact)
            .where(
                P22FinancialFact.company_id == company_id,
                P22FinancialFact.metric == metric,
                P22FinancialFact.known_from <= datetime.combine(as_of_date, datetime.max.time()),
            )
            .order_by(P22FinancialFact.known_from.desc())
        ).scalars().all()
        return [{c.key: getattr(r, c.key) for c in P22FinancialFact.__table__.columns} for r in rows]

    # ------------------------------------------------------------------
    # Asset (spec §3.2, §4.2 Block B)
    # ------------------------------------------------------------------

    def upsert_asset(
        self,
        *,
        company_id: int,
        name: str,
        therapeutic_area: str,
        modality: Optional[str] = None,
        target_protein: Optional[str] = None,
        indication: Optional[str] = None,
        is_lead: Optional[bool] = None,
    ) -> int:
        """
        Insert one `p22_asset` row. Callers (`ingest/asset_normalization.py`)
        are responsible for checking `get_asset_by_company_and_name` first —
        this method itself does not dedupe, mirroring `upsert_trial`'s
        contract of doing exactly the DB operation its name says.
        """
        row = P22Asset(
            company_id=company_id,
            name=name,
            modality=modality,
            target_protein=target_protein,
            therapeutic_area=therapeutic_area,
            indication=indication,
            is_lead=is_lead,
        )
        self.session.add(row)
        self.session.flush()
        return row.asset_id

    def get_asset_by_company_and_name(self, company_id: int, name: str) -> Optional[Dict[str, Any]]:
        row = self.session.execute(
            select(P22Asset).where(P22Asset.company_id == company_id, P22Asset.name == name)
        ).scalars().first()
        if row is None:
            return None
        return {c.key: getattr(row, c.key) for c in P22Asset.__table__.columns}

    # ------------------------------------------------------------------
    # Trial (spec §2.2, §3.2)
    # ------------------------------------------------------------------

    def upsert_trial(
        self,
        *,
        nct_id: str,
        asset_id: Optional[int] = None,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        enrollment: Optional[int] = None,
        primary_completion_date: Optional[date] = None,
        uses_biomarker_selection: Optional[bool] = None,
        is_randomized: Optional[bool] = None,
        has_active_comparator: Optional[bool] = None,
        primary_endpoint_text: Optional[str] = None,
        endpoint_changed_midtrial: Optional[bool] = None,
        countries: Optional[List[str]] = None,
        known_from: Optional[datetime] = None,
    ) -> None:
        """
        Upsert a trial keyed on `nct_id` (spec §3.2's `p22_trial` primary key).
        Unlike `p22_financial_fact`, this is a plain upsert, not a bitemporal
        restatement chain — CT.gov re-fetches naturally overwrite a trial's
        latest known state in place (a trial's status/enrollment/phase evolve
        over its life and there is one current record per NCT ID, not a
        history of prior states to preserve here; the actual change-over-time
        signal spec §2.2 wants lives in the separate version-history endpoint,
        not in this table's row history).
        """
        values: Dict[str, Any] = {
            "asset_id": asset_id,
            "phase": phase,
            "status": status,
            "enrollment": enrollment,
            "primary_completion_date": primary_completion_date,
            "uses_biomarker_selection": uses_biomarker_selection,
            "is_randomized": is_randomized,
            "has_active_comparator": has_active_comparator,
            "primary_endpoint_text": primary_endpoint_text,
            "endpoint_changed_midtrial": endpoint_changed_midtrial,
            "countries": countries,
            "known_from": known_from,
        }
        insert_stmt = pg_insert(P22Trial).values(nct_id=nct_id, **values)
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["nct_id"],
            set_={k: insert_stmt.excluded[k] for k in values},
        )
        self.session.execute(upsert_stmt)
        self.session.flush()

    def get_trial(self, nct_id: str) -> Optional[Dict[str, Any]]:
        row = self.session.execute(select(P22Trial).where(P22Trial.nct_id == nct_id)).scalars().first()
        if row is None:
            return None
        return {c.key: getattr(row, c.key) for c in P22Trial.__table__.columns}

    # ------------------------------------------------------------------
    # Patent expiry — acquirer side (spec §2.3, §4.1 Block A)
    # ------------------------------------------------------------------

    def upsert_patent_expiry(
        self,
        *,
        acquirer_id: int,
        application_no: str,
        loe_date: date,
        source: str,
        product_name: Optional[str] = None,
        therapeutic_area: Optional[str] = None,
        ttm_revenue_usd: Optional[float] = None,
        exclusivity_type: Optional[str] = None,
    ) -> int:
        """
        Insert one patent/exclusivity expiry row, idempotently. There's no
        natural DB-level unique key here — the spec's own §3.2 SQL sketch
        gives `patent_expiry` a bare serial primary key — so idempotency is
        enforced at the application level on `(acquirer_id, application_no,
        loe_date, source)`: re-processing the same landed Orange Book
        snapshot (e.g. a re-run before the next quarterly refresh) must not
        keep inserting duplicate rows. A patent's expiry date, once filed,
        doesn't change, so "already have this row" is treated as a true
        no-op, not a candidate for update.

        Returns:
            The (new or pre-existing) `patent_expiry_id`.
        """
        existing = self.session.execute(
            select(P22PatentExpiry.patent_expiry_id).where(
                P22PatentExpiry.acquirer_id == acquirer_id,
                P22PatentExpiry.application_no == application_no,
                P22PatentExpiry.loe_date == loe_date,
                P22PatentExpiry.source == source,
            )
        ).scalars().first()
        if existing is not None:
            return existing

        row = P22PatentExpiry(
            acquirer_id=acquirer_id,
            product_name=product_name,
            application_no=application_no,
            therapeutic_area=therapeutic_area,
            loe_date=loe_date,
            ttm_revenue_usd=ttm_revenue_usd,
            exclusivity_type=exclusivity_type,
            source=source,
        )
        self.session.add(row)
        self.session.flush()
        return row.patent_expiry_id

    def get_patent_expiries_for_acquirer(self, acquirer_id: int) -> List[Dict[str, Any]]:
        rows = self.session.execute(
            select(P22PatentExpiry).where(P22PatentExpiry.acquirer_id == acquirer_id)
        ).scalars().all()
        return [{c.key: getattr(r, c.key) for c in P22PatentExpiry.__table__.columns} for r in rows]

    # ------------------------------------------------------------------
    # Review queue (spec §3.4)
    # ------------------------------------------------------------------

    def add_review_item(
        self,
        *,
        item_type: str,
        payload: Dict[str, Any],
        evidence_url: Optional[str] = None,
        priority: int = 0,
    ) -> int:
        row = P22ReviewItem(
            item_type=item_type,
            payload=payload,
            evidence_url=evidence_url,
            priority=priority,
            status="pending",
        )
        self.session.add(row)
        self.session.flush()
        return row.item_id

    def get_pending_review_items(self, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        stmt = select(P22ReviewItem).where(P22ReviewItem.status == "pending")
        if item_type:
            stmt = stmt.where(P22ReviewItem.item_type == item_type)
        stmt = stmt.order_by(P22ReviewItem.priority.desc(), P22ReviewItem.item_id.asc())
        rows = self.session.execute(stmt).scalars().all()
        return [{c.key: getattr(r, c.key) for c in P22ReviewItem.__table__.columns} for r in rows]

    def get_review_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        row = self.session.execute(select(P22ReviewItem).where(P22ReviewItem.item_id == item_id)).scalars().first()
        if row is None:
            return None
        return {c.key: getattr(row, c.key) for c in P22ReviewItem.__table__.columns}

    def resolve_review_item(
        self,
        *,
        item_id: int,
        status: str,
        reviewed_by: str,
        note: Optional[str] = None,
    ) -> None:
        """
        Mark a review item confirmed/rejected/needs_info. `reviewed_at` is the
        actual review time (this is a plain audit timestamp, not a bitemporal
        fact-validity one) — the underlying-data timestamp that matters for
        any downstream write lives in `payload['known_from']` and is the
        caller's job to apply, not this method's (spec §3.4).
        """
        self.session.execute(
            update(P22ReviewItem)
            .where(P22ReviewItem.item_id == item_id)
            .values(status=status, reviewed_by=reviewed_by, reviewed_at=datetime.now(timezone.utc), note=note)
        )
        self.session.flush()

    # ------------------------------------------------------------------
    # Fetch failures (spec §7.2)
    # ------------------------------------------------------------------

    def log_fetch_failure(
        self,
        *,
        source: str,
        entity: Optional[str] = None,
        url: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> int:
        row = P22FetchFailure(source=source, entity=entity, url=url, error_message=error_message)
        self.session.add(row)
        self.session.flush()
        return row.failure_id

    def get_unresolved_fetch_failures(self) -> List[Dict[str, Any]]:
        rows = self.session.execute(
            select(P22FetchFailure).where(P22FetchFailure.resolved.is_(False))
        ).scalars().all()
        return [{c.key: getattr(r, c.key) for c in P22FetchFailure.__table__.columns} for r in rows]

    # ------------------------------------------------------------------
    # Price archive and corporate actions (spec §2.0.7, added v0.6)
    # ------------------------------------------------------------------

    def upsert_price_daily(
        self,
        *,
        company_id: int,
        trade_date: date,
        vendor: str,
        open_raw: Optional[float] = None,
        high_raw: Optional[float] = None,
        low_raw: Optional[float] = None,
        close_raw: Optional[float] = None,
        volume_raw: Optional[int] = None,
        known_from: Optional[datetime] = None,
    ) -> None:
        """
        Insert one RAW daily OHLCV row. On conflict with an existing
        `(company_id, trade_date, vendor)` row, do nothing — spec §2.0.7 is
        explicit that this table is "as traded, never rewritten." If a
        vendor genuinely needs to correct a bad print, that is a manual
        decision, not something an ingest re-run should do silently.
        """
        stmt = pg_insert(P22PriceDaily).values(
            company_id=company_id,
            trade_date=trade_date,
            vendor=vendor,
            open_raw=open_raw,
            high_raw=high_raw,
            low_raw=low_raw,
            close_raw=close_raw,
            volume_raw=volume_raw,
            known_from=known_from,
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=["company_id", "trade_date", "vendor"])
        self.session.execute(stmt)
        self.session.flush()

    def upsert_corporate_action(
        self,
        *,
        company_id: int,
        ex_date: date,
        action_type: str,
        source: str,
        ratio: Optional[float] = None,
        cash_amount: Optional[float] = None,
        new_ticker: Optional[str] = None,
        is_verified: bool = False,
        known_from: Optional[datetime] = None,
        source_url: Optional[str] = None,
    ) -> None:
        """
        Upsert a corporate action keyed on `(company_id, ex_date, action_type)`.
        Unlike price rows, these ARE allowed to be revised in place — e.g. an
        `is_verified` flip once an SEC-filing reconciliation confirms a
        vendor-sourced split (spec §2.0.7's sourcing precedence).
        """
        values: Dict[str, Any] = {
            "company_id": company_id,
            "ex_date": ex_date,
            "action_type": action_type,
            "ratio": ratio,
            "cash_amount": cash_amount,
            "new_ticker": new_ticker,
            "source": source,
            "is_verified": is_verified,
            "known_from": known_from,
            "source_url": source_url,
        }
        insert_stmt = pg_insert(P22CorporateAction).values(**values)
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["company_id", "ex_date", "action_type"],
            set_={k: insert_stmt.excluded[k] for k in values if k not in ("company_id", "ex_date", "action_type")},
        )
        self.session.execute(upsert_stmt)
        self.session.flush()

    def get_raw_close(
        self, company_id: int, trade_date: date, vendor: Optional[str] = None
    ) -> Optional[float]:
        """Raw close for one `(company_id, trade_date)`, optionally pinned to one vendor."""
        stmt = select(P22PriceDaily.close_raw).where(
            P22PriceDaily.company_id == company_id,
            P22PriceDaily.trade_date == trade_date,
        )
        if vendor is not None:
            stmt = stmt.where(P22PriceDaily.vendor == vendor)
        row = self.session.execute(stmt.limit(1)).scalars().first()
        return float(row) if row is not None else None

    def get_adjusted_close(
        self,
        company_id: int,
        trade_date: date,
        as_of: date,
        vendor: Optional[str] = None,
    ) -> Optional[float]:
        """
        Split-adjusted close for `(company_id, trade_date)` as of `as_of`
        (spec §2.0.7). Combines the raw price lookup with every corporate
        action for the company; `ingest/price_archive.adjusted_close` applies
        the `known_from <= as_of` lookahead guard and the split-factor math.
        `None` if no raw price is on file — never a synthetic zero.
        """
        from src.ml.pipeline.p22_biotech_ma.ingest.price_archive import (
            CorporateActionRatio,
            adjusted_close,
        )

        raw_close = self.get_raw_close(company_id, trade_date, vendor=vendor)
        if raw_close is None:
            return None

        action_rows = self.session.execute(
            select(P22CorporateAction).where(P22CorporateAction.company_id == company_id)
        ).scalars().all()
        actions = [
            CorporateActionRatio(
                ex_date=a.ex_date,
                action_type=a.action_type,
                ratio=float(a.ratio) if a.ratio is not None else None,
                # A missing known_from means we don't actually know when the
                # pipeline learned this fact — treat it as "not yet known"
                # (date.max) rather than defaulting to ex_date, which would
                # optimistically assume same-day knowledge and risk exactly
                # the lookahead leak this table exists to prevent.
                known_from_date=a.known_from.date() if a.known_from is not None else date.max,
            )
            for a in action_rows
        ]
        return adjusted_close(raw_close, actions, trade_date, as_of)
