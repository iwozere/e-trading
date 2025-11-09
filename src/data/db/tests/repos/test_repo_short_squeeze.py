from __future__ import annotations
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from src.data.db.repos.repo_short_squeeze import (
    ScreenerSnapshotRepo, DeepScanMetricsRepo, SqueezeAlertRepo,
    AdHocCandidateRepo, FINRAShortInterestRepo
)
from src.data.db.models.model_short_squeeze import AlertLevel


def test_screener_and_metrics_and_alerts(db_session: Session):
    screener = ScreenerSnapshotRepo(db_session)
    metrics = DeepScanMetricsRepo(db_session)
    alerts = SqueezeAlertRepo(db_session)

    run_d = date.today()
    screener.create_snapshot({"ticker": "AAA", "run_date": run_d, "screener_score": 10})
    top = screener.get_top_candidates(run_d, limit=5)
    assert len(top) >= 1 and top[0].ticker == "AAA"

    # metrics upsert + latest
    today = date.today()
    metrics.upsert_metrics({"ticker": "AAA", "date": today, "squeeze_score": 0.5})
    latest = metrics.get_latest_metrics("AAA")
    assert latest is not None

    # alerts create + cooldown check
    now = datetime.now()
    a = alerts.create_alert({
        "ticker": "AAA",
        "alert_level": AlertLevel.HIGH.value,
        "timestamp": now,
        "cooldown_expires": now + timedelta(minutes=10),
        "sent": True,
    })
    assert a.id is not None
    assert alerts.check_cooldown("AAA", AlertLevel.HIGH) is True

    # cleanup expired returns count (none yet)
    assert alerts.cleanup_expired_cooldowns() >= 0


def test_adhoc_and_finra(db_session: Session):
    adhoc = AdHocCandidateRepo(db_session)
    finra = FINRAShortInterestRepo(db_session)

    cand = adhoc.add_candidate("bbb", reason="user-request")
    assert cand.ticker == "BBB"
    assert adhoc.get_candidate("bbb") is not None
    assert any(x.ticker == "BBB" for x in adhoc.get_active_candidates())
    assert adhoc.deactivate_candidate("bbb") is True

    # FINRA upsert and latest
    sd = date.today()
    rec = finra.upsert_finra_data({
        "ticker": "CCC",
        "settlement_date": sd,
        "short_interest_shares": 1000,
        "short_interest_pct": 12.5,
    })
    assert rec.id is not None
    assert finra.get_latest_short_interest("ccc") is not None
