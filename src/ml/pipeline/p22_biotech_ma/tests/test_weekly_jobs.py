"""Tests for the 2026-11-09 automation: label cut-off, emails, live learned ranking and the two weekly jobs."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest import learned
from src.ml.pipeline.p22_biotech_ma.backtest.labels import labels_complete_through
from src.ml.pipeline.p22_biotech_ma.backtest.learned import LEARNED_FEATURES, LearnedParams, fit_final_model
from src.ml.pipeline.p22_biotech_ma.jobs import run_backtest, run_label_review_digest
from src.ml.pipeline.p22_biotech_ma.outputs import backtest_report, email_notify, label_review_digest
from src.ml.pipeline.p22_biotech_ma.scoring import learned_live

_NOW = datetime(2026, 11, 9, 8, 0, tzinfo=timezone.utc)


def _candidate(item_id: int, filed: str, company_id: int = 1, created: datetime = _NOW, **extra: Any) -> Dict[str, Any]:
    return {"item_id": item_id, "item_type": "deal_candidate", "created_at": created, "evidence_url": f"https://sec.gov/{item_id}",
            "payload": {"company_id": company_id, "form_type": "SC 14D9", "filed_date": filed,
                        "all_names_in_filing": ["Target Inc", "Big Pharma Co"], **extra}}


# ---------------------------------------------------------------------
# label cut-off
# ---------------------------------------------------------------------

def test_labels_complete_through_is_day_before_oldest_pending_filing():
    pending = [_candidate(1, "2026-10-20"), _candidate(2, "2026-09-03"), _candidate(3, "not a date")]
    assert labels_complete_through(pending, date(2026, 11, 9)) == date(2026, 9, 2)


def test_labels_complete_through_is_today_when_nothing_pending():
    assert labels_complete_through([], date(2026, 11, 9)) == date(2026, 11, 9)
    assert labels_complete_through([_candidate(1, "2026-12-01")], date(2026, 11, 9)) == date(2026, 11, 9)


# ---------------------------------------------------------------------
# email helper
# ---------------------------------------------------------------------

def test_html_table_escapes_cells_but_keeps_links():
    table = email_notify.html_table(["a"], [["<b>x</b>"], [email_notify.link("https://x.org/?a=1&b=2", "doc")], [None], [1.234]])
    assert "&lt;b&gt;x&lt;/b&gt;" in table
    assert '<a href="https://x.org/?a=1&amp;b=2">doc</a>' in table
    assert "1.23" in table


def test_queue_email_builds_html_message_for_system_user_by_default():
    sent: List[Dict[str, Any]] = []
    assert email_notify.queue_email("T", "text", "<p>x</p>", create_message=sent.append)
    message = sent[0]
    assert message["channels"] == ["email"] and message["content"]["html"] == "<p>x</p>"
    assert message["recipient_id"] == str(email_notify.SYSTEM_USER_ID)
    assert email_notify.build_email("T", "t", "h", user_id=7)["recipient_id"] == "7"


def test_queue_email_returns_false_when_db_write_fails():
    def boom(_: Dict[str, Any]) -> None:
        raise RuntimeError("db down")

    assert email_notify.queue_email("T", "text", "<p>x</p>", create_message=boom) is False


# ---------------------------------------------------------------------
# backtest report
# ---------------------------------------------------------------------

def _metrics(learned50: float, naive50: float) -> List[Dict[str, Any]]:
    rows = []
    for name, lift50 in (("learned", learned50), ("naive_informed", naive50), ("model", 0.8), ("block:strategic_fit", 1.1)):
        for k in (25, 50, 200):
            rows.append({"ranking": name, "k": k, "lift": lift50 if k == 50 else 1.0, "snapshots": 75})
    return rows


def test_backtest_report_headline_and_change_since_last_run():
    summary = {"labels_complete_through": "2026-11-06", "snapshots": 103, "deals_entered": 532,
               "metrics_learned_window": _metrics(1.62, 1.54)}
    previous = {"labels_complete_through": "2026-10-30", "metrics_learned_window": _metrics(1.50, 1.54)}
    title, text, body = backtest_report.build_backtest_report(summary, previous, {"saved": True, "train_rows": 65057, "train_positives": 4154})
    assert title == "P22 weekly backtest: learned lift@50 1.62 vs naive 1.54 (beats naive)"
    assert "+0.12" in body and "block:strategic_fit" in body and "65057 rows" in body
    assert "learned" in text


def test_backtest_report_flags_below_naive_and_unsaved_model():
    summary = {"metrics_learned_window": _metrics(1.40, 1.54)}
    title, _, body = backtest_report.build_backtest_report(summary, None, {"saved": False})
    assert "BELOW naive" in title and "keeps the previous model" in body


def test_previous_summary_picks_newest_earlier_run(tmp_path):
    for day, value in (("2026-10-31", 1), ("2026-11-07", 2), ("2026-11-14", 3)):
        folder = tmp_path / day / "backtest"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({"v": value}), encoding="utf-8")
    assert backtest_report.previous_summary(tmp_path, tmp_path / "2026-11-14") == {"v": 2}
    assert backtest_report.previous_summary(tmp_path, tmp_path / "2026-10-31") is None


# ---------------------------------------------------------------------
# label-review digest
# ---------------------------------------------------------------------

def test_digest_lists_new_and_oldest_and_other_queues():
    pending = [
        _candidate(1, "2026-11-05", company_id=1),
        _candidate(2, "2025-02-01", company_id=2, created=_NOW - timedelta(days=60)),
        {"item_id": 3, "item_type": "process_event", "created_at": _NOW, "payload": {}},
    ]
    companies = {1: {"ticker": "TGT", "name": "Target Inc"}, 2: {"ticker": "OLD", "name": "Old Co"}}
    title, text, body, stats = label_review_digest.build_digest(pending, companies, _NOW, 7)
    assert stats == {"deal_candidates_pending": 2, "deal_candidates_new": 1, "labels_complete_through": "2025-01-31",
                     "other_pending": {"process_event": 1}}
    assert title.startswith("P22 label review: 1 new deal candidates, 2 pending")
    assert "Big Pharma Co" in body and "https://sec.gov/2" in body
    assert label_review_digest._row(pending[0], companies)[5] == "Big Pharma Co"  # the company itself is not an "other party"
    assert "TGT" in text and "OLD" not in text  # text lists only the new ones
    assert label_review_digest.should_send(stats)


def test_digest_not_sent_when_queues_empty():
    *_, stats = label_review_digest.build_digest([], {}, _NOW, 7)
    assert not label_review_digest.should_send(stats)


# ---------------------------------------------------------------------
# live learned ranking
# ---------------------------------------------------------------------

def _trained_model() -> learned.SavedModel:
    rows = []
    for i in range(60):
        row: Dict[str, Any] = {name: 1.0 for name in LEARNED_FEATURES}
        row.update({"as_of": date(2018, 1, 1), "company_id": i, "positive": int(i % 3 == 0), "f_phase_max": 3.0 if i % 3 == 0 else 1.0})
        rows.append(row)
    model = fit_final_model(rows, date(2026, 11, 6), LearnedParams(min_train_positives=5, max_iter=30, min_samples_leaf=2))
    assert model is not None
    return model


def test_learned_ranking_ranks_live_rows_with_backtest_columns(monkeypatch):
    model = _trained_model()
    monkeypatch.setattr(learned_live, "baseline_features",
                        lambda ids, as_of, ctx: {cid: {"phase_max": 3.0 if cid == 2 else 1.0, "cash_runway_months": 12.0} for cid in ids})
    rows = [{"company_id": cid, "composite": 0.5, "rank_by_composite": cid, "tier": 0, "subscores": {"features": {}}} for cid in (1, 2)]
    ranking = learned_live.learned_ranking(object(), date(2026, 11, 9), rows, model)
    assert ranking[2][1] == 1 and ranking[1][1] == 2  # phase 3 ranks first
    learned_live.apply_learned(rows, ranking, model)
    assert rows[1]["rank_by_learned"] == 1 and rows[1]["subscores"]["learned"]["labels_through"] == "2026-11-06"
    assert [r["company_id"] for r in learned_live.by_learned_rank(rows)] == [2, 1]
    assert learned_live.learned_ranking(object(), date(2026, 11, 9), [], model) == {}


def test_by_learned_rank_puts_unranked_last():
    rows = [{"company_id": 1, "rank_by_learned": None, "rank_by_composite": 1}, {"company_id": 2, "rank_by_learned": 1, "rank_by_composite": 2}]
    assert [r["company_id"] for r in learned_live.by_learned_rank(rows)] == [2, 1]


# ---------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------

class _Uow:
    def __init__(self, pending: List[Dict[str, Any]]) -> None:
        self.p22 = SimpleNamespace(get_pending_review_items=lambda item_type=None: [p for p in pending if item_type in (None, p["item_type"])],
                                   list_companies_full=lambda: [{"company_id": 1, "ticker": "TGT", "name": "Target Inc"}])

    def __enter__(self) -> "_Uow":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _db(pending: List[Dict[str, Any]]) -> Any:
    return lambda: SimpleNamespace(uow=lambda: _Uow(pending))


def test_run_backtest_uses_review_cutoff_saves_model_and_emails(monkeypatch, tmp_path):
    seen: Dict[str, Any] = {}
    model_path = tmp_path / "models" / "learned_model.pkl"
    monkeypatch.setattr(run_backtest, "setup_run_logging", lambda: tmp_path / "2026-11-14")
    monkeypatch.setattr(run_backtest, "DatabaseService", _db([_candidate(1, "2026-10-01")]))
    monkeypatch.setattr(run_backtest, "LEARNED_MODEL_PATH", model_path)
    monkeypatch.setattr(run_backtest, "RESULTS_DIR", tmp_path)

    def fake_execute(start, end, labels_through, step_months, ks, out_dir):
        seen["labels_through"] = labels_through
        out_dir.mkdir(parents=True)
        return SimpleNamespace(score_rows=[]), {"snapshots": 100, "metrics_learned_window": _metrics(1.62, 1.54)}

    monkeypatch.setattr(run_backtest, "execute", fake_execute)
    monkeypatch.setattr(run_backtest, "fit_final_model", lambda rows, through: _trained_model())
    monkeypatch.setattr(run_backtest, "queue_email", lambda title, text, body, user_id: seen.setdefault("title", title) is not None)

    report = run_backtest.run(today=date(2026, 11, 14))
    assert seen["labels_through"] == date(2026, 9, 30)
    assert report["success"] and report["model_saved"] and report["email_queued"] and report["beats_naive_at_50"]
    assert report["learned_lift_at_50"] == 1.62 and report["deal_candidates_pending"] == 1
    assert model_path.exists() and (tmp_path / "2026-11-14" / "backtest" / "learned_model.pkl").exists()
    assert "1.62" in seen["title"]


def test_run_backtest_reports_failure_without_dera_quarters(monkeypatch, tmp_path):
    monkeypatch.setattr(run_backtest, "setup_run_logging", lambda: tmp_path)
    monkeypatch.setattr(run_backtest, "DatabaseService", _db([]))
    monkeypatch.setattr(run_backtest, "execute", lambda *args: None)
    report = run_backtest.run(today=date(2026, 11, 14))
    assert report["success"] is False and report["labels_complete_through"] == "2026-11-14"


def test_run_label_review_digest_sends_only_when_pending(monkeypatch, tmp_path):
    sent: List[str] = []
    monkeypatch.setattr(run_label_review_digest, "setup_run_logging", lambda: tmp_path)
    monkeypatch.setattr(run_label_review_digest, "queue_email", lambda title, text, body, user_id: sent.append(title) is None)
    monkeypatch.setattr(run_label_review_digest, "DatabaseService", _db([_candidate(1, "2026-11-05")]))
    report = run_label_review_digest.run(now=_NOW)
    assert report["email_queued"] and report["deal_candidates_new"] == 1 and len(sent) == 1
    monkeypatch.setattr(run_label_review_digest, "DatabaseService", _db([]))
    assert run_label_review_digest.run(now=_NOW)["email_queued"] is False and len(sent) == 1
