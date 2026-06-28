"""Tests for P17 NotificationAgent (Tier A/B email build + queue)."""

from pathlib import Path
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.agents.notification_agent import NotificationAgent
from src.ml.pipeline.p17_penny_stocks.config import P17AlertConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate

DATE = "2026-06-26"


def _cand(ticker, tier, score, **kw) -> Candidate:
    return Candidate(ticker=ticker, tier=tier, final_score=score, price=4.0, **kw)


def _agent(user_id="123", **cfg_kw) -> NotificationAgent:
    cfg_kw.setdefault("enabled", True)
    cfg_kw.setdefault("email_enabled", True)
    return NotificationAgent(P17AlertConfig(**cfg_kw), DATE, user_id)


# ── Filtering / early returns ───────────────────────────────────────────────

def test_skips_when_email_disabled():
    agent = _agent(email_enabled=False)
    assert agent.run([_cand("AAA", "B", 60)])["emailed"] == 0


def test_skips_when_no_user_id():
    agent = _agent(user_id=None)
    assert agent.run([_cand("AAA", "B", 60)])["reason"] == "no user_id"


def test_skips_when_no_tier_ab():
    agent = _agent()
    res = agent.run([_cand("AAA", "C", 40), _cand("BBB", "W", 10)])
    assert res["emailed"] == 0 and "no tier" in res["reason"]


# ── Queue path (services injected) ──────────────────────────────────────────

def _inject_services(monkeypatch, email="me@example.com"):
    created = {}
    users_mod = types.ModuleType("src.data.db.services.users_service")
    notif_mod = types.ModuleType("src.data.db.services.notification_service")

    class UsersService:
        def get_user_notification_channels(self, uid):
            return {"email": email} if email else {}

    class NotificationService:
        def create_message(self, payload):
            created["payload"] = payload

    users_mod.UsersService = UsersService
    notif_mod.NotificationService = NotificationService
    monkeypatch.setitem(sys.modules, "src.data.db.services.users_service", users_mod)
    monkeypatch.setitem(sys.modules, "src.data.db.services.notification_service", notif_mod)
    return created


def test_queues_email_for_tier_ab(monkeypatch):
    created = _inject_services(monkeypatch)
    agent = _agent()
    res = agent.run([_cand("AAA", "B", 60), _cand("ZZZ", "A", 80), _cand("CCC", "C", 40)])
    assert res["emailed"] == 2                      # A and B only
    payload = created["payload"]
    assert payload["channels"] == ["email"]
    assert payload["content"]["source"] == "p17_penny_stocks"
    # highest score first in the subject
    assert "ZZZ" in payload["content"]["title"]
    assert "AAA" in payload["content"]["html"] and "ZZZ" in payload["content"]["html"]


def test_no_email_channel_skips(monkeypatch):
    _inject_services(monkeypatch, email=None)
    res = _agent().run([_cand("AAA", "B", 60)])
    assert res["emailed"] == 0 and res["reason"] == "no email channel"


# ── Formatting ──────────────────────────────────────────────────────────────

def test_pretty_catalysts_humanises_slug():
    c = _cand("AAA", "B", 60,
              catalyst_signals=["catalyst_material_agreement_2026-06-25"])
    assert NotificationAgent._pretty_catalysts(c) == "material agreement (2026-06-25)"


def test_format_text_and_html_contain_picks():
    picks = [_cand("AAA", "B", 60, company_name="Alpha Inc",
                   catalyst_signals=["catalyst_tier1_news_2026-06-20"])]
    text = NotificationAgent.format_text(picks, DATE)
    html = NotificationAgent.format_html(picks, DATE)
    assert "AAA" in text and "Alpha Inc" in text and DATE in text
    assert "AAA" in html and "<table" in html
