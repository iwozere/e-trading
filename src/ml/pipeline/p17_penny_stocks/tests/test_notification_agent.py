"""Tests for P17 NotificationAgent (Tier A/B email build + queue)."""

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.agents.notification_agent import NotificationAgent
from src.ml.pipeline.p17_penny_stocks.config import P17AlertConfig, P17FilingsConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate

DATE = "2026-06-26"


def _cand(ticker, tier, score, **kw) -> Candidate:
    return Candidate(ticker=ticker, tier=tier, final_score=score, price=4.0, **kw)


def _agent(user_id: str | None = "123", **cfg_kw) -> NotificationAgent:
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


def _inject_services(monkeypatch, email: str | None = "me@example.com"):
    created = {}
    users_mod = types.ModuleType("src.data.db.services.users_service")
    notif_mod = types.ModuleType("src.data.db.services.notification_service")

    class UsersService:
        def get_user_notification_channels(self, uid):
            return {"email": email} if email else {}

    class NotificationService:
        def create_message(self, payload):
            created["payload"] = payload

    setattr(users_mod, "UsersService", UsersService)
    setattr(notif_mod, "NotificationService", NotificationService)
    monkeypatch.setitem(sys.modules, "src.data.db.services.users_service", users_mod)
    monkeypatch.setitem(sys.modules, "src.data.db.services.notification_service", notif_mod)
    return created


def test_queues_email_for_tier_ab(monkeypatch):
    created = _inject_services(monkeypatch)
    agent = _agent()
    res = agent.run([_cand("AAA", "B", 60), _cand("ZZZ", "A", 80), _cand("CCC", "C", 40)])
    assert res["emailed"] == 2  # A and B only
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
    c = _cand("AAA", "B", 60, catalyst_signals=["catalyst_material_agreement_2026-06-25"])
    assert NotificationAgent._pretty_catalysts(c) == "material agreement (2026-06-25)"


def test_format_text_and_html_contain_picks():
    picks = [_cand("AAA", "B", 60, company_name="Alpha Inc", catalyst_signals=["catalyst_tier1_news_2026-06-20"])]
    text = NotificationAgent.format_text(picks, DATE)
    html = NotificationAgent.format_html(picks, DATE)
    assert "AAA" in text and "Alpha Inc" in text and DATE in text
    assert "AAA" in html and "<table" in html


def test_format_html_escapes_external_data():
    """Company name/ticker come from Yahoo/NASDAQ data — must not inject raw HTML."""
    picks = [_cand("AAA", "B", 60, company_name="<script>alert(1)</script> Corp")]
    html = NotificationAgent.format_html(picks, DATE)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ── Fresh-13G watch section ─────────────────────────────────────────────────


def test_watch_only_email_is_queued(monkeypatch):
    created = _inject_services(monkeypatch)
    agent = _agent()
    res = agent.run([_cand("WWW", "C", 40, fresh_13g=True, company_name="Watch Co"), _cand("ZZZ", "W", 5)])
    assert res == {"emailed": 0, "watch": 1}
    content = created["payload"]["content"]
    assert "watch: 1 fresh 13G" in content["title"]
    assert "Watch — fresh 13G filings" in content["html"] and "WWW" in content["html"]
    assert "NOT a buy signal" in content["message"]


def test_watch_excludes_tier_ab_and_respects_cap():
    agent = NotificationAgent(P17AlertConfig(), DATE, "1", filings_config=P17FilingsConfig(watch_max_names=2))
    cands = [
        _cand("AAA", "B", 70, fresh_13g=True),
        _cand("C1", "C", 50, fresh_13g=True),
        _cand("C2", "C", 45, fresh_13g=True),
        _cand("C3", "W", 20, fresh_13g=True),
    ]
    assert [c.ticker for c in agent.watch_list(cands)] == ["C1", "C2"]


def test_watch_disabled():
    agent = NotificationAgent(P17AlertConfig(), DATE, "1", filings_config=P17FilingsConfig(watch_13g_enabled=False))
    assert agent.watch_list([_cand("C1", "C", 50, fresh_13g=True)]) == []


def test_next_event_and_filings_rendered():
    pick = _cand(
        "AAA", "B", 60, next_catalyst_type="pdufa", next_catalyst_days=12, next_catalyst_confidence="confirmed",
        insider_buy_count=2, insider_buyers=2, insider_buy_value=85_000.0,
    )
    text = NotificationAgent.format_text([pick], DATE)
    assert "pdufa in 12d (confirmed)" in text
    assert "2 insiders bought $85k" in text
    assert "pdufa in 12d (confirmed)" in NotificationAgent.format_html([pick], DATE)
