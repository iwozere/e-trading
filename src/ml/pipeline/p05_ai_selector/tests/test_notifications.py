"""Tests for P05Pipeline._send_notifications (rich Telegram + email delivery)."""

import sys
import types
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p05_ai_selector.pipeline import P05Pipeline
from src.ml.pipeline.p05_ai_selector.stages.stage4_output import Stage4Output


def _make_pick(rank: int = 1, confidence: int = 7) -> dict:
    return {
        "rank": rank,
        "ticker": f"TICK{rank}",
        "confidence": confidence,
        "bias": "long",
        "thesis": "A compelling thesis for this setup.",
        "risk_factors": ["Macro headwinds"],
        "time_horizon": "3-6 months",
        "exit_strategy": {
            "add_conditions": ["Pullback to $90"],
            "hold_conditions": ["Revenue stays positive"],
            "thesis_breakers": ["Miss guidance by >5%"],
            "profit_targets": [{"price_level": 120.0, "action": "Trim 25%", "note": "lock gains"}],
            "time_horizon_note": "Plays out over 2-3 quarters.",
        },
    }


def _picks(n: int = 5) -> list:
    return [_make_pick(i + 1) for i in range(n)]


def _run_with_channels(channels: dict, tmp_path: Path, user_id="2"):
    """
    Invoke _send_notifications with NotificationService / UsersService faked out
    via sys.modules, returning the mocked NotificationService instance so the
    caller can assert on create_message calls.
    """
    run_dir = tmp_path / "2026-06-26"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "top_picks.csv").write_text("rank,ticker\n1,TICK1\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# report", encoding="utf-8")

    fake_service = MagicMock()
    users_instance = MagicMock()
    users_instance.get_user_notification_channels.return_value = channels

    ns_mod = types.ModuleType("src.data.db.services.notification_service")
    setattr(ns_mod, "NotificationService", MagicMock(return_value=fake_service))
    us_mod = types.ModuleType("src.data.db.services.users_service")
    setattr(us_mod, "UsersService", MagicMock(return_value=users_instance))

    pipeline = P05Pipeline.__new__(P05Pipeline)  # skip __init__ (no instance state needed)
    output = Stage4Output(results_base=tmp_path)

    with patch.dict(
        sys.modules,
        {
            "src.data.db.services.notification_service": ns_mod,
            "src.data.db.services.users_service": us_mod,
        },
    ):
        pipeline._send_notifications(
            output=output,
            picks=_picks(),
            market_context="Markets are resilient.",
            trigger_reason="P18 flagged 5 ticker(s)",
            run_date=date(2026, 6, 26),
            results_dir=run_dir,
            user_id=user_id,
        )
    return fake_service


def _calls_by_channel(fake_service) -> dict:
    out = {}
    for call in fake_service.create_message.call_args_list:
        msg = call.args[0]
        out[msg["channels"][0]] = msg
    return out


def test_sends_both_channels(tmp_path):
    fake = _run_with_channels({"email": "a@b.com", "telegram_chat_id": "123"}, tmp_path)
    assert fake.create_message.call_count == 2
    by_channel = _calls_by_channel(fake)
    assert set(by_channel) == {"telegram", "email"}


def test_email_carries_html_and_attachments(tmp_path):
    fake = _run_with_channels({"email": "a@b.com", "telegram_chat_id": "123"}, tmp_path)
    email = _calls_by_channel(fake)["email"]
    content = email["content"]
    assert content["html"] and "Market Context" in content["html"]
    assert content["html"] and "Markets are resilient." in content["html"]
    files = content["attachments"]["files"]
    assert any(f.endswith("top_picks.csv") for f in files)
    assert any(f.endswith("report.md") for f in files)
    assert email["recipient_id"] == "2"


def test_telegram_is_text_only(tmp_path):
    fake = _run_with_channels({"email": "a@b.com", "telegram_chat_id": "123"}, tmp_path)
    tg = _calls_by_channel(fake)["telegram"]
    assert "html" not in tg["content"]
    assert "attachments" not in tg["content"]
    assert tg["content"]["message"]


def test_email_only_when_no_telegram(tmp_path):
    fake = _run_with_channels({"email": "a@b.com"}, tmp_path)
    assert fake.create_message.call_count == 1
    assert list(_calls_by_channel(fake)) == ["email"]


def test_no_channels_sends_nothing(tmp_path):
    fake = _run_with_channels({}, tmp_path)
    assert fake.create_message.call_count == 0


def test_missing_user_id_sends_nothing(tmp_path):
    # user_id None must short-circuit before any service lookup.
    pipeline = P05Pipeline.__new__(P05Pipeline)
    output = Stage4Output(results_base=tmp_path)
    # Should not raise even though no DB modules are patched in.
    pipeline._send_notifications(
        output=output,
        picks=_picks(),
        market_context="ctx",
        trigger_reason="reason",
        run_date=date(2026, 6, 26),
        results_dir=tmp_path,
        user_id=None,
    )
