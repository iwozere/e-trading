"""
P17 Notification Agent

Stage 8: deliver the day's actionable picks (Tier A / B) as a human-readable
email. Delivery mirrors the P05 pattern — the message is queued via the
notification DB service (``NotificationService``) and the separate notification
processor performs the actual SMTP send. The recipient's email is resolved from
their ``user_id`` via ``UsersService`` so no SMTP credentials live in the
pipeline.

Any failure here is logged and swallowed: notifications must never abort the
pipeline, whose results are already written to disk by the reporting agent.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p17_penny_stocks.config import P17AlertConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate

_logger = setup_logger(__name__)


class NotificationAgent:
    """
    Email the actionable Tier A / B candidates for a run.

    Tier B is the requested alert scope; Tier A is included because an elite pick
    should never be omitted from an alert that already carries the lower tier.
    """

    ALERT_TIERS = ("A", "B")

    def __init__(
        self,
        alert_config: P17AlertConfig,
        target_date: str,
        user_id: Any = None,
    ) -> None:
        self.cfg = alert_config
        self.target_date = target_date
        self.user_id = user_id

    def run(self, candidates: List[Candidate]) -> Dict[str, Any]:
        """
        Build and queue the Tier A/B email.

        Returns:
            Dict with ``emailed`` (count of picks in the email) and, when nothing
            was sent, a ``reason`` describing why.
        """
        if not self.cfg.enabled or not self.cfg.email_enabled:
            return {"emailed": 0, "reason": "email disabled"}

        if not self.user_id:
            _logger.warning("No user_id provided — skipping P17 Tier A/B email")
            return {"emailed": 0, "reason": "no user_id"}

        picks = sorted(
            (c for c in candidates if c.tier in self.ALERT_TIERS),
            key=lambda c: c.final_score,
            reverse=True,
        )
        if not picks:
            _logger.info("No Tier A/B candidates for %s — no email queued", self.target_date)
            return {"emailed": 0, "reason": "no tier A/B candidates"}

        try:
            from src.data.db.services.notification_service import NotificationService
            from src.data.db.services.users_service import UsersService

            channels = UsersService().get_user_notification_channels(int(self.user_id))
        except (ValueError, TypeError):
            _logger.warning("Invalid user_id %r — skipping P17 email", self.user_id)
            return {"emailed": 0, "reason": "invalid user_id"}
        except Exception:
            _logger.exception("Could not resolve notification channels for user %s", self.user_id)
            return {"emailed": 0, "reason": "channel resolution failed"}

        if not channels or not channels.get("email"):
            _logger.warning("No email channel for user %s — skipping P17 email", self.user_id)
            return {"emailed": 0, "reason": "no email channel"}

        subject = self._subject(picks)
        try:
            NotificationService().create_message({
                "message_type": "REPORT",
                "channels": ["email"],
                "recipient_id": str(self.user_id),
                "content": {
                    "title": subject,
                    "message": self.format_text(picks, self.target_date),
                    "html": self.format_html(picks, self.target_date),
                    "source": "p17_penny_stocks",
                },
                "priority": "NORMAL",
                "message_metadata": {"source": "p17_penny_stocks"},
            })
            _logger.info("Queued P17 Tier A/B email (%d picks) for user %s", len(picks), self.user_id)
            return {"emailed": len(picks)}
        except Exception:
            _logger.exception("Failed to queue P17 email for user %s", self.user_id)
            return {"emailed": 0, "reason": "queue failed"}

    # ── Formatting ─────────────────────────────────────────────────────────

    def _subject(self, picks: List[Candidate]) -> str:
        top = picks[0].ticker
        extra = f" +{len(picks) - 1} more" if len(picks) > 1 else ""
        return f"P17 Penny Screener — {self.target_date} — {top}{extra} (Tier A/B)"

    @staticmethod
    def _pretty_catalysts(c: Candidate) -> str:
        """Turn raw catalyst signal slugs into a short human-readable phrase."""
        out: List[str] = []
        for sig in c.catalyst_signals[:3]:
            # e.g. "catalyst_material_agreement_2026-06-25" -> "material agreement (2026-06-25)"
            body = sig[len("catalyst_"):] if sig.startswith("catalyst_") else sig
            parts = body.rsplit("_", 1)
            if len(parts) == 2 and parts[1].count("-") == 2:
                out.append(f"{parts[0].replace('_', ' ')} ({parts[1]})")
            else:
                out.append(body.replace("_", " "))
        return ", ".join(out) if out else "—"

    @classmethod
    def format_text(cls, picks: List[Candidate], target_date: str) -> str:
        """Plain-text fallback body."""
        lines = [
            f"P17 Explosive Penny Stock Screener — {target_date}",
            f"{len(picks)} actionable candidate(s) (Tier A / B), highest score first.",
            "",
        ]
        for i, c in enumerate(picks, 1):
            lines.append(
                f"{i}. {c.ticker} ({c.company_name or 'n/a'}) — Tier {c.tier} | "
                f"score {c.final_score:.1f}"
            )
            lines.append(
                f"     price ${c.price:.2f} | rvol {c.relative_volume:.1f}x | "
                f"catalyst {c.catalyst_score:.0f} | sector {c.sector or 'n/a'}"
            )
            lines.append(f"     catalysts: {cls._pretty_catalysts(c)}")
            if c.signals:
                lines.append(f"     signals: {', '.join(c.signals[:6])}")
            lines.append("")
        lines.append(
            "Risk note: penny-stock candidates are high-volatility and speculative. "
            "This is an automated screen, not investment advice."
        )
        return "\n".join(lines)

    @classmethod
    def format_html(cls, picks: List[Candidate], target_date: str) -> str:
        """Rich, human-readable HTML body (a summary table + per-name detail)."""
        tier_badge = {
            "A": "#0b8043",  # green
            "B": "#1a73e8",  # blue
        }
        rows = []
        for i, c in enumerate(picks, 1):
            colour = tier_badge.get(c.tier, "#5f6368")
            rows.append(f"""
            <tr style="border-bottom:1px solid #eee;">
              <td style="padding:8px 10px;text-align:right;color:#888;">{i}</td>
              <td style="padding:8px 10px;font-weight:600;">{c.ticker}</td>
              <td style="padding:8px 10px;color:#444;">{c.company_name or '—'}</td>
              <td style="padding:8px 10px;text-align:center;">
                <span style="background:{colour};color:#fff;border-radius:4px;
                  padding:2px 8px;font-size:12px;font-weight:600;">{c.tier}</span>
              </td>
              <td style="padding:8px 10px;text-align:right;font-weight:600;">{c.final_score:.1f}</td>
              <td style="padding:8px 10px;text-align:right;">${c.price:.2f}</td>
              <td style="padding:8px 10px;text-align:right;">{c.relative_volume:.1f}x</td>
              <td style="padding:8px 10px;text-align:right;">{c.catalyst_score:.0f}</td>
              <td style="padding:8px 10px;color:#444;font-size:13px;">{cls._pretty_catalysts(c)}</td>
            </tr>""")

        table = f"""
        <table style="border-collapse:collapse;width:100%;font-family:Arial,Helvetica,sans-serif;
          font-size:14px;">
          <thead>
            <tr style="background:#f1f3f4;text-align:left;">
              <th style="padding:8px 10px;">#</th>
              <th style="padding:8px 10px;">Ticker</th>
              <th style="padding:8px 10px;">Company</th>
              <th style="padding:8px 10px;text-align:center;">Tier</th>
              <th style="padding:8px 10px;text-align:right;">Score</th>
              <th style="padding:8px 10px;text-align:right;">Price</th>
              <th style="padding:8px 10px;text-align:right;">RVol</th>
              <th style="padding:8px 10px;text-align:right;">Catalyst</th>
              <th style="padding:8px 10px;">Catalyst signals</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>"""

        return f"""
        <div style="font-family:Arial,Helvetica,sans-serif;color:#202124;line-height:1.5;">
          <h2 style="margin:0 0 4px;">P17 Explosive Penny Stock Screener</h2>
          <p style="margin:0 0 16px;color:#5f6368;">
            {target_date} — <b>{len(picks)}</b> actionable candidate(s), Tier A / B, highest score first.
          </p>
          {table}
          <p style="margin:18px 0 0;color:#9aa0a6;font-size:12px;">
            Penny-stock candidates are high-volatility and speculative. This is an automated
            screen, not investment advice.
          </p>
        </div>"""
