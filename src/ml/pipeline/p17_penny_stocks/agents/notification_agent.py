"""
P17 Notification Agent

Stage 8: deliver the day's actionable picks (Tier A / B) as a human-readable
email, plus a "watch" section of fresh-13G names (new passive >5% holder filed in
the last few days — not a buy signal; see CR-2026-11-06 §8). Delivery mirrors the P05 pattern — the message is queued via the
notification DB service (``NotificationService``) and the separate notification
processor performs the actual SMTP send. The recipient's email is resolved from
their ``user_id`` via ``UsersService`` so no SMTP credentials live in the
pipeline.

Any failure here is logged and swallowed: notifications must never abort the
pipeline, whose results are already written to disk by the reporting agent.
"""

import html
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.config import P17AlertConfig, P17FilingsConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate
from src.notification.logger import setup_logger

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
        user_id: Optional[Any] = None,
        filings_config: Optional[P17FilingsConfig] = None,
    ) -> None:
        self.cfg = alert_config
        self.target_date = target_date
        self.user_id = user_id
        self.filings_cfg = filings_config or P17FilingsConfig()

    def watch_list(self, candidates: List[Candidate]) -> List[Candidate]:
        """Fresh-13G names not already in the Tier A/B picks, highest score first."""
        if not self.filings_cfg.watch_13g_enabled:
            return []
        watch = [c for c in candidates if c.fresh_13g and c.tier not in self.ALERT_TIERS]
        watch.sort(key=lambda c: c.final_score, reverse=True)
        return watch[: self.filings_cfg.watch_max_names]

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
        watch = self.watch_list(candidates)
        if not picks and not watch:
            _logger.info("No Tier A/B candidates or fresh-13G names for %s — no email queued", self.target_date)
            return {"emailed": 0, "watch": 0, "reason": "no tier A/B candidates or watch names"}

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

        subject = self._subject(picks, watch)
        try:
            NotificationService().create_message(
                {
                    "message_type": "REPORT",
                    "channels": ["email"],
                    "recipient_id": str(self.user_id),
                    "content": {
                        "title": subject,
                        "message": self.format_text(picks, self.target_date, watch),
                        "html": self.format_html(picks, self.target_date, watch),
                        "source": "p17_penny_stocks",
                    },
                    "priority": "NORMAL",
                    "message_metadata": {"source": "p17_penny_stocks"},
                }
            )
            _logger.info(
                "Queued P17 email (%d Tier A/B picks, %d watch) for user %s", len(picks), len(watch), self.user_id
            )
            return {"emailed": len(picks), "watch": len(watch)}
        except Exception:
            _logger.exception("Failed to queue P17 email for user %s", self.user_id)
            return {"emailed": 0, "watch": 0, "reason": "queue failed"}

    # ── Formatting ─────────────────────────────────────────────────────────

    def _subject(self, picks: List[Candidate], watch: List[Candidate] | None = None) -> str:
        watch = watch or []
        watch_part = f" | watch: {len(watch)} fresh 13G" if watch else ""
        if not picks:
            return f"P17 Penny Screener — {self.target_date} — watch: {len(watch)} fresh 13G (no Tier A/B)"
        top = picks[0].ticker
        extra = f" +{len(picks) - 1} more" if len(picks) > 1 else ""
        return f"P17 Penny Screener — {self.target_date} — {top}{extra} (Tier A/B){watch_part}"

    @staticmethod
    def _pretty_next_catalyst(c: Candidate) -> str:
        """e.g. "pdufa in 12d (confirmed)" or "—"."""
        if not c.next_catalyst_type or c.next_catalyst_days is None:
            return "—"
        conf = f" ({c.next_catalyst_confidence})" if c.next_catalyst_confidence else ""
        when = "today" if c.next_catalyst_days == 0 else f"in {c.next_catalyst_days}d"
        return f"{c.next_catalyst_type.replace('_', ' ')} {when}{conf}"

    @staticmethod
    def _pretty_filings(c: Candidate) -> str:
        """Short insider / ownership summary, e.g. "13G; 2 insiders bought $85k"."""
        parts: List[str] = []
        if c.fresh_13g:
            parts.append("13G")
        if c.fresh_13d:
            parts.append("13D")
        if c.insider_buy_count:
            who = f"{c.insider_buyers} insider{'s' if c.insider_buyers != 1 else ''}"
            parts.append(f"{who} bought ${c.insider_buy_value / 1000:,.0f}k")
        if c.insider_sell_count:
            parts.append(f"insider sells ${c.insider_sell_value / 1000:,.0f}k")
        return "; ".join(parts) if parts else "—"

    @staticmethod
    def _pretty_catalysts(c: Candidate) -> str:
        """Turn raw catalyst signal slugs into a short human-readable phrase."""
        out: List[str] = []
        for sig in c.catalyst_signals[:3]:
            # e.g. "catalyst_material_agreement_2026-06-25" -> "material agreement (2026-06-25)"
            body = sig[len("catalyst_") :] if sig.startswith("catalyst_") else sig
            parts = body.rsplit("_", 1)
            if len(parts) == 2 and parts[1].count("-") == 2:
                out.append(f"{parts[0].replace('_', ' ')} ({parts[1]})")
            else:
                out.append(body.replace("_", " "))
        return ", ".join(out) if out else "—"

    @classmethod
    def format_text(cls, picks: List[Candidate], target_date: str, watch: List[Candidate] | None = None) -> str:
        """Plain-text fallback body."""
        lines = [
            f"P17 Explosive Penny Stock Screener — {target_date}",
            f"{len(picks)} actionable candidate(s) (Tier A / B), highest score first.",
            "",
        ]
        for i, c in enumerate(picks, 1):
            lines.append(f"{i}. {c.ticker} ({c.company_name or 'n/a'}) — Tier {c.tier} | score {c.final_score:.1f}")
            lines.append(
                f"     price ${c.price:.2f} | rvol {c.relative_volume:.1f}x | "
                f"catalyst {c.catalyst_score:.0f} | sector {c.sector or 'n/a'}"
            )
            lines.append(f"     catalysts: {cls._pretty_catalysts(c)}")
            lines.append(f"     next event: {cls._pretty_next_catalyst(c)} | filings: {cls._pretty_filings(c)}")
            if c.signals:
                lines.append(f"     signals: {', '.join(c.signals[:6])}")
            lines.append("")
        if watch:
            lines += [
                "WATCH — fresh 13G filings (new passive >5% holder in the last few days). NOT a buy signal.",
                "",
            ]
            for c in watch:
                lines.append(
                    f"- {c.ticker} ({c.company_name or 'n/a'}) — Tier {c.tier} | ${c.price:.2f} | "
                    f"5d {c.price_5d_return:+.0%} | next event: {cls._pretty_next_catalyst(c)} | "
                    f"filings: {cls._pretty_filings(c)}"
                )
            lines.append("")
        lines.append(
            "Risk note: penny-stock candidates are high-volatility and speculative. "
            "This is an automated screen, not investment advice."
        )
        return "\n".join(lines)

    @classmethod
    def format_html(cls, picks: List[Candidate], target_date: str, watch: List[Candidate] | None = None) -> str:
        """Rich, human-readable HTML body (a summary table + per-name detail)."""
        tier_badge = {
            "A": "#0b8043",  # green
            "B": "#1a73e8",  # blue
        }
        rows = []
        for i, c in enumerate(picks, 1):
            colour = tier_badge.get(c.tier, "#5f6368")
            # Ticker/company name/catalyst text originate from external data (Yahoo
            # Finance, NASDAQ symbol files, EDGAR filing descriptions) — escape before
            # interpolating into HTML that gets emailed out.
            ticker = html.escape(c.ticker)
            company = html.escape(c.company_name) if c.company_name else "—"
            tier = html.escape(c.tier)
            catalysts = html.escape(cls._pretty_catalysts(c))
            next_event = html.escape(cls._pretty_next_catalyst(c))
            rows.append(f"""
            <tr style="border-bottom:1px solid #eee;">
              <td style="padding:8px 10px;text-align:right;color:#888;">{i}</td>
              <td style="padding:8px 10px;font-weight:600;">{ticker}</td>
              <td style="padding:8px 10px;color:#444;">{company}</td>
              <td style="padding:8px 10px;text-align:center;">
                <span style="background:{colour};color:#fff;border-radius:4px;
                  padding:2px 8px;font-size:12px;font-weight:600;">{tier}</span>
              </td>
              <td style="padding:8px 10px;text-align:right;font-weight:600;">{c.final_score:.1f}</td>
              <td style="padding:8px 10px;text-align:right;">${c.price:.2f}</td>
              <td style="padding:8px 10px;text-align:right;">{c.relative_volume:.1f}x</td>
              <td style="padding:8px 10px;text-align:right;">{c.catalyst_score:.0f}</td>
              <td style="padding:8px 10px;color:#444;font-size:13px;">{catalysts}</td>
              <td style="padding:8px 10px;color:#444;font-size:13px;">{next_event}</td>
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
              <th style="padding:8px 10px;">Next event</th>
            </tr>
          </thead>
          <tbody>{"".join(rows)}</tbody>
        </table>"""

        return f"""
        <div style="font-family:Arial,Helvetica,sans-serif;color:#202124;line-height:1.5;">
          <h2 style="margin:0 0 4px;">P17 Explosive Penny Stock Screener</h2>
          <p style="margin:0 0 16px;color:#5f6368;">
            {target_date} — <b>{len(picks)}</b> actionable candidate(s), Tier A / B, highest score first.
          </p>
          {table if picks else '<p style="color:#5f6368;">No Tier A / B candidates today.</p>'}
          {cls._watch_html(watch or [])}
          <p style="margin:18px 0 0;color:#9aa0a6;font-size:12px;">
            Penny-stock candidates are high-volatility and speculative. This is an automated
            screen, not investment advice.
          </p>
        </div>"""

    @classmethod
    def _watch_html(cls, watch: List[Candidate]) -> str:
        """HTML "watch" section for fresh-13G names (empty string when none)."""
        if not watch:
            return ""
        rows = []
        for c in watch:
            company = html.escape(c.company_name) if c.company_name else "—"
            rows.append(f"""
            <tr style="border-bottom:1px solid #eee;">
              <td style="padding:6px 10px;font-weight:600;">{html.escape(c.ticker)}</td>
              <td style="padding:6px 10px;color:#444;">{company}</td>
              <td style="padding:6px 10px;text-align:center;">{html.escape(c.tier)}</td>
              <td style="padding:6px 10px;text-align:right;">${c.price:.2f}</td>
              <td style="padding:6px 10px;text-align:right;">{c.price_5d_return:+.0%}</td>
              <td style="padding:6px 10px;color:#444;font-size:13px;">{html.escape(cls._pretty_filings(c))}</td>
              <td style="padding:6px 10px;color:#444;font-size:13px;">{html.escape(cls._pretty_next_catalyst(c))}</td>
            </tr>""")
        return f"""
          <h3 style="margin:22px 0 4px;">Watch — fresh 13G filings</h3>
          <p style="margin:0 0 10px;color:#5f6368;font-size:13px;">
            A new passive holder above 5% filed in the last few days. <b>Not a buy signal</b>: in a May–Nov 2026
            study these names averaged +6.8% over the next 10 sessions vs −2.0% for the screen overall.
          </p>
          <table style="border-collapse:collapse;width:100%;font-family:Arial,Helvetica,sans-serif;font-size:14px;">
            <thead>
              <tr style="background:#f1f3f4;text-align:left;">
                <th style="padding:6px 10px;">Ticker</th>
                <th style="padding:6px 10px;">Company</th>
                <th style="padding:6px 10px;text-align:center;">Tier</th>
                <th style="padding:6px 10px;text-align:right;">Price</th>
                <th style="padding:6px 10px;text-align:right;">5d</th>
                <th style="padding:6px 10px;">Filings</th>
                <th style="padding:6px 10px;">Next event</th>
              </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
          </table>"""
