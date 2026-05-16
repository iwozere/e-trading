"""
P17 Reporting Agent

Generates three output artefacts from the scored candidate list:
  - {date}_candidates.csv   — flat CSV of all scored candidates
  - {date}_report.json      — full structured report (top candidates + metadata)
  - {date}_report.md        — human-readable Markdown daily report
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p17_penny_stocks.config import P17ScoringConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate

_FearGreedDownloader = None
try:
    from src.data.downloader.fear_greed_downloader import FearGreedDownloader as _FearGreedDownloader  # type: ignore[assignment]
except Exception:
    pass

_logger = setup_logger(__name__)


class ReportingAgent:
    """
    Stage 7: Generate output files from the ranked candidate list.

    Files are written to results_dir and the paths are returned for
    downstream notification/archival use.
    """

    def __init__(
        self,
        scoring_config: P17ScoringConfig,
        results_dir: Path,
        run_date: str,
        top_n: int = 20,
    ) -> None:
        self.cfg = scoring_config
        self.results_dir = results_dir
        self.run_date = run_date
        self.top_n = top_n

    def run(self, candidates: List[Candidate]) -> dict:
        """
        Write all report artefacts.

        Args:
            candidates: Scored and sorted list (descending by final_score).

        Returns:
            Dict with paths to generated files and summary counts.
        """
        if not candidates:
            _logger.warning("No candidates to report")
            return {"candidates": 0}

        fg_context = self._load_fear_greed()

        csv_path = self._write_csv(candidates)
        json_path = self._write_json(candidates, fg_context)
        md_path = self._write_markdown(candidates, fg_context)

        tier_counts = self._tier_counts(candidates)
        _logger.info(
            "Report written: %d total | A=%d B=%d C=%d W=%d",
            len(candidates),
            tier_counts.get("A", 0), tier_counts.get("B", 0),
            tier_counts.get("C", 0), tier_counts.get("W", 0),
        )

        return {
            "candidates": len(candidates),
            "tier_a": tier_counts.get("A", 0),
            "tier_b": tier_counts.get("B", 0),
            "tier_c": tier_counts.get("C", 0),
            "tier_w": tier_counts.get("W", 0),
            "explosive": sum(1 for c in candidates if c.explosive_candidate),
            "fear_greed_score": fg_context.get("score") if fg_context else None,
            "fear_greed_label": fg_context.get("label") if fg_context else None,
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "md_path": str(md_path),
        }

    # ── CSV ────────────────────────────────────────────────────────────────

    def _write_csv(self, candidates: List[Candidate]) -> Path:
        rows = [c.to_dict() for c in candidates]
        df = pd.DataFrame(rows)
        path = self.results_dir / f"{self.run_date}_candidates.csv"
        df.to_csv(path, index=False)
        _logger.info("CSV written: %s (%d rows)", path, len(df))
        return path

    # ── JSON ───────────────────────────────────────────────────────────────

    def _write_json(self, candidates: List[Candidate], fg_context: Optional[dict]) -> Path:
        top = candidates[: self.top_n]

        report = {
            "run_date": self.run_date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "market_sentiment": fg_context,
            "summary": {
                "total_candidates": len(candidates),
                **self._tier_counts(candidates),
                "explosive_count": sum(1 for c in candidates if c.explosive_candidate),
            },
            "top_candidates": [self._candidate_to_json(c) for c in top],
        }

        path = self.results_dir / f"{self.run_date}_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        _logger.info("JSON report written: %s", path)
        return path

    def _candidate_to_json(self, c: Candidate) -> dict:
        return {
            "ticker": c.ticker,
            "company": c.company_name,
            "tier": c.tier,
            "explosive": c.explosive_candidate,
            "final_score": round(c.final_score, 1),
            "price": round(c.price, 2),
            "market_cap_m": round(c.market_cap / 1_000_000, 1) if c.market_cap else None,
            "float_m": round(c.float_shares / 1_000_000, 1) if c.float_shares else None,
            "rvol": round(c.relative_volume, 2),
            "20d_return_pct": round(c.price_20d_return * 100, 1),
            "short_interest_pct": (
                round(c.short_interest_pct_float * 100, 1)
                if c.short_interest_pct_float is not None else None
            ),
            "revenue_growth_pct": (
                round(c.revenue_growth_yoy * 100, 1)
                if c.revenue_growth_yoy is not None else None
            ),
            "dilution_penalty": c.dilution_penalty,
            "signals": c.signals,
            "score_breakdown": c.score_breakdown(),
        }

    # ── Markdown ───────────────────────────────────────────────────────────

    def _write_markdown(self, candidates: List[Candidate], fg_context: Optional[dict]) -> Path:
        tiers = self._tier_counts(candidates)
        explosive = [c for c in candidates if c.explosive_candidate]
        tier_a = [c for c in candidates if c.tier == "A"]
        tier_b = [c for c in candidates if c.tier == "B"][:5]
        tier_c = [c for c in candidates if c.tier == "C"][:5]

        fg_row = ""
        if fg_context:
            score = fg_context.get("score", "?")
            label = fg_context.get("label", "").replace("_", " ").title()
            fg_row = f"| Market sentiment (F&G) | {score:.0f} — {label} |"

        lines = [
            f"# P17 Penny Stock Screener — {self.run_date}",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total candidates | {len(candidates)} |",
            f"| Tier A (elite) | {tiers.get('A', 0)} |",
            f"| Tier B (momentum) | {tiers.get('B', 0)} |",
            f"| Tier C (speculative) | {tiers.get('C', 0)} |",
            f"| Explosive candidates | {len(explosive)} |",
        ]
        if fg_row:
            lines.append(fg_row)
        lines.append("")

        if fg_context and fg_context.get("label") == "extreme_fear":
            lines += [
                "> **⚠ Market Sentiment: Extreme Fear "
                f"({fg_context.get('score', '?'):.0f})** — "
                "broad risk-off environment. Validate all setups carefully "
                "and consider reducing position size.",
                "",
            ]

        if explosive:
            lines += ["## Explosive Candidates", ""]
            lines += self._candidate_table(explosive[:10])
            lines.append("")

        if tier_a:
            lines += ["## Tier A — Elite", ""]
            lines += self._candidate_table(tier_a[:10])
            lines.append("")

        if tier_b:
            lines += ["## Tier B — Momentum (top 5)", ""]
            lines += self._candidate_table(tier_b)
            lines.append("")

        if tier_c:
            lines += ["## Tier C — Speculative (top 5)", ""]
            lines += self._candidate_table(tier_c)
            lines.append("")

        # Risk warnings
        high_dilution = [c for c in candidates if c.dilution_penalty >= 20]
        if high_dilution:
            lines += ["## ⚠ Dilution Warnings", ""]
            for c in high_dilution:
                lines.append(
                    f"- **{c.ticker}**: penalty={c.dilution_penalty:.0f}  "
                    f"signals={', '.join(c.dilution_signals)}"
                )
            lines.append("")

        lines += [
            "---",
            f"*Generated by P17 Penny Stock Screener at "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        ]

        path = self.results_dir / f"{self.run_date}_report.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        _logger.info("Markdown report written: %s", path)
        return path

    # ── Fear & Greed ───────────────────────────────────────────────────────

    def _load_fear_greed(self) -> Optional[dict]:
        """
        Load the CNN Fear & Greed index value for run_date from the local cache.

        Falls back to the most recent available date if the exact date is absent
        (weekends, holidays).  Returns None on any failure so the rest of the
        report is unaffected.
        """
        if _FearGreedDownloader is None:
            _logger.debug("FearGreedDownloader not available — skipping sentiment overlay")
            return None
        try:
            dl = _FearGreedDownloader()
            df = dl.load()
            if df.empty:
                _logger.warning("Fear & Greed cache is empty — skipping sentiment overlay")
                return None

            ts = pd.Timestamp(self.run_date)
            if ts in df.index:
                row = df.loc[ts]
            else:
                prior = df[df.index <= ts]
                if prior.empty:
                    return None
                row = prior.iloc[-1]

            score = float(row["fear_greed_score"])
            label = str(row["label"])
            date_str = str(pd.Timestamp(row.name).date())  # type: ignore[arg-type]
            _logger.info("Fear & Greed (%s): %.0f — %s", date_str, score, label)
            return {"score": score, "label": label, "date": date_str}
        except Exception:
            _logger.warning("Fear & Greed data unavailable — skipping sentiment overlay")
            return None

    @staticmethod
    def _candidate_table(candidates: List[Candidate]) -> List[str]:
        header = (
            "| Ticker | Score | Tier | Price | RVol | 20d% | SI% | Rev% | Signals |"
        )
        sep = "|--------|-------|------|-------|------|------|-----|------|---------|"
        rows = [header, sep]
        for c in candidates:
            si = f"{c.short_interest_pct_float * 100:.0f}%" if c.short_interest_pct_float else "—"
            rev = f"{c.revenue_growth_yoy * 100:.0f}%" if c.revenue_growth_yoy else "—"
            sig = ", ".join(c.signals[:3]) if c.signals else "—"
            rows.append(
                f"| {c.ticker} | {c.final_score:.1f} | {c.tier} "
                f"| ${c.price:.2f} | {c.relative_volume:.1f}x "
                f"| {c.price_20d_return * 100:.0f}% | {si} | {rev} | {sig} |"
            )
        return rows

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _tier_counts(candidates: List[Candidate]) -> dict:
        counts: dict = {}
        for c in candidates:
            counts[c.tier] = counts.get(c.tier, 0) + 1
        return counts

    def get_alert_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Return candidates that exceed the alert score threshold."""
        return [c for c in candidates if c.final_score >= self.cfg.min_alert_score]
