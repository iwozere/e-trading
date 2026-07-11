"""Stage 3 — LLM Synthesis: top-25 → top-5 picks with full exit strategies."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, cast

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p05_ai_selector.config import LLM_MAX_TOKENS, LLM_MODEL
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SYSTEM_PROMPT = """You are a quantitative equity and crypto analyst. You receive a ranked list of stock
and crypto candidates with their quantitative signal summaries. Your tasks are:

1. Select the top 5 most actionable picks for the next 1–12 months (time horizon
   varies by setup quality), ranked by conviction.

2. For each pick, write a concise thesis and a complete position management guide:
   - Specific price levels for adding to the position
   - Catalyst-based hold conditions (what must remain true to stay in)
   - Thesis-breakers: concrete, falsifiable events that trigger an immediate exit
   - Profit-taking levels with partial trim percentages and price targets
   - A brief time-horizon note explaining why patience is the edge for this setup

3. Write a brief market context paragraph.

Guidelines:
- Favour setups with signal confluence; be sceptical of single-signal stories.
- For crypto: ignore traditional fundamentals; focus on momentum, volume, and
  institutional signals only.
- Earnings within 7 days = binary risk event; note it explicitly in risk_factors.
- Price levels must be specific (e.g. "$44–46") not vague (e.g. "near support").
- Profit-taking levels should reflect realistic upside from the current price.
- thesis_breakers must be concrete falsifiable events, not market platitudes.

Return ONLY the structured JSON defined in the tool schema."""

_TOOL_SCHEMA = {
    "name": "submit_picks",
    "description": "Submit the top-5 AI-selected picks with full exit strategies.",
    "input_schema": {
        "type": "object",
        "properties": {
            "picks": {
                "type": "array",
                "description": "Exactly 5 ranked picks.",
                "items": {
                    "type": "object",
                    "required": [
                        "rank",
                        "ticker",
                        "confidence",
                        "bias",
                        "thesis",
                        "risk_factors",
                        "time_horizon",
                        "exit_strategy",
                    ],
                    "properties": {
                        "rank": {"type": "integer"},
                        "ticker": {"type": "string"},
                        "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
                        "bias": {"type": "string", "enum": ["long", "short", "neutral"]},
                        "thesis": {"type": "string"},
                        "risk_factors": {"type": "array", "items": {"type": "string"}},
                        "time_horizon": {"type": "string"},
                        "exit_strategy": {
                            "type": "object",
                            "required": [
                                "add_conditions",
                                "hold_conditions",
                                "thesis_breakers",
                                "profit_targets",
                                "time_horizon_note",
                            ],
                            "properties": {
                                "add_conditions": {"type": "array", "items": {"type": "string"}},
                                "hold_conditions": {"type": "array", "items": {"type": "string"}},
                                "thesis_breakers": {"type": "array", "items": {"type": "string"}},
                                "profit_targets": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "price_level": {"type": "number"},
                                            "action": {"type": "string"},
                                            "note": {"type": "string"},
                                        },
                                    },
                                },
                                "time_horizon_note": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "market_context": {"type": "string"},
            "notification_override": {
                "type": "boolean",
                "description": "Set true if any pick has confidence >= 9.",
            },
        },
        "required": ["picks", "market_context", "notification_override"],
    },
}


class Stage3LLMSynthesizer:
    """
    Calls Claude via Anthropic SDK to synthesise the top-5 picks and exit strategies.

    Single API call: all 25 data packets in one prompt.
    Uses tool_use (forced) so the response is always structured JSON.
    """

    def __init__(self, api_key: str = "", model: str = LLM_MODEL):
        from src.config.provider_config import get_api_key

        self._api_key = api_key or get_api_key("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY is required. Set it in the environment or config.")
        self._model = model

    def run(self, stage2_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Build data packets from Stage 2 output, call Claude, parse and return results.

        Args:
            stage2_df: Stage 2 output DataFrame (top-25 rows).

        Returns:
            Dict with keys: picks, market_context, notification_override, tokens_used.

        Raises:
            ValueError: If the response is malformed.
            Exception: On Anthropic API errors (caller should handle).
        """
        packets = self._build_data_packets(stage2_df)
        raw = self._call_claude(packets)
        return self._parse_response(raw)

    def _build_data_packets(self, stage2_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert Stage 2 rows to the LLM input schema defined in spec §7.1."""
        packets = []
        for _, row in stage2_df.iterrows():
            try:
                breakdown = json.loads(str(row.get("signal_breakdown", "{}")))
            except (json.JSONDecodeError, TypeError):
                breakdown = {}

            fund_bd = breakdown.get("fundamentals", {})
            p18_bd = breakdown.get("p18", {})

            earnings_date = str(row.get("earnings_date", ""))
            earnings_in_days: Any = None
            if earnings_date:
                try:
                    from datetime import date

                    ed = date.fromisoformat(earnings_date)
                    earnings_in_days = (ed - date.today()).days
                except ValueError:
                    pass

            packet: Dict[str, Any] = {
                "ticker": str(row["ticker"]),
                "name": str(row.get("name", row["ticker"])),
                "sector": str(row.get("sector", "")),
                "asset_type": str(row.get("asset_type", "equity")),
                "price": float(row.get("last_price", 0)),
                "market_cap_b": float(row.get("market_cap_b", 0)),
                "technicals": {
                    "rsi_14": breakdown.get("rsi_14"),
                    "sma20_above_sma50": breakdown.get("sma20_above_sma50"),
                    "volume_surge_ratio": breakdown.get("volume_surge_ratio"),
                    "momentum_5d_pct": breakdown.get("momentum_5d_pct"),
                    "atr_compression": breakdown.get("atr_compression"),
                    "pct_from_52w_high": breakdown.get("pct_from_52w_high"),
                    "pct_from_52w_low": breakdown.get("pct_from_52w_low"),
                },
                "fundamentals": {
                    "pe_ratio": fund_bd.get("pe_ratio"),
                    "profit_margin_pct": (
                        float(fund_bd["profit_margin"]) * 100 if fund_bd.get("profit_margin") is not None else None
                    ),
                    "debt_to_equity": fund_bd.get("debt_to_equity"),
                    "revenue_yoy_pct": (
                        float(fund_bd["revenue_growth"]) * 100 if fund_bd.get("revenue_growth") is not None else None
                    ),
                    "dividend_yield_pct": fund_bd.get("dividend_yield"),
                    "available": bool(row.get("fundamentals_available", False)),
                },
                "institutional_flow": {
                    "p18_score": int(row.get("p18_score", 0)),
                    "signals_active": [k for k, v in p18_bd.items() if v is True],
                    "institution_count": 0,
                    "form4_insider_buy": p18_bd.get("form4_insider_buy", False),
                },
                "contextual": {
                    "earnings_in_days": earnings_in_days,
                    "earnings_date": earnings_date or None,
                },
                "deterministic_score": float(row.get("total_score", 0)),
            }
            packets.append(packet)
        return packets

    def _call_claude(self, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send all 25 packets to Claude in a single API call using tool_use."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        _logger.info("Stage3: calling Claude %s with %d candidates", self._model, len(packets))

        response = client.messages.create(
            model=self._model,
            max_tokens=LLM_MAX_TOKENS,
            tools=cast(Any, [_TOOL_SCHEMA]),
            tool_choice={"type": "tool", "name": "submit_picks"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(packets, indent=2)}],
        )

        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        _logger.info("Stage3: Claude response received — %d tokens used", tokens_used)

        # A truncated response (hit max_tokens) leaves the forced tool_use block with an
        # incomplete/empty input. Surface this explicitly rather than failing later with a
        # confusing "picks is empty" error.
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "max_tokens":
            raise ValueError(
                f"Claude response truncated at max_tokens ({LLM_MAX_TOKENS}); "
                f"tool_use input is incomplete. Increase LLM_MAX_TOKENS."
            )

        # The tool_use block is not guaranteed to be content[0] (the model may emit a
        # leading text block); select it by type.
        tool_block = next(
            (block for block in response.content if getattr(block, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            raise ValueError(f"No tool_use block in Claude response (stop_reason={stop_reason}).")
        return {"tool_input": tool_block.input, "tokens_used": tokens_used}  # type: ignore[attr-defined]

    def _parse_response(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and extract the structured pick data from Claude's tool response.

        Raises:
            ValueError: If the required structure is missing.
        """
        tool_input = raw.get("tool_input", {})
        picks = tool_input.get("picks")
        if not picks or not isinstance(picks, list) or len(picks) == 0:
            raise ValueError(f"LLM response missing 'picks' or picks is empty. Raw tool_input: {tool_input}")

        for i, pick in enumerate(picks):
            for required_key in ("rank", "ticker", "confidence", "bias", "thesis", "exit_strategy"):
                if required_key not in pick:
                    raise ValueError(f"Pick #{i} missing required field '{required_key}'. Pick: {pick}")

        notification_override = bool(tool_input.get("notification_override", False))
        # Also check if any pick has confidence >= 9 as a safety net
        if any(int(p.get("confidence", 0)) >= 9 for p in picks):
            notification_override = True

        return {
            "picks": picks,
            "market_context": str(tool_input.get("market_context", "")),
            "notification_override": notification_override,
            "tokens_used": raw.get("tokens_used", 0),
        }
