"""
P20 Kestrel — LLM client.

Wraps the Anthropic SDK with:
- Monthly budget check (hard stop at 120%)
- Cache lookup before calling API
- Structured JSON output with one retry on parse failure
- Cost tracking in k20_llm_runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import anthropic

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import (
    HAIKU_MODEL,
    LLM_MONTHLY_BUDGET_USD,
    SONNET_MODEL,
)

_kestrel = _KestrelService()
get_llm_monthly_spend = _kestrel.get_llm_monthly_spend
get_llm_run_cached = _kestrel.get_llm_run_cached
insert_llm_run = _kestrel.insert_llm_run
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_BUDGET_WARN_PCT = 0.80
_BUDGET_DOSSIER_STOP_PCT = 1.00
_BUDGET_FULL_STOP_PCT = 1.20

# Token pricing (USD per token; rough estimates — verify vs docs.claude.com)
_PRICING: Dict[str, Dict[str, float]] = {
    HAIKU_MODEL: {"in": 0.80e-6, "out": 4.0e-6},
    SONNET_MODEL: {"in": 3.0e-6, "out": 15.0e-6},
}


def _compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pricing = _PRICING.get(model, _PRICING[SONNET_MODEL])
    return round(pricing["in"] * tokens_in + pricing["out"] * tokens_out, 6)


class KestrelLLMClient:
    """Budget-aware LLM client for the P20 pipeline."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def _check_budget(self, task_type: str) -> None:
        """
        Check monthly budget before calling the API.

        Args:
            task_type: The task being requested.

        Raises:
            RuntimeError: If budget limits are exceeded.
        """
        spend = get_llm_monthly_spend()
        pct = spend / LLM_MONTHLY_BUDGET_USD if LLM_MONTHLY_BUDGET_USD else 0

        if pct >= _BUDGET_FULL_STOP_PCT:
            raise RuntimeError(
                f"LLM budget full stop: ${spend:.2f} / ${LLM_MONTHLY_BUDGET_USD:.2f} ({pct:.0%})"
            )

        if pct >= _BUDGET_DOSSIER_STOP_PCT and task_type in ("dossier", "risk_diff", "form10"):
            raise RuntimeError(f"LLM dossier budget stop at {pct:.0%}; classification continues")

        if pct >= _BUDGET_WARN_PCT:
            _logger.warning("LLM budget at %.0f%% ($%.2f / $%.2f)", pct * 100, spend, LLM_MONTHLY_BUDGET_USD)

    def call(
        self,
        task_type: str,
        input_ref: str,
        system_prompt: str,
        user_prompt: str,
        model: str,
        ticker: str | None = None,
        max_tokens: int = 1024,
    ) -> Dict[str, Any] | None:
        """
        Make a cached LLM call with budget enforcement.

        Args:
            task_type: Logical task name (e.g. 'classify_8k', 'dossier').
            input_ref: Cache key suffix (e.g. accession number, ticker+date).
            system_prompt: System message.
            user_prompt: User message (pre-trimmed).
            model: Model ID from config.
            ticker: Associated ticker for audit (optional).
            max_tokens: Max output tokens.

        Returns:
            Parsed JSON dict or None on unrecoverable failure.
        """
        # Cache lookup
        cached = get_llm_run_cached(task_type, input_ref)
        if cached and cached.get("output_json"):
            _logger.debug("LLM cache hit: %s/%s", task_type, input_ref)
            return cached["output_json"]

        self._check_budget(task_type)

        raw_output = self._call_api(system_prompt, user_prompt, model, max_tokens)
        if raw_output is None:
            return None

        # Parse JSON
        output_json, tokens_in, tokens_out, raw_text = raw_output
        if output_json is None:
            # Retry once
            _logger.warning("JSON parse failed; retrying %s/%s", task_type, input_ref)
            raw_output2 = self._call_api(system_prompt, user_prompt, model, max_tokens)
            if raw_output2:
                output_json, tokens_in2, tokens_out2, raw_text = raw_output2
                tokens_in += tokens_in2
                tokens_out += tokens_out2

        cost = _compute_cost(model, tokens_in, tokens_out)
        verdict = output_json.get("verdict") if output_json else None

        insert_llm_run(
            {
                "ticker": ticker,
                "task_type": task_type,
                "input_ref": input_ref,
                "output_json": output_json,
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
                "verdict": verdict,
            }
        )

        return output_json

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
    ) -> tuple | None:
        """
        Make one raw Anthropic API call.

        Returns:
            Tuple of (output_json_or_None, tokens_in, tokens_out, raw_text) or None on error.
        """
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = getattr(response.content[0], "text", "") if response.content else ""
            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens

            output_json: Dict[str, Any] | None = None
            try:
                # Strip markdown code fences if present
                text = raw_text.strip()
                if text.startswith("```"):
                    text = text.split("```", 2)[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.rsplit("```", 1)[0]
                output_json = json.loads(text)
            except json.JSONDecodeError:
                _logger.debug("JSON decode failed on raw output: %.200s", raw_text)

            return output_json, tokens_in, tokens_out, raw_text

        except anthropic.APIError:
            _logger.exception("Anthropic API error")
            return None
