"""
FMP response classification (spec §6.2), shaped by the error bodies the free-key probe recorded (probe F5).

Shared by the production client and the probe; promoted here from ``probe/probe_client.py`` in WP-2.
"""

from __future__ import annotations

from typing import Any

#: Classifications the client retries (spec §6.4). Every other one is final.
RETRYABLE = frozenset({"rate_limited", "server_error", "network_error"})


def classify(status: int, body: Any, text: str) -> str:
    """
    Classify an FMP response.

    Args:
        status: HTTP status code.
        body: Parsed JSON body, or ``None`` if the body is not JSON.
        text: Raw body text.

    Returns:
        One of ``ok``, ``ok_empty``, ``plan_restricted``, ``quota_exhausted``, ``unauthorized``, ``not_found``,
        ``rate_limited``, ``server_error``, ``api_error``.
    """
    lowered = text.lower()
    if "limit reach" in _error_text(status, body, lowered):
        return "quota_exhausted"
    if status == 401:
        return "unauthorized"
    if status in (402, 403) or (status >= 400 and ("subscription" in lowered or "premium" in lowered)):
        return "plan_restricted"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server_error"
    if status >= 400:
        return "api_error"
    if isinstance(body, dict) and ("Error Message" in body or "error" in body):
        message = str(body.get("Error Message", body.get("error", ""))).lower()
        if "subscription" in message or "premium" in message or "plan" in message:
            return "plan_restricted"
        if "invalid api key" in message:
            return "unauthorized"
        return "api_error"
    if body is None:
        return "ok_empty" if not text.strip() else "api_error"
    if body in ([], {}):
        return "ok_empty"
    return "ok"


def _error_text(status: int, body: Any, lowered: str) -> str:
    """
    The lower-cased error message of a response, ``""`` for a data response.

    Only error bodies are searched: data can quote FMP's wording (the 2026-04-30 Altria earnings call transcript says
    "limit reached"), and a quota match on data stops the whole run.
    """
    if isinstance(body, dict) and ("Error Message" in body or "error" in body):
        return str(body.get("Error Message", body.get("error", ""))).lower()
    if status >= 400 or body is None:
        return lowered
    return ""


def row_count(body: Any) -> int:
    """Return the number of records in a parsed body: list length, a wrapped list's length, or 1 for an object."""
    if isinstance(body, list):
        return len(body)
    if isinstance(body, dict):
        for key in ("historical", "data", "results"):
            if isinstance(body.get(key), list):
                return len(body[key])
        return 1 if body else 0
    return 0
