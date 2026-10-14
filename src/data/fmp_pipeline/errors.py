"""Run-stopping conditions shared by the FMP client and the downloader (spec §6.2)."""

from __future__ import annotations


class QuotaExhausted(Exception):
    """FMP reported that the account's call or bandwidth quota is used up. Stop at once; never retry in a loop."""


class AuthAbort(Exception):
    """Every recent request was rejected with 401/403: the key is revoked or the plan lapsed. Abort the run."""


class BandwidthCapReached(Exception):
    """The rolling-window bandwidth used has reached the configured safety fraction of the cap (spec §6.3)."""
