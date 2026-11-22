"""
Survivor-bias study — P24 falling knife on the EODHD store (docs/Design.md, "Strategies").

P24's own runner opens the FMP vintage in its workers; this one opens the EODHD store and calls P24's ``run_security``
unchanged, with the frozen live parameters and every signal date from 2000 on (period labels are replaced by the
study's periods afterwards).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import pandas as pd

from src.data.eodhd_pipeline import config as eodhd_config
from src.data.eodhd_pipeline.store import EodhdStore
from src.ml.pipeline.p24_falling_knife.backtest import RunSpec, run_security, trades_frame
from src.ml.pipeline.p24_falling_knife.config import DAILY, P24Config, live_params
from src.ml.pipeline.p24_falling_knife.data import load_daily_batch
from src.ml.pipeline.p24_falling_knife.universe import Security
from src.ml.research.survivor_bias import config
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def knife_config() -> P24Config:
    """
    P24 configuration for the study: frozen live daily parameters, no random control, signals from 2000.

    Returns:
        Configuration whose ``upper_bound`` window starts in 2000 (labels are not used by the study).
    """
    base = P24Config()
    window = replace(base.window, upper_bound_start=config.P24_SIGNALS_FROM, data_start=config.DATA_START)
    return replace(base, daily=live_params(DAILY), window=window, random_control_draws=0)


_STORE: Optional[EodhdStore] = None


def _init_worker(data_root: str) -> None:
    global _STORE
    _STORE = EodhdStore(Path(data_root))


def _run_group(securities: Sequence[Security], spec: RunSpec) -> List[Dict[str, object]]:
    if _STORE is None:
        raise RuntimeError("worker store not initialised")
    end = spec.as_of
    if end is None:
        raise RuntimeError("run spec has no as_of date")
    daily = load_daily_batch(cast(Any, _STORE), [s.security_id for s in securities], config.DATA_START, end)
    rows: List[Dict[str, object]] = []
    for security in securities:
        frame = daily.get(security.security_id)
        if frame is None:
            continue
        try:
            part = run_security(security, frame, spec, rng_seed=spec.config.seed * 1_000_003 + security.security_id)
        except (ValueError, IndexError, KeyError):
            _logger.exception("P24 run failed for security %s (%s)", security.security_id, security.symbol)
            continue
        rows.extend(part.trades)
    return rows


def run_knife(store: EodhdStore, securities: Sequence[Security], workers: Optional[int] = None) -> pd.DataFrame:
    """
    P24 daily trades of the ``full`` and ``drop_only`` variants on the EODHD store.

    Args:
        store: Open EODHD store.
        securities: Stocks to run (P24 ``Security`` rows of the store).
        workers: Process count (default CPU count − 1); 1 runs in-process.

    Returns:
        Trade frame as P24 writes it (``delisted_in_trade`` marks trades that end with the delisting).
    """
    spec = RunSpec(config=knife_config(), timeframes=(DAILY,), variants=config.P24_VARIANTS, include_holdout=True,
                   as_of=store.as_of)
    buckets: Dict[int, List[Security]] = {}
    for security in securities:
        buckets.setdefault(security.security_id % eodhd_config.PRICE_BUCKETS, []).append(security)
    groups = [buckets[k] for k in sorted(buckets)]
    rows: List[Dict[str, object]] = []
    n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    if n_workers == 1:
        _init_worker(str(store.data_root))
        for group in groups:
            rows.extend(_run_group(group, spec))
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(str(store.data_root),)) as pool:
            futures = [pool.submit(_run_group, group, spec) for group in groups]
            for done, future in enumerate(as_completed(futures), 1):
                rows.extend(future.result())
                if done % 8 == 0 or done == len(futures):
                    _logger.info("P24 on EODHD: %d of %d buckets done, %d trades", done, len(futures), len(rows))
    return trades_frame(rows)
