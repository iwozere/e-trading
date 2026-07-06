"""Pure signal computation for trading-strategy-pack (Strategies 1–6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List

import pandas as pd

from src.data.data_manager import DataManager
from src.notification.logger import setup_logger
from src.strategy_pack.indicators import atr_rma, ema, supertrend_line_and_direction
from src.strategy_pack.models import PackSignal, utc_now_iso

_logger = setup_logger(__name__)

STRATEGY_IDS = {
    1: "SP-1",
    2: "SP-2",
    3: "SP-3",
    4: "SP-4",
    5: "SP-5",
    6: "SP-6",
}


@dataclass
class RunContext:
    dm: DataManager
    end: datetime
    config: Dict[str, Any]
    variant: str = "A"


def _window_start(end: datetime, lookback_days: int) -> datetime:
    return end - timedelta(days=lookback_days)


def _load_df(dm: DataManager, symbol: str, timeframe: str, end: datetime, lookback_days: int) -> pd.DataFrame:
    start = _window_start(end, lookback_days)
    df = dm.get_ohlcv(symbol, timeframe, start_date=start, end_date=end, force_refresh=False)
    if df is None or df.empty:
        _logger.warning("No OHLCV for %s %s", symbol, timeframe)
        return pd.DataFrame()
    df = df.sort_index().dropna(subset=["open", "high", "low", "close"], how="any")
    return df


def _bar_close_ts(idx: pd.Timestamp) -> str:
    if hasattr(idx, "to_pydatetime"):
        return idx.isoformat()
    return str(idx)


def _float_or_none(x: Any) -> Any:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except TypeError:
        return x
    return float(x)


def run_strategy_1(ctx: RunContext) -> List[PackSignal]:
    cfg = ctx.config.get("strategy_1", {})
    tf = cfg.get("timeframe", "1d")
    lookback = int(cfg.get("lookback_days", 400))
    top_k = int(cfg.get("top_k", 10))
    m_long = int(cfg.get("momentum_long_days", 126))
    m_short = int(cfg.get("momentum_short_days", 63))
    symbols: List[str] = list(cfg.get("symbols", []))

    rows = []
    latest_bar = utc_now_iso()
    for sym in symbols:
        df = _load_df(ctx.dm, sym, tf, ctx.end, lookback)
        if len(df) < m_long + 5:
            continue
        latest_bar = _bar_close_ts(df.index[-1])
        c = df["close"]
        mom6 = c.iloc[-1] / c.iloc[-1 - m_long] - 1.0
        mom3 = c.iloc[-1] / c.iloc[-1 - m_short] - 1.0 if len(c) > m_short + 1 else mom6
        vol = c.pct_change().rolling(20).std()
        vol_s = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else 1e-6
        score = mom6 / vol_s if cfg.get("use_vol_scaled") else mom6
        rows.append({"symbol": sym, "mom6": float(mom6), "mom3": float(mom3), "score": float(score)})

    if not rows:
        return []

    rows.sort(key=lambda r: r["score"], reverse=True)
    picked = rows[:top_k]
    weight = 1.0 / len(picked) if picked else 0.0
    targets = {r["symbol"]: weight for r in picked}
    ranks = {r["symbol"]: r["score"] for r in rows}

    return [
        PackSignal(
            strategy_id=STRATEGY_IDS[1],
            variant=ctx.variant,
            symbol="__PORTFOLIO__",
            signal="REBALANCE",
            bar_timeframe=tf,
            bar_close_ts=latest_bar,
            price=0.0,
            reason_code="momentum_top_k",
            metadata={
                "top_k": top_k,
                "targets": targets,
                "ranks_score": ranks,
                "momentum_window_days": m_long,
            },
            notify_recommended=True,
        )
    ]


def run_strategy_2(ctx: RunContext) -> List[PackSignal]:
    cfg = ctx.config.get("strategy_2", {})
    tf = cfg.get("timeframe", "1d")
    lookback = int(cfg.get("lookback_days", 400))
    slow = int(cfg.get("sma_slow", 200))
    fast = int(cfg.get("sma_fast", 50))
    use_fast = bool(cfg.get("use_sma_fast_confirm"))
    symbols: List[str] = list(cfg.get("symbols", []))
    out: List[PackSignal] = []

    for sym in symbols:
        df = _load_df(ctx.dm, sym, tf, ctx.end, lookback)
        if len(df) < slow + 2:
            continue
        c = df["close"]
        sma_s = c.rolling(slow).mean()
        sma_f = c.rolling(fast).mean() if use_fast else None
        uptrend = c > sma_s
        if use_fast and sma_f is not None:
            uptrend = uptrend & (c > sma_f)

        prev_up = uptrend.iloc[-2]
        curr_up = uptrend.iloc[-1]
        cross_up = curr_up and not prev_up
        cross_down = (not curr_up) and prev_up

        bar_ix = df.index[-1]
        price = float(c.iloc[-1])
        sig = "STATUS"
        reason = "trend_hold" if curr_up else "trend_flat"
        notify = False
        if cross_up:
            sig, reason, notify = "BUY", "cross_above_sma_slow", True
        elif cross_down:
            sig, reason, notify = "SELL", "cross_below_sma_slow", True

        out.append(
            PackSignal(
                strategy_id=STRATEGY_IDS[2],
                variant=ctx.variant,
                symbol=sym,
                signal=sig,
                bar_timeframe=tf,
                bar_close_ts=_bar_close_ts(bar_ix),
                price=price,
                reason_code=reason,
                metadata={
                    "sma_slow": _float_or_none(sma_s.iloc[-1]),
                    "in_uptrend": curr_up,
                    "cross_up": cross_up,
                    "cross_down": cross_down,
                },
                notify_recommended=notify,
            )
        )
    return out


def run_strategy_3(ctx: RunContext) -> List[PackSignal]:
    cfg = ctx.config.get("strategy_3", {})
    tf = cfg.get("timeframe", "1d")
    lookback = int(cfg.get("lookback_days", 800))
    w_sma = int(cfg.get("weekly_sma", 52))
    w_fast = int(cfg.get("weekly_sma_fast", 26))
    symbols: List[str] = list(cfg.get("symbols", []))
    out: List[PackSignal] = []

    for sym in symbols:
        d = _load_df(ctx.dm, sym, tf, ctx.end, lookback)
        if len(d) < w_sma * 5:
            continue
        w = d.resample("W-SUN").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        w = w.dropna(how="all")
        c = w["close"]
        if len(c) < w_sma + 2:
            continue
        sma = c.rolling(w_sma).mean()
        uptrend = c > sma
        prev_up = uptrend.iloc[-2]
        curr_up = uptrend.iloc[-1]
        cross_up = curr_up and not prev_up
        cross_down = (not curr_up) and prev_up

        bar_ix = w.index[-1]
        price = float(c.iloc[-1])
        sig = "STATUS"
        reason = "weekly_uptrend" if curr_up else "weekly_flat"
        notify = False
        if cross_up:
            sig, reason, notify = "BUY", "weekly_cross_above_sma", True
        elif cross_down:
            sig, reason, notify = "SELL", "weekly_cross_below_sma", True

        out.append(
            PackSignal(
                strategy_id=STRATEGY_IDS[3],
                variant=ctx.variant,
                symbol=sym,
                signal=sig,
                bar_timeframe="1w",
                bar_close_ts=_bar_close_ts(bar_ix),
                price=price,
                reason_code=reason,
                metadata={
                    "weekly_sma": float(sma.iloc[-1]),
                    "weekly_sma_fast": float(c.rolling(w_fast).mean().iloc[-1]) if len(c) >= w_fast else None,
                    "in_uptrend": curr_up,
                },
                notify_recommended=notify,
            )
        )
    return out


def run_strategy_4(ctx: RunContext) -> List[PackSignal]:
    cfg = ctx.config.get("strategy_4", {})
    tf = cfg.get("timeframe", "1d")
    lookback = int(cfg.get("lookback_days", 260))
    targets: Dict[str, float] = dict(cfg.get("targets", {}))
    sma_n = int(cfg.get("sma_trend_filter", 200))
    skip_add = bool(cfg.get("skip_add_if_below_sma", True))
    current: Dict[str, float] = dict(cfg.get("current_weights", {}))

    flags: Dict[str, Any] = {}
    last_prices: Dict[str, float] = {}
    for sym, w_tgt in targets.items():
        df = _load_df(ctx.dm, sym, tf, ctx.end, lookback)
        if df.empty or len(df) < sma_n + 1:
            flags[sym] = "no_data"
            continue
        c = df["close"]
        sma = c.rolling(sma_n).mean()
        last = float(c.iloc[-1])
        last_prices[sym] = last
        below = last < float(sma.iloc[-1])
        flags[sym] = {"below_sma_filter": below, "skip_new_adds": skip_add and below}

    meta = {
        "targets": targets,
        "current_weights_assumed": current,
        "per_symbol": flags,
        "last_prices": last_prices,
    }
    return [
        PackSignal(
            strategy_id=STRATEGY_IDS[4],
            variant=ctx.variant,
            symbol="__PORTFOLIO__",
            signal="REBALANCE",
            bar_timeframe=tf,
            bar_close_ts=utc_now_iso(),
            price=0.0,
            reason_code="targets_vs_policy",
            metadata=meta,
            notify_recommended=True,
        )
    ]


def run_strategy_5(ctx: RunContext) -> List[PackSignal]:
    cfg = ctx.config.get("strategy_5", {})
    sym = str(cfg.get("symbol", "BTCUSDT"))
    tf = str(cfg.get("timeframe", "4h"))
    lookback = int(cfg.get("lookback_days", 120))
    n = int(cfg.get("donchian", 20))
    vol_ma = int(cfg.get("volume_ma", 20))
    atr_p = int(cfg.get("atr_period", 14))
    rr = float(cfg.get("risk_reward", 1.5))

    df = _load_df(ctx.dm, sym, tf, ctx.end, lookback)
    if len(df) < max(n, vol_ma, atr_p) + 3:
        return []

    variant = (ctx.variant or "A").upper()
    bar_ix = df.index[-1]
    price = float(df["close"].iloc[-1])

    if variant == "A":
        prev_high_max = df["high"].shift(1).rolling(n).max().iloc[-1]
        prev_low_min = df["low"].shift(1).rolling(n).min().iloc[-1]
        vol_ok = float(df["volume"].iloc[-1]) > float(df["volume"].rolling(vol_ma).mean().iloc[-1])
        breakout = price > float(prev_high_max) and vol_ok
        atrv = float(atr_rma(df["high"], df["low"], df["close"], atr_p).iloc[-1])
        sl = float(prev_low_min)
        risk = max(price - sl, 1e-8)
        tp = price + rr * risk
        sig = "BUY" if breakout else "STATUS"
        notify = breakout
        return [
            PackSignal(
                strategy_id=STRATEGY_IDS[5],
                variant="A",
                symbol=sym,
                signal=sig,
                bar_timeframe=tf,
                bar_close_ts=_bar_close_ts(bar_ix),
                price=price,
                reason_code="donchian_volume_breakout" if breakout else "no_breakout",
                metadata={
                    "donchian_high_prev": float(prev_high_max),
                    "donchian_low_prev": float(prev_low_min),
                    "stop_suggest": sl,
                    "take_profit_suggest": tp,
                    "volume_ok": vol_ok,
                },
                notify_recommended=notify,
            )
        ]

    if variant == "B":
        c = df["close"]
        sma = c.rolling(20).mean()
        rising = sma.iloc[-1] > sma.iloc[-5]
        dist = (c.iloc[-1] - sma.iloc[-1]).abs() / max(sma.iloc[-1], 1e-12)
        pullback = dist <= 0.01 and c.iloc[-1] > sma.iloc[-1] and rising
        exit_sig = c.iloc[-1] < sma.iloc[-1]
        sig = "BUY" if pullback else ("SELL" if exit_sig else "STATUS")
        notify = sig in ("BUY", "SELL")
        return [
            PackSignal(
                strategy_id=STRATEGY_IDS[5],
                variant="B",
                symbol=sym,
                signal=sig,
                bar_timeframe=tf,
                bar_close_ts=_bar_close_ts(bar_ix),
                price=price,
                reason_code="sma20_pullback" if pullback else ("below_sma20" if exit_sig else "hold"),
                metadata={"sma20": float(sma.iloc[-1]), "sma_rising": bool(rising)},
                notify_recommended=notify,
            )
        ]

    # Variant C — EMA(9) vs EMA(21)
    e9 = ema(df["close"], 9)
    e21 = ema(df["close"], 21)
    rising = e9.iloc[-1] > e9.iloc[-3]
    long_ok = e9.iloc[-1] > e21.iloc[-1] and rising
    prev_long = e9.iloc[-2] > e21.iloc[-2]
    cross_up = long_ok and not prev_long
    cross_dn = e9.iloc[-1] < e21.iloc[-1] and prev_long
    sig = "BUY" if cross_up else ("SELL" if cross_dn else "STATUS")
    notify = sig in ("BUY", "SELL")
    return [
        PackSignal(
            strategy_id=STRATEGY_IDS[5],
            variant="C",
            symbol=sym,
            signal=sig,
            bar_timeframe=tf,
            bar_close_ts=_bar_close_ts(bar_ix),
            price=price,
            reason_code="ema9_21_cross",
            metadata={"ema9": float(e9.iloc[-1]), "ema21": float(e21.iloc[-1])},
            notify_recommended=notify,
        )
    ]


def run_strategy_6(ctx: RunContext) -> List[PackSignal]:
    cfg = ctx.config.get("strategy_6", {})
    sym = str(cfg.get("symbol", "BTCUSDT"))
    tf = str(cfg.get("timeframe", "4h"))
    lookback = int(cfg.get("lookback_days", 400))
    ema_p = int(cfg.get("ema_period", 50))
    st_atr = int(cfg.get("supertrend_atr_period", 10))
    st_mult = float(cfg.get("supertrend_multiplier", 3.0))
    long_only = bool(cfg.get("long_only", True))
    atr_stop_p = int(cfg.get("atr_stop_period", 14))
    atr_stop_m = float(cfg.get("atr_stop_multiplier", 2.0))

    df = _load_df(ctx.dm, sym, tf, ctx.end, lookback).copy()
    if len(df) < max(ema_p, st_atr, atr_stop_p) + 5:
        return []

    c = df["close"]
    e = ema(c, ema_p)
    st_line, direction = supertrend_line_and_direction(df, st_atr, st_mult)
    atr_stop = atr_rma(df["high"], df["low"], df["close"], atr_stop_p)

    i = -1
    long_cond = float(c.iloc[i]) > float(e.iloc[i]) and int(direction.iloc[i]) == 1
    short_cond = float(c.iloc[i]) < float(e.iloc[i]) and int(direction.iloc[i]) == -1
    prev_long = float(c.iloc[i - 1]) > float(e.iloc[i - 1]) and int(direction.iloc[i - 1]) == 1
    prev_short = float(c.iloc[i - 1]) < float(e.iloc[i - 1]) and int(direction.iloc[i - 1]) == -1

    sig = "STATUS"
    reason = "flat_or_hold"
    notify = False
    price = float(c.iloc[i])
    bar_ix = df.index[i]
    st_val = float(st_line.iloc[i]) if pd.notna(st_line.iloc[i]) else price
    sl_long = price - atr_stop_m * float(atr_stop.iloc[i])

    if long_cond and not prev_long:
        sig, reason, notify = "LONG", "ema_and_supertrend_long", True
    elif prev_long and not long_cond:
        sig, reason, notify = "EXIT", "long_conditions_broken", True
    elif not long_only and short_cond and not prev_short:
        sig, reason, notify = "SHORT", "ema_and_supertrend_short", True
    elif not long_only and prev_short and not short_cond:
        sig, reason, notify = "EXIT", "short_conditions_broken", True

    return [
        PackSignal(
            strategy_id=STRATEGY_IDS[6],
            variant=ctx.variant,
            symbol=sym,
            signal=sig,
            bar_timeframe=tf,
            bar_close_ts=_bar_close_ts(bar_ix),
            price=price,
            reason_code=reason,
            metadata={
                "ema": float(e.iloc[i]),
                "supertrend": st_val,
                "direction": int(direction.iloc[i]),
                "atr_stop_suggest": sl_long if long_cond or prev_long else None,
                "long_only": long_only,
            },
            notify_recommended=notify,
        )
    ]


RUNNERS: Dict[int, Callable[[RunContext], List[PackSignal]]] = {
    1: run_strategy_1,
    2: run_strategy_2,
    3: run_strategy_3,
    4: run_strategy_4,
    5: run_strategy_5,
    6: run_strategy_6,
}
