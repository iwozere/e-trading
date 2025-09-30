# src/telegram/services/alerts_eval_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta
import json, math, hashlib

import pandas as pd
import yaml

UTC = timezone.utc
def utcnow() -> datetime: return datetime.now(UTC)

# ---- Repos: your real classes are already in place (create/get/list_by_status/update/etc.) ----
class AlertsRepo:
    def list_active(self, user_id: int | None = None, *, limit: int | None = None, offset: int = 0, older_first: bool = False): ...
    def update(self, alert_id: int, **values) -> bool: ...

# ---- Adapters to your existing facades ------------------------------------------
class MarketDataProviderSelector:
    def get_provider(self, ticker: str): ...

class ProviderAdapter:
    """Make different provider APIs look the same."""
    def __init__(self, selector: MarketDataProviderSelector):
        self.selector = selector

    def get_ohlcv(self, ticker: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        p = self.selector.get_provider(ticker)
        for fn in ("get_ohlcv", "get_ohlcv_df", "fetch_ohlcv"):
            if hasattr(p, fn):
                df = getattr(p, fn)(ticker, timeframe, bars=bars)
                break
        else:
            raise RuntimeError("Provider has no get_ohlcv/fetch_ohlcv method")

        # Canonicalize
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Provider must return a DataFrame with DatetimeIndex")
        cols = {c.lower(): c for c in df.columns}
        rename = {}
        for want in ("open", "high", "low", "close", "volume"):
            if want not in cols:
                # Try typical aliases
                for alias in (want, want[0], "adj_close" if want == "close" else want):
                    if alias in df.columns:
                        rename[alias] = want
                        break
        if rename:
            df = df.rename(columns=rename)
        df = df[["open","high","low","close","volume"]].copy()
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()

class IndicatorEngine:
    def compute(self, ohlcv: pd.DataFrame, spec: Dict[str, Any]) -> pd.Series | pd.DataFrame: ...

def _series_from_engine(ind_engine: IndicatorEngine, df: pd.DataFrame, spec: Dict[str, Any]) -> Optional[pd.Series]:
    out = ind_engine.compute(df, spec)
    if out is None: return None
    if isinstance(out, pd.DataFrame):
        if out.empty: return None
        return out.iloc[:, 0]
    return out

# ---- Evaluator -------------------------------------------------------------------
@dataclass
class AlertsEvalService:
    repo: AlertsRepo
    md_selector: MarketDataProviderSelector
    ind_engine: IndicatorEngine
    logger: Any

    # public entry
    def evaluate_all(self, *, user_id: int | None = None) -> int:
        adapter = ProviderAdapter(self.md_selector)
        alerts = list(self.repo.list_active(user_id))
        self.logger.info("alerts: evaluating %d alert(s)", len(alerts))
        count = 0
        for a in alerts:
            try:
                if self._evaluate_one(a, adapter):
                    count += 1
            except Exception as exc:
                self.logger.exception("alert %s failed", getattr(a, "id", "?"))
                # fallback: mark ERROR if config says so is handled inside _evaluate_one
                self.repo.update(a.id, status="ERROR", last_trigger_condition=str(exc))
        return count

    # core per-alert
    def _evaluate_one(self, alert, adapter: ProviderAdapter) -> bool:
        cfg = self._load_cfg(alert.config_json)
        rule = cfg.get("rule")
        rearm = cfg.get("re_arm")
        options = cfg.get("options", {}) or {}
        notify = cfg.get("notify", {}) or {}

        ticker = cfg["ticker"]; tf = cfg["timeframe"]
        on_error = (options.get("on_error") or "ERROR").upper()
        eval_once = bool(options.get("evaluate_once_per_bar", True))

        # state_json is optional in your current schema; use if present
        state = self._loads(getattr(alert, "state_json", None)) or {}
        sides_state: Dict[str, str] = state.get("sides") or {}

        # Fetch enough bars only for indicators; we only need the LAST CLOSED bar
        lookback = max(200, self._required_lookback(rule, rearm))
        df = adapter.get_ohlcv(ticker, tf, bars=lookback)
        if df is None or df.empty:
            if on_error == "ERROR":
                self.repo.update(alert.id, status="ERROR", last_trigger_condition="No OHLCV")
            return False

        df = self._trim_to_closed_only(df, tf)
        if df is None or len(df) == 0:
            if on_error == "ERROR":
                self.repo.update(alert.id, status="ERROR", last_trigger_condition="No closed bar")
            return False

        last_bar_ts = pd.Timestamp(df.index[-1]).to_pydatetime().astimezone(UTC)

        # once-per-bar guard
        if eval_once and state.get("last_bar_ts") == last_bar_ts.isoformat():
            self.logger.debug("alert %s already processed for %s", alert.id, last_bar_ts)
            return False

        # evaluate rule and rearm USING ONE BAR + stored sides (no prev bar compute)
        triggered, latest_sides, snapshot_rule = self._eval_node(rule, df, sides_state)
        rearmed, latest_sides2, snapshot_rearm = False, {}, None
        if rearm:
            rearmed, latest_sides2, snapshot_rearm = self._eval_node(rearm, df, sides_state)

        # merge sides for persistence
        sides_state.update(latest_sides)
        sides_state.update(latest_sides2)

        # transitions
        changed = False
        now = utcnow()

        if alert.status == "ARMED":
            if triggered:
                new_count = (alert.trigger_count or 0) + 1
                self.repo.update(
                    alert.id,
                    status="TRIGGERED",
                    trigger_count=new_count,
                    last_triggered_at=now,
                    last_trigger_condition=json.dumps(snapshot_rule or {}, ensure_ascii=False),
                )
                changed = True
                # notification is sent by your bot/another layer; or inject a Notifier here if you prefer
        elif alert.status == "TRIGGERED":
            if rearm and rearmed:
                self.repo.update(alert.id, status="ARMED")
                changed = True
        # INACTIVE/ERROR -> ignore

        # persist state if column exists
        state["last_bar_ts"] = last_bar_ts.isoformat()
        state["sides"] = sides_state
        try:
            self.repo.update(alert.id, state_json=json.dumps(state))
        except Exception:
            # column might not exist yet; ignore
            pass

        return changed

    # ----- helpers -----
    def _load_cfg(self, raw: str) -> Dict[str, Any]:
        try:
            return yaml.safe_load(raw)
        except Exception:
            return json.loads(raw)

    def _loads(self, raw: Optional[str]) -> Optional[Dict[str, Any]]:
        if not raw: return None
        try: return json.loads(raw)
        except Exception: return None

    def _trim_to_closed_only(self, df: pd.DataFrame, tf: str) -> Optional[pd.DataFrame]:
        minutes = {"1m":1,"5m":5,"15m":15,"1h":60,"4h":240,"1d":1440}[tf]
        cutoff = utcnow() - timedelta(minutes=minutes/2)
        last_ts = pd.Timestamp(df.index[-1]).to_pydatetime().astimezone(UTC)
        if last_ts > cutoff:
            df = df.iloc[:-1]
        return df if len(df) else None

    def _required_lookback(self, *exprs) -> int:
        # conservative; refine per-indicator later
        def scan(e):
            if not e: return 0
            if "and" in e or "or" in e:
                items = (e.get("and") or []) + (e.get("or") or [])
                return max((scan(x) for x in items), default=0)
            if "not" in e: return scan(e["not"])
            for op in ("gt","gte","lt","lte","eq","ne","between","outside","inside_band","outside_band","crosses_above","crosses_below"):
                if op in e:
                    return max(self._need(e[op].get("lhs")), self._need(e[op].get("rhs")),
                               self._need(e[op].get("value")), self._need(e[op].get("lower")),
                               self._need(e[op].get("upper")))
            return 0
        return max((scan(x) for x in exprs if x), default=0)

    def _need(self, operand) -> int:
        if not isinstance(operand, dict): return 0
        ind = operand.get("indicator")
        if not ind: return 0
        t = (ind.get("type") or "").upper(); p = ind.get("params", {})
        if t in ("SMA","EMA"): return int(p.get("period",50)) + 5
        if t == "RSI": return int(p.get("period",14)) + 5
        if t == "MACD": return int(p.get("slow",26)) + int(p.get("signal",9)) + 10
        return 100

    def _eval_node(self, expr: Dict[str, Any], df: pd.DataFrame, sides_state: Dict[str,str]) -> Tuple[bool, Dict[str,str], Dict[str,float] | None]:
        if expr is None: return False, {}, None
        # logic
        if "and" in expr:
            sides, snap = {}, {}
            for sub in expr["and"]:
                ok, s, sub_snap = self._eval_node(sub, df, sides_state); sides.update(s)
                if sub_snap: snap.update(sub_snap)
                if not ok: return False, sides, snap
            return True, sides, snap
        if "or" in expr:
            sides, snap = {}, {}
            for sub in expr["or"]:
                ok, s, sub_snap = self._eval_node(sub, df, sides_state); sides.update(s)
                if sub_snap: snap.update(sub_snap)
                if ok: return True, sides, snap
            return False, sides, snap
        if "not" in expr:
            ok, s, snap = self._eval_node(expr["not"], df, sides_state)
            return (not ok), s, snap

        # leaves
        for op in ("gt","gte","lt","lte","eq","ne","between","outside","inside_band","outside_band","crosses_above","crosses_below"):
            if op in expr:
                return self._eval_leaf(op, expr[op], df, sides_state)
        raise ValueError(f"Unknown expr node: {expr}")

    def _eval_leaf(self, op: str, node: Dict[str,Any], df: pd.DataFrame, sides_state: Dict[str,str]) -> Tuple[bool, Dict[str,str], Dict[str,float]]:
        def resolve(operand) -> Tuple[Optional[float], Dict[str,float]]:
            if operand is None: return None, {}
            if "value" in operand:
                v = float(operand["value"]); return v, {"value": v}
            if "field" in operand:
                s = df[operand["field"]]; v = s.iloc[-1]
                return (None, {}) if (isinstance(v, float) and math.isnan(v)) else (float(v), {operand["field"]: float(v)})
            if "indicator" in operand:
                series = _series_from_engine(self.ind_engine, df, operand["indicator"])
                if series is None or len(series) == 0: return None, {}
                v = series.iloc[-1]
                return (None, {}) if (isinstance(v, float) and math.isnan(v)) else (float(v), {operand["indicator"].get("output") or operand["indicator"]["type"]: float(v)})
            return None, {}

        sides: Dict[str,str] = {}
        snap: Dict[str,float] = {}

        if op in ("gt","gte","lt","lte","eq","ne"):
            lhs, s1 = resolve(node.get("lhs")); rhs, s2 = resolve(node.get("rhs")); snap.update(s1); snap.update(s2)
            if lhs is None or rhs is None: return False, sides, snap
            return {"gt": lhs>rhs, "gte": lhs>=rhs, "lt": lhs<rhs, "lte": lhs<=rhs, "eq": lhs==rhs, "ne": lhs!=rhs}[op], sides, snap

        if op in ("between","outside","inside_band","outside_band"):
            if op in ("inside_band","outside_band"):
                op = "between" if op == "inside_band" else "outside"
            val, s0 = resolve(node.get("value")); lo, s1 = resolve(node.get("lower")); hi, s2 = resolve(node.get("upper"))
            snap.update(s0); snap.update(s1); snap.update(s2)
            if None in (val, lo, hi): return False, sides, snap
            inside = (lo <= val <= hi)
            return (inside if op == "between" else not inside), sides, snap

        if op in ("crosses_above","crosses_below"):
            lhs, s1 = resolve(node.get("lhs")); rhs, s2 = resolve(node.get("rhs")); snap.update(s1); snap.update(s2)
            if lhs is None or rhs is None: return False, sides, snap
            key = self._cross_key(node)
            prev_side = sides_state.get(key)
            curr_side = "above" if lhs > rhs else ("below" if lhs < rhs else (prev_side or "equal"))
            sides[key] = curr_side

            if prev_side is None:
                return False, sides, snap  # first observation, don't trigger
            if op == "crosses_above":
                return (prev_side in ("below", "equal") and curr_side == "above"), sides, snap
            else:
                return (prev_side in ("above", "equal") and curr_side == "below"), sides, snap

        raise ValueError(f"Unsupported op {op}")

    def _cross_key(self, node: Dict[str,Any]) -> str:
        payload = json.dumps(node, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()
