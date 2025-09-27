# src/telegram/screener/services/alerts_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Iterable
from datetime import datetime, timezone, timedelta
import math

import pandas as pd

# Expect your project logger
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


# --- Contracts (replace with your real imports) --------------------------------

class AlertsRepo:
    def list_active(self, session) -> Iterable[Any]: ...
    def update_status(self, session, alert_id: int, status: str, error_message: Optional[str] = None) -> None: ...
    def save_state(self, session, alert_id: int, state: Dict[str, Any]) -> None: ...
    def record_event(self, session, alert_id: int, event_type: str, payload: Dict[str, Any]) -> None: ...

class MarketDataProviderSelector:
    def get_provider(self, ticker: str): ...

class Notifier:
    def send_telegram(self, user_id: int, text: str) -> None: ...
    def send_email(self, email: str, subject: str, body: str) -> None: ...

class IndicatorEngine:
    """TA-Lib first, fallback to Backtrader. Implement outside."""
    def compute(self, ohlcv: pd.DataFrame, spec: Dict[str, Any]) -> pd.Series | pd.DataFrame: ...

# --- Service -------------------------------------------------------------------

@dataclass
class AlertsService:
    repo: AlertsRepo
    md_selector: MarketDataProviderSelector
    notifier: Notifier
    ind_engine: IndicatorEngine

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    # Entry point for background job every 15 minutes
    def evaluate_all_alerts(self, session) -> None:
        alerts = list(self.repo.list_active(session))
        logger.info("alerts_service: evaluating %d alerts", len(alerts))

        for alert in alerts:
            try:
                self._evaluate_one(session, alert)
            except Exception as exc:
                logger.exception("Alert %s evaluation failed", getattr(alert, "id", "?"))
                try:
                    self.repo.update_status(session, alert.id, "ERROR", error_message=str(exc))
                finally:
                    # Continue with others
                    continue

    # --- Core evaluation -------------------------------------------------------

    def _evaluate_one(self, session, alert) -> None:
        """
        Evaluate business logic on the last COMPLETED bar.
        Status transitions:
          ARMED --(rule True)--> TRIGGERED  [send notif]
          TRIGGERED --(re_arm True)--> ARMED
        """
        now = self._utcnow()

        cfg = alert.config_yaml or alert.config_json  # however you store it; parse to dict
        if isinstance(cfg, str):
            import yaml, json
            try:
                cfg = yaml.safe_load(cfg)
            except Exception:
                cfg = json.loads(cfg)

        rule = cfg.get("rule")
        re_arm = cfg.get("re_arm")
        ticker = cfg["ticker"]
        tf = cfg["timeframe"]
        options = cfg.get("options", {})
        eval_once = options.get("evaluate_once_per_bar", True)
        on_error = options.get("on_error", "ERROR")

        # Load prior state if any
        state: Dict[str, Any] = getattr(alert, "state_json", None) or {}

        # 1) Fetch data (native TF), require last CLOSED bar
        provider = self.md_selector.get_provider(ticker)
        lookback = self._required_lookback(rule, re_arm)
        ohlcv = provider.get_ohlcv(ticker, tf, lookback)
        if ohlcv is None or len(ohlcv) == 0:
            self._handle_no_data(session, alert, on_error, "No OHLCV")
            return

        ohlcv = self._trim_to_closed_bar(ohlcv, now, tf)
        if ohlcv is None or len(ohlcv) == 0:
            self._handle_no_data(session, alert, on_error, "No closed bar")
            return

        last_bar_ts = pd.Timestamp(ohlcv.index[-1]).to_pydatetime().replace(tzinfo=timezone.utc)

        # 2) Once-per-bar guard
        if eval_once and state.get("last_bar_ts") == last_bar_ts.isoformat():
            logger.debug("Alert %s already evaluated for bar %s", alert.id, last_bar_ts)
            return

        # 3) Evaluate rule / rearm
        rule_val, sides = self._eval_expr(rule, ohlcv, state)
        rearm_val, sides2 = (False, {}) if re_arm is None else self._eval_expr(re_arm, ohlcv, state)

        sides.update(sides2)

        # 4) Transitions
        if alert.status == "ARMED":
            if rule_val:
                self._notify(session, alert, cfg)
                self.repo.update_status(session, alert.id, "TRIGGERED")
                state["last_triggered_at"] = now.isoformat()
        elif alert.status == "TRIGGERED":
            if re_arm and rearm_val:
                self.repo.update_status(session, alert.id, "ARMED")
        elif alert.status in ("INACTIVE", "ERROR"):
            # ignore
            pass

        # 5) Persist state
        state.update({
            "last_eval_at": now.isoformat(),
            "last_bar_ts": last_bar_ts.isoformat(),
            "last_rule_value": bool(rule_val),
            "last_rearm_value": bool(rearm_val),
            "sides": sides,
        })
        self.repo.save_state(session, alert.id, state)
        self.repo.record_event(session, alert.id, "eval", {
            "bar_ts": state["last_bar_ts"], "rule": rule_val, "rearm": rearm_val
        })

    # --- Helpers ----------------------------------------------------------------

    def _handle_no_data(self, session, alert, on_error: str, reason: str) -> None:
        if on_error == "ERROR":
            self.repo.update_status(session, alert.id, "ERROR", error_message=reason)
        logger.warning("Alert %s: %s", alert.id, reason)

    def _trim_to_closed_bar(self, df: pd.DataFrame, now: datetime, tf: str) -> Optional[pd.DataFrame]:
        """Drop a possibly in-flight bar; keep only fully closed bars."""
        # Simplistic TF duration mapping; adjust if you have a util
        minutes = {"1m":1, "5m":5, "15m":15, "1h":60, "4h":240, "1d":1440}[tf]
        cutoff = now - timedelta(minutes=minutes/2)  # conservative
        last_ts = pd.Timestamp(df.index[-1]).to_pydatetime().replace(tzinfo=timezone.utc)
        if last_ts > cutoff:
            df = df.iloc[:-1]
        return df if len(df) else None

    def _required_lookback(self, *exprs) -> int:
        """Compute minimal lookback from indicators’ params. Conservative default 300 bars."""
        def scan(e):
            if not e: return 0
            if "and" in e or "or" in e:
                return max(scan(x) for x in e.get("and", []) + e.get("or", []))
            if "not" in e:
                return scan(e["not"])
            for op in ("gt","gte","lt","lte","eq","ne","crosses_above","crosses_below","between","outside","inside_band","outside_band"):
                if op in e:
                    return max(self._operand_lookback(e[op].get("lhs")),
                               self._operand_lookback(e[op].get("rhs")),
                               self._operand_lookback(e[op].get("value")),
                               self._operand_lookback(e[op].get("lower")),
                               self._operand_lookback(e[op].get("upper")))
            return 0
        need = max(scan(x) for x in exprs)
        return max(need, 300)

    def _operand_lookback(self, operand) -> int:
        if not isinstance(operand, dict): return 0
        ind = operand.get("indicator")
        if not ind: return 0
        t = ind.get("type","").upper()
        p = ind.get("params",{})
        # rough estimates
        if t in ("SMA","EMA"): return max(200, int(p.get("period",50)) + 5)
        if t == "RSI": return max(200, int(p.get("period",14)) + 5)
        if t == "MACD": return max(200, int(p.get("slow",26)) + int(p.get("signal",9)) + 10)
        if t in ("BBANDS","ATR","ADX","STOCH","WILLR","CCI","ROC","MFI"): return 250
        return 200

    def _eval_expr(self, expr: Dict[str, Any], ohlcv: pd.DataFrame, state: Dict[str,Any]) -> tuple[bool, Dict[str,str]]:
        """Return (bool_value, sides_dict). sides_dict used for cross detection memory."""
        if expr is None:
            return False, {}

        if "and" in expr:
            sides = {}
            for sub in expr["and"]:
                val, s = self._eval_expr(sub, ohlcv, state)
                sides.update(s)
                if not val: return False, sides
            return True, sides

        if "or" in expr:
            sides = {}
            for sub in expr["or"]:
                val, s = self._eval_expr(sub, ohlcv, state)
                sides.update(s)
                if val: return True, sides
            return False, sides

        if "not" in expr:
            val, sides = self._eval_expr(expr["not"], ohlcv, state)
            return (not val), sides

        # Comparator / band / cross nodes
        for op in ("gt","gte","lt","lte","eq","ne","between","outside","inside_band","outside_band","crosses_above","crosses_below"):
            if op in expr:
                node = expr[op]
                return self._eval_node(op, node, ohlcv, state)

        raise ValueError(f"Unknown expression node: {expr}")

    def _eval_node(self, op: str, node: Dict[str,Any], ohlcv: pd.DataFrame, state: Dict[str,Any]) -> tuple[bool, Dict[str,str]]:
        def resolve(operand):
            if operand is None: return None
            if "value" in operand:
                return float(operand["value"])
            if "field" in operand:
                series = ohlcv[operand["field"]]
                v = series.iloc[-1]
                return None if (isinstance(v, float) and (math.isnan(v))) else float(v)
            if "indicator" in operand:
                out = self.ind_engine.compute(ohlcv, operand["indicator"])
                if isinstance(out, pd.DataFrame):
                    # Expect indicator.output chosen to a series; else take first column
                    series = out.iloc[:,0]
                else:
                    series = out
                if series is None or len(series) == 0:
                    return None
                v = series.iloc[-1]
                return None if (isinstance(v, float) and (math.isnan(v))) else float(v)
            return None

        sides: Dict[str,str] = {}
        # Simple comparators
        if op in ("gt","gte","lt","lte","eq","ne"):
            lhs = resolve(node.get("lhs"))
            rhs = resolve(node.get("rhs"))
            if lhs is None or rhs is None: return (False, sides)
            if op == "gt":  return (lhs >  rhs, sides)
            if op == "gte": return (lhs >= rhs, sides)
            if op == "lt":  return (lhs <  rhs, sides)
            if op == "lte": return (lhs <= rhs, sides)
            if op == "eq":  return (lhs == rhs, sides)
            if op == "ne":  return (lhs != rhs, sides)

        # Bands
        if op in ("between","outside"):
            val = resolve(node.get("value"))
            lo  = resolve(node.get("lower"))
            hi  = resolve(node.get("upper"))
            if None in (val, lo, hi): return (False, sides)
            inside = (lo <= val <= hi)
            return (inside, sides) if op == "between" else ((not inside), sides)

        if op in ("inside_band","outside_band"):
            # alias to between/outside
            return self._eval_node("between" if op=="inside_band" else "outside", node, ohlcv, state)

        # Crosses (edge-aware)
        if op in ("crosses_above","crosses_below"):
            lhs = resolve(node.get("lhs"))
            rhs = resolve(node.get("rhs"))
            if lhs is None or rhs is None: return (False, sides)

            key = self._cross_key(node)
            prev_side = (state.get("sides") or {}).get(key)

            side = "above" if lhs > rhs else ("below" if lhs < rhs else prev_side or "equal")
            sides[key] = side

            if prev_side is None:
                return (False, sides)
            if op == "crosses_above":
                return (prev_side in ("below","equal") and side == "above", sides)
            else:
                return (prev_side in ("above","equal") and side == "below", sides)

        raise ValueError(f"Unsupported op {op}")

    def _cross_key(self, node: Dict[str,Any]) -> str:
        """Stable key to remember side; we hash the node structure deterministically."""
        import json, hashlib
        payload = json.dumps(node, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def _notify(self, session, alert, cfg: Dict[str,Any]) -> None:
        user_id = alert.user_id
        text = self._format_message(cfg)
        self.notifier.send_telegram(user_id, text)
        notify = cfg.get("notify", {})
        if notify.get("email") and getattr(alert, "user_email", None):
            self.notifier.send_email(alert.user_email, f"[ALERT] {cfg.get('name','')}", text)
        self.repo.record_event(session, alert.id, "notification", {"text": text})

    def _format_message(self, cfg: Dict[str,Any]) -> str:
        import json
        name = cfg.get("name","<no-name>")
        pretty = json.dumps(cfg, ensure_ascii=False, indent=2)
        return f"ALERT TRIGGERED: {name}\n\nCONFIG:\n{pretty}\n"
