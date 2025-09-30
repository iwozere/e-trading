# ---------------------------------------------------------------------------
# service.py — orchestrator
# ---------------------------------------------------------------------------
import asyncio
import pandas as pd
from typing import Tuple, Dict, Any
from src.common import get_ohlcv, determine_provider
from src.indicators.utils import coerce_ohlcv, resample_df
from src.indicators.registry import INDICATOR_META
from src.indicators.adapters.ta_lib_adapter import TaLibAdapter
from src.indicators.adapters.pandas_ta_adapter import PandasTaAdapter
from src.indicators.adapters.fundamentals_adapter import FundamentalsAdapter

from src.indicators.models import IndicatorBatchConfig, IndicatorResultSet, IndicatorSpec, IndicatorValue, TickerIndicatorsRequest

class IndicatorService:
    def __init__(self, prefer: Dict[str,int] | None = None):
        self.adapters = {
            "ta-lib": TaLibAdapter(),
            "pandas-ta": PandasTaAdapter(),
            "fundamentals": FundamentalsAdapter(),
        }
        self.prefer = prefer or {}

    def _select_provider(self, name: str):
        meta = INDICATOR_META.get(name)
        if not meta:
            raise ValueError(f"Unknown indicator: {name}")
        # stable order but allow overrides via prefer (lower value = higher priority)
        candidates = sorted(meta.providers, key=lambda p: self.prefer.get(p, 0))
        for prov in candidates:
            if self.adapters[prov].supports(name):
                return self.adapters[prov]
        raise RuntimeError(f"No adapter supports {name}")

    def _build_inputs(self, df: pd.DataFrame, name: str, spec: IndicatorSpec | None = None):
        meta = INDICATOR_META[name]
        inputs = {}
        if meta.kind == "tech":
            for key in meta.inputs:
                col = (spec.input_map.get(key) if spec else None) or key
                if col not in df.columns:
                    raise ValueError(f"Missing input column '{col}' for {name}")
                inputs[key] = df[col]
        return inputs

    # --- API 1: compute from OHLCV DataFrame with batch config (tech + fund supported) ---
    def compute(self, df: pd.DataFrame, config: IndicatorBatchConfig, fund_params: Dict[str,Any] | None = None) -> pd.DataFrame:
        base = coerce_ohlcv(df)
        base = resample_df(base, config.timeframe)
        out = base.copy()
        for spec in config.indicators:
            meta = INDICATOR_META[spec.name]
            adapter = self._select_provider(spec.name)
            # Per-indicator resample (tech only)
            working = out if meta.kind == "fund" else resample_df(base, spec.timeframe) if spec.timeframe else base
            inputs = self._build_inputs(working, spec.name, spec)
            params = dict(spec.params)
            if meta.kind == "fund":
                params.update((fund_params or {}))
            res = adapter.compute(spec.name, working, inputs, params)
            final_names = spec.output if isinstance(spec.output, dict) else {"value": spec.output}
            for k, s in res.items():
                cname = final_names.get(k)
                if not cname: continue
                # align to out index; for fundamentals (len==1) ffill across index
                if len(s.index) == 1:
                    s = pd.Series(s.iloc[0], index=out.index)
                else:
                    s = s.reindex(out.index).ffill()
                out[cname] = s
        if config.dropna_after:
            out = out.dropna()
        return out

    # --- API 2: compute by ticker request (fetch OHLCV + fundamentals internally) ---
    async def compute_for_ticker(self, req: TickerIndicatorsRequest) -> IndicatorResultSet:
        provider = req.provider or determine_provider(req.ticker)
        df = get_ohlcv(req.ticker, req.timeframe, req.period, provider)
        # Build a batch config from simple names with sane outputs
        specs: List[IndicatorSpec] = []
        for name in req.indicators:
            meta = INDICATOR_META[name]
            if meta.kind == "tech":
                outmap = {"value": name} if meta.outputs==["value"] else {o: f"{name}_{o}" for o in meta.outputs}
                specs.append(IndicatorSpec(name=name, output=outmap if len(outmap)>1 else outmap["value"]))
            else:
                specs.append(IndicatorSpec(name=name, output=name))
        cfg = IndicatorBatchConfig(timeframe=req.timeframe, indicators=specs)
        df_all = self.compute(df, cfg, fund_params={"ticker": req.ticker, "provider": provider})
        # Build compact result set (latest values)
        tech: Dict[str, IndicatorValue] = {}
        fund: Dict[str, IndicatorValue] = {}
        for name in req.indicators:
            cols = [c for c in df_all.columns if c==name or c.startswith(f"{name}_")]
            for c in cols:
                v = df_all[c].iloc[-1] if len(df_all) else None
                target = tech if INDICATOR_META[name].kind=="tech" else fund
                target[c] = IndicatorValue(name=c, value=None if pd.isna(v) else v)
        return IndicatorResultSet(ticker=req.ticker, technical=tech, fundamental=fund)
