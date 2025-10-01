# service.py - COMPLETE VERSION
# Standard library
import asyncio
from typing import Dict, Any, List

# Third party
import pandas as pd

# Local application
from src.common import get_ohlcv, determine_provider
from src.indicators.utils import coerce_ohlcv, resample_df, validate_indicator_config
from src.indicators.registry import INDICATOR_META
from src.indicators.adapters.ta_lib_adapter import TaLibAdapter
from src.indicators.adapters.pandas_ta_adapter import PandasTaAdapter
from src.indicators.adapters.fundamentals_adapter import FundamentalsAdapter
from src.indicators.models import (
    IndicatorBatchConfig, IndicatorResultSet,
    IndicatorSpec, IndicatorValue, TickerIndicatorsRequest
)


class IndicatorService:
    def __init__(self, prefer: Dict[str, int] | None = None):
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

    async def compute(
        self,
        df: pd.DataFrame,
        config: IndicatorBatchConfig,
        fund_params: Dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Async compute supporting both tech and fundamental indicators"""
        validate_indicator_config(config)

        base = coerce_ohlcv(df)
        base = resample_df(base, config.timeframe)
        out = base.copy()

        for spec in config.indicators:
            meta = INDICATOR_META[spec.name]
            adapter = self._select_provider(spec.name)

            # Per-indicator resample (tech only)
            working = (
                out if meta.kind == "fund"
                else resample_df(base, spec.timeframe) if spec.timeframe
                else base
            )

            inputs = self._build_inputs(working, spec.name, spec)
            params = dict(spec.params)

            if meta.kind == "fund":
                params.update(fund_params or {})

            # Await async compute
            res = await adapter.compute(spec.name, working, inputs, params)

            final_names = (
                spec.output if isinstance(spec.output, dict)
                else {"value": spec.output}
            )

            for k, s in res.items():
                cname = final_names.get(k)
                if not cname:
                    continue

                # Align to out index
                if len(s.index) == 1:
                    s = pd.Series(s.iloc[0], index=out.index)
                else:
                    s = s.reindex(out.index).ffill()

                out[cname] = s

        if config.dropna_after:
            out = out.dropna()

        return out

    async def compute_for_ticker(
        self,
        req: TickerIndicatorsRequest
    ) -> IndicatorResultSet:
        """Compute indicators for a ticker (both technical and fundamental)"""
        provider = req.provider or determine_provider(req.ticker)

        # Fetch OHLCV in thread pool (it's sync)
        df = await asyncio.to_thread(
            get_ohlcv, req.ticker, req.timeframe, req.period, provider
        )

        # Build specs
        specs: List[IndicatorSpec] = []
        for name in req.indicators:
            meta = INDICATOR_META[name]
            if meta.kind == "tech":
                outmap = (
                    {"value": name} if meta.outputs == ["value"]
                    else {o: f"{name}_{o}" for o in meta.outputs}
                )
                specs.append(IndicatorSpec(
                    name=name,
                    output=outmap if len(outmap) > 1 else outmap["value"]
                ))
            else:
                specs.append(IndicatorSpec(name=name, output=name))

        cfg = IndicatorBatchConfig(timeframe=req.timeframe, indicators=specs)

        # Compute is async and handles both tech + fund
        df_all = await self.compute(
            df, cfg,
            fund_params={"ticker": req.ticker, "provider": provider}
        )

        # Build results
        tech: Dict[str, IndicatorValue] = {}
        fund: Dict[str, IndicatorValue] = {}

        for name in req.indicators:
            cols = [c for c in df_all.columns
                   if c == name or c.startswith(f"{name}_")]
            for c in cols:
                v = df_all[c].iloc[-1] if len(df_all) else None
                target = tech if INDICATOR_META[name].kind == "tech" else fund
                target[c] = IndicatorValue(
                    name=c,
                    value=None if pd.isna(v) else v
                )

        return IndicatorResultSet(
            ticker=req.ticker,
            technical=tech,
            fundamental=fund
        )