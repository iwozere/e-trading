"""
P08Pipeline — deprecated.

MTF (Multi-Timeframe) support has been merged into P07Pipeline (p07_combined) as a
configurable option.  Use P07Pipeline with enable_mtf=True instead:

    from src.ml.pipeline.p07_combined.pipeline import P07Pipeline
    p = P07Pipeline(enable_mtf=True)
    p.run_batch(ticker_files)

This shim is kept for backward-compatibility during the deprecation window.
Remove once no scheduler or caller references this entrypoint directly.
"""

import warnings
from pathlib import Path
from typing import List

from src.ml.pipeline.p07_combined.pipeline import P07Pipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_DEPRECATION_MSG = (
    "P08Pipeline is deprecated. "
    "Use P07Pipeline with enable_mtf=True instead: "
    "P07Pipeline(enable_mtf=True, result_root='results/p08_mtf')."
)


class P08Pipeline:
    """
    Deprecated shim — delegates to P07Pipeline(enable_mtf=True).
    """

    def __init__(self, db_url: str | None = None, result_root: Path = Path("results/p08_mtf")):
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        _logger.warning("P08Pipeline is deprecated. %s", _DEPRECATION_MSG)
        self._delegate = P07Pipeline(result_root=result_root, db_url=db_url, enable_mtf=True)

    @property
    def db_url(self) -> str:
        return self._delegate.db_url

    @property
    def data_loader(self):
        return self._delegate.data_loader

    def run_batch(self, ticker_files: List[Path], train_years: List[str] | None = None):
        return self._delegate.run_batch(ticker_files, train_years=train_years)

    def run_robustness(self, ticker: str, timeframe: str):
        return self._delegate.run_robustness(ticker, timeframe)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P08 MTF Pipeline (deprecated — use P07 --enable-mtf)")
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--tf", type=str)
    parser.add_argument("--years", type=str)
    args = parser.parse_args()

    import warnings as _w

    _w.filterwarnings("ignore", category=DeprecationWarning)
    p = P08Pipeline()

    data_dir = Path("data")
    pattern = "*_*_*.csv"
    if args.ticker and args.tf:
        pattern = f"{args.ticker}_{args.tf}_*.csv"
    elif args.ticker:
        pattern = f"{args.ticker}_*_*.csv"

    ticker_files = list(data_dir.glob(pattern))
    train_years = args.years.split(",") if args.years else None

    if ticker_files:
        p.run_batch(ticker_files, train_years=train_years)
    else:
        _logger.warning("No data files found matching pattern %s", pattern)
