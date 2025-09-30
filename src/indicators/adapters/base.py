# ---------------------------------------------------------------------------
# adapters/base.py
# ---------------------------------------------------------------------------
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseAdapter(ABC):
    @abstractmethod
    def supports(self, name: str) -> bool: ...

    @abstractmethod
    def compute(self, name: str, df: pd.DataFrame | None, inputs: Dict[str, pd.Series], params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Return dict of canonical output→Series aligned to df.index (or scalar for fundamentals)."""
        ...
