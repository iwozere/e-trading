from typing import Any, Dict, List
from abc import ABC, abstractmethod
import pandas as pd

class SignalPlugin(ABC):
    """
    Abstract base class for custom indicator signal plugins in the alert evaluation system.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The unique identifier for this plugin (e.g., 'bbands_signal').
        Used in the AST rule node: {"plugin": "<name>", "params": {...}}
        """
        pass

    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """
        Returns a JSON schema dictionary for validating the 'params' 
        this plugin expects from the frontend ConfigBuilder.
        """
        pass

    @abstractmethod
    def get_required_indicators(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns a list of indicator specifications needed to evaluate this signal.
        The AlertEvaluator uses this to calculate required indicators before calling evaluate().
        Example config:
        [
            {
               "type": "BOLLINGER",
               "output": "bbands_calc",
               "params": {"period": params["period"], "std_dev": params["dev_up"]}
            }
        ]
        """
        pass

    @abstractmethod
    def evaluate(self, 
                 params: Dict[str, Any], 
                 market_data: pd.DataFrame, 
                 indicators: Dict[str, pd.Series]) -> bool:
        """
        Evaluate the logic condition based on the provided data.
        
        Args:
            params: The parameters provided by the user in the rule configuration.
            market_data: OHLCV DataFrame
            indicators: Dictionary of pre-calculated indicator series (keyed by "output" from get_required_indicators)
            
        Returns:
            True if the condition is met (alert triggers), False otherwise.
        """
        pass
