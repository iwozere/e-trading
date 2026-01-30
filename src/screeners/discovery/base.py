from abc import ABC, abstractmethod
from typing import List

class IDiscoveryProvider(ABC):
    """Interface for symbol discovery providers."""

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Returns a list of symbols to be screened."""
        pass
