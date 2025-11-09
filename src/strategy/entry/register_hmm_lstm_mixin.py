"""
Registration script for HMM-LSTM Entry Mixin

This script registers the HMMLSTMEntryMixin with the entry mixin factory
so it can be used with the CustomStrategy framework.

Usage:
    Import this module to automatically register the mixin:

    from src.strategy.entry.register_hmm_lstm_mixin import *

    Or call the registration function explicitly:

    from src.strategy.entry.register_hmm_lstm_mixin import register_hmm_lstm_mixin
    register_hmm_lstm_mixin()
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.strategy.entry.entry_mixin_factory import ENTRY_MIXIN_REGISTRY
from src.strategy.entry.hmm_lstm_entry_mixin import HMMLSTMEntryMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def register_hmm_lstm_mixin():
    """Register the HMM-LSTM entry mixin."""
    try:
        ENTRY_MIXIN_REGISTRY["HMMLSTMEntryMixin"] = HMMLSTMEntryMixin
        _logger.info("Successfully registered HMMLSTMEntryMixin")
        return True
    except Exception:
        _logger.exception("Failed to register HMMLSTMEntryMixin")
        return False

def check_registration():
    """Check if the mixin is properly registered."""
    return "HMMLSTMEntryMixin" in ENTRY_MIXIN_REGISTRY

def list_registered_mixins():
    """List all registered entry mixins."""
    return list(ENTRY_MIXIN_REGISTRY.keys())

# Auto-register when module is imported
if __name__ == "__main__":
    # Manual registration for testing
    success = register_hmm_lstm_mixin()
    if success:
        print("✓ HMMLSTMEntryMixin registered successfully")
        print(f"Available entry mixins: {list_registered_mixins()}")
    else:
        print("✗ Failed to register HMMLSTMEntryMixin")
else:
    # Auto-register when imported
    register_hmm_lstm_mixin()
