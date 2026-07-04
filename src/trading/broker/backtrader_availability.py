"""
Single place to probe for optional ``backtrader``.

Core broker code should not ``import backtrader`` directly; import
``BACKTRADER_AVAILABLE`` from here instead. The bridge module loads
``backtrader`` itself when building ``BacktraderBrokerBridge``.
"""

try:
    import backtrader  # noqa: F401

    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
