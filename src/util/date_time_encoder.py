"""
DateTime Encoder Module

This module provides a custom JSON encoder for handling datetime objects and other special types
that are not directly JSON serializable.
"""

import datetime
import json

import numpy as np
import pandas as pd


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects and other special types."""

    def default(self, obj):
        """Convert special types to JSON serializable format."""
        if isinstance(obj, (datetime.datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
