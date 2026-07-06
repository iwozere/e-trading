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

    def default(self, o):
        """Convert special types to JSON serializable format."""
        if isinstance(o, (datetime.datetime, pd.Timestamp)):
            return o.isoformat()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
