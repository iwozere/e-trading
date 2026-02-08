import pytest
import pandas as pd
import os
from src.vectorbt.data.loader import DataLoader

def test_data_loader_integrity():
    # Note: Requires test data in data/ folder or mocks
    # For now we check if the class can be initialized and handles missing data gracefully
    loader = DataLoader(data_dir="data")

    # We check if data directory exists, otherwise we skip or mock
    if not os.path.exists("data"):
        pytest.skip("Test data directory 'data' not found")

    try:
        data = loader.load_all_symbols("1h", start_date="2024-01-01", end_date="2024-01-10")

        if data is not None:
            # Check MultiIndex structure
            assert data.columns.nlevels == 2
            assert 'Close' in data.columns.get_level_values('column')

            # Check no missing data after alignment
            assert data.isna().sum().sum() == 0

            print("✅ Data loader integrity test passed")
    except Exception as e:
        if "No files found" in str(e):
            pytest.skip(f"No CSV files found in data/ for pattern: {e}")
        else:
            raise e
