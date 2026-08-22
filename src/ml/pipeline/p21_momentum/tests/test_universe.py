"""Unit tests for src.ml.pipeline.p21_momentum.data.universe."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.ml.pipeline.p21_momentum.data.universe import (
    UniverseConstituent,
    fetch_universe,
    universe_to_json,
)


class TestFetchUniverse(unittest.TestCase):
    @patch("src.ml.pipeline.p21_momentum.data.universe.get_sp500_constituents_with_sector")
    def test_fetch_universe_maps_dataframe_to_constituents(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame(
            {"ticker": ["AAPL", "BRK-B"], "sector": ["Information Technology", "Financials"]}
        )
        result = fetch_universe()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], UniverseConstituent(ticker="AAPL", sector="Information Technology"))
        self.assertEqual(result[1], UniverseConstituent(ticker="BRK-B", sector="Financials"))


class TestUniverseToJson(unittest.TestCase):
    def test_universe_to_json_shape(self):
        constituents = [
            UniverseConstituent(ticker="AAPL", sector="Information Technology"),
            UniverseConstituent(ticker="MSFT", sector="Information Technology"),
        ]
        payload = universe_to_json(constituents, as_of="2026-08-31")
        self.assertEqual(payload["as_of"], "2026-08-31")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(
            payload["constituents"],
            [
                {"ticker": "AAPL", "sector": "Information Technology"},
                {"ticker": "MSFT", "sector": "Information Technology"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
