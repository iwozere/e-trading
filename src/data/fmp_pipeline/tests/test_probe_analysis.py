"""Tests for ``src.data.fmp_pipeline.probe.schema`` and ``src.data.fmp_pipeline.probe.analysis`` (synthetic data)."""

from __future__ import annotations

from src.data.fmp_pipeline.probe.analysis import date_range, dividend_basis, find_reused_tickers, split_check
from src.data.fmp_pipeline.probe.schema import infer_schema, synthesize


def test_infer_schema_list():
    schema = infer_schema([
        {"date": "2025-06-02", "close": 1.5, "volume": 10, "label": "x", "flag": True},
        {"date": "2025-06-03", "close": None, "volume": 11, "label": "2025-06-03 09:30:00"},
    ])
    assert schema.container == "list" and schema.records == 2
    assert schema.fields["date"].formats["date"] == 2
    assert schema.fields["close"].types == {"float": 1, "null": 1}
    assert schema.fields["close"].null_rate(2) == 0.5
    assert schema.fields["flag"].null_rate(2) == 0.5
    assert schema.fields["label"].formats == {"text": 1, "datetime": 1}


def test_infer_schema_wrapped_object_and_scalar():
    wrapped = infer_schema({"symbol": "X", "historical": [{"date": "2025-06-02"}]})
    assert wrapped.container == "wrapped_list" and wrapped.wrapper_key == "historical"
    assert infer_schema({"a": [1], "b": {"c": 1}}).fields["b"].types == {"object": 1}
    assert infer_schema(5).container == "scalar"


def test_synthesize_preserves_shape_but_not_values():
    original = [{"date": "2025-06-02", "close": 199.12, "volume": 123, "symbol": "AAPL", "ok": True, "x": None,
                 "when": "2025-06-02T13:30:00.000Z", "year": "2025", "num": "12.5", "l": [], "o": {}}]
    fake = synthesize(infer_schema(original), count=3)
    assert len(fake) == 3
    assert set(fake[0]) == set(original[0])
    assert fake[0]["close"] != 199.12 and fake[0]["symbol"] != "AAPL"
    assert fake[0]["date"] == "2001-02-05" and isinstance(fake[0]["volume"], int)
    assert fake[0]["x"] is None and fake[0]["l"] == [] and fake[0]["o"] == {}
    assert synthesize(infer_schema({"historical": [{"a": 1}]}))["historical"][0] == {"a": 100}
    assert synthesize(infer_schema({"a": "b"})) == {"a": "SYNTH0"}
    assert synthesize(infer_schema([])) == []


def test_dividend_basis():
    rows = [{"date": "2020-08-07", "dividend": 0.82, "adjDividend": 0.205, "yield": 0.7},
            {"date": "2020-05-08", "dividend": 0.82, "adjDividend": 0.205}]
    basis = dividend_basis(rows, "2020-08-07", 0.82)
    assert basis.found and basis.fields_matching == ("dividend",)
    assert basis.values == {"dividend": 0.82, "adjDividend": 0.205}
    assert not dividend_basis(rows, "2021-01-01", 0.82).found


def test_split_check_verdicts():
    raw = [{"date": "2020-08-28", "close": 499.0}, {"date": "2020-08-31", "close": 129.0}]
    adjusted = [{"date": "2020-08-28", "close": 124.8}, {"date": "2020-08-31", "close": 129.0}]
    assert split_check("e", raw, "2020-08-31", 4.0).verdict == "raw"
    assert split_check("e", adjusted, "2020-08-31", 4.0).verdict == "split_adjusted"
    assert split_check("e", [{"date": "2020-08-28", "close": 300.0}, {"date": "2020-08-31", "close": 129.0}],
                       "2020-08-31", 4.0).verdict == "unclear"
    assert split_check("e", raw[:1], "2020-08-31", 4.0).verdict == "no_data"
    adj_only = [{"date": r["date"], "adjClose": r["close"]} for r in raw]
    assert split_check("e", adj_only, "2020-08-31", 4.0).verdict == "raw"


def test_date_range():
    assert date_range([{"date": "2025-06-03 10:00:00"}, {"date": "2025-06-02"}, {}]) == ("2025-06-02", "2025-06-03")
    assert date_range([]) == (None, None)


def test_find_reused_tickers_ranks_evidence():
    delisted = [
        {"symbol": "OLD", "companyName": "Old A", "ipoDate": "1990-01-01", "delistedDate": "2001-01-01"},
        {"symbol": "OLD", "companyName": "Old B", "ipoDate": "2005-01-01", "delistedDate": "2010-01-01"},
        {"symbol": "TWICE", "companyName": "T A", "ipoDate": "1990-01-01", "delistedDate": "2023-01-01"},
        {"symbol": "TWICE", "companyName": "T B", "ipoDate": "2000-01-01", "delistedDate": "2024-01-01"},
        {"symbol": "BACK", "companyName": "Back Co", "ipoDate": "1999-01-01", "delistedDate": "2019-01-01"},
        {"symbol": "SAME", "companyName": "Same", "ipoDate": "1999-01-01", "delistedDate": "2019-01-01"},
        {"symbol": "SAME", "companyName": "Same", "ipoDate": "1999-01-01", "delistedDate": "2019-01-01"},
    ]
    changes = [
        {"oldSymbol": "SWAP", "newSymbol": "NEW1", "companyName": "First", "date": "2022-01-01"},
        {"oldSymbol": "OTHER", "newSymbol": "SWAP", "companyName": "Second", "date": "2023-01-01"},
        {"oldSymbol": "A1", "newSymbol": "A2", "companyName": "Same", "date": "2023-01-01"},
    ]
    found = find_reused_tickers(delisted, ["BACK", "AAPL"], changes)
    assert [(c.symbol, c.reason) for c in found] == [
        ("TWICE", "delisted more than once"),
        ("OLD", "delisted more than once"),
        ("BACK", "delisted and actively trading"),
        ("SWAP", "freed and retaken via symbol-change"),
    ]
    assert len(find_reused_tickers(delisted, ["BACK"], changes, limit=2)) == 2
