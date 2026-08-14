"""Unit tests for `coverage` classification."""

from src.portfolio.management.coverage import CoverageStatus, classify, evaluate


def test_classify_uncovered_when_no_protective_qty():
    assert classify(position_qty=100, protective_qty=0) == CoverageStatus.UNCOVERED


def test_classify_covered_when_fully_protected():
    assert classify(position_qty=100, protective_qty=100) == CoverageStatus.COVERED


def test_classify_covered_when_over_protected():
    """More protective quantity than held is still just 'covered'."""
    assert classify(position_qty=100, protective_qty=150) == CoverageStatus.COVERED


def test_classify_partially_covered():
    assert classify(position_qty=100, protective_qty=40) == CoverageStatus.PARTIALLY_COVERED


def test_evaluate_missing_symbol_treated_as_uncovered():
    rows = evaluate(positions={"AAA": 100}, protective_qty_by_symbol={})
    assert len(rows) == 1
    assert rows[0].status == CoverageStatus.UNCOVERED
    assert rows[0].protected_qty == 0


def test_evaluate_caps_protected_qty_at_position_qty():
    rows = evaluate(positions={"AAA": 100}, protective_qty_by_symbol={"AAA": 250})
    assert rows[0].protected_qty == 100
    assert rows[0].status == CoverageStatus.COVERED


def test_evaluate_sorted_by_ticker():
    rows = evaluate(positions={"BBB": 10, "AAA": 20}, protective_qty_by_symbol={})
    assert [r.ticker for r in rows] == ["AAA", "BBB"]
