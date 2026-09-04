"""Tests for `src.data.pipeline.runner`."""

from __future__ import annotations

import textwrap

import pytest

from src.data.pipeline.base_plugin import PluginSpec
from src.data.pipeline.runner import _parse_script_output, _resolve_scope, run_one


def test_resolve_scope_all_returns_full_registry():
    from src.data.pipeline.registry import PLUGIN_REGISTRY

    assert _resolve_scope("all") == list(PLUGIN_REGISTRY)


def test_resolve_scope_by_category():
    specs = _resolve_scope("p22")
    assert len(specs) > 0
    assert all(spec.category == "p22" for spec in specs)


def test_resolve_scope_by_exact_name():
    specs = _resolve_scope("P22 Daily Price Ingest")
    assert len(specs) == 1
    assert specs[0].name == "P22 Daily Price Ingest"


def test_resolve_scope_unknown_raises():
    with pytest.raises(ValueError):
        _resolve_scope("no-such-scope-or-plugin")


def test_parse_script_output_extracts_json():
    stdout = "some log line\n__SCHEDULER_RESULT__:{\"ok\": true, \"count\": 3}\ntrailing\n"
    assert _parse_script_output(stdout) == {"ok": True, "count": 3}


def test_parse_script_output_missing_marker_returns_empty():
    assert _parse_script_output("just some logs\nno marker here\n") == {}


def test_parse_script_output_malformed_json_reports_parse_error():
    result = _parse_script_output("__SCHEDULER_RESULT__:{not valid json\n")
    assert "parse_error" in result


def test_run_one_success():
    # script_path must resolve under an allowed dir relative to PROJECT_ROOT, so
    # this smoke test writes into (and cleans up) a real allowed location rather
    # than using tmp_path directly.
    from src.data.pipeline.base_plugin import PROJECT_ROOT

    real_script = PROJECT_ROOT / "src" / "data" / "pipeline" / "tests" / "_dummy_ok_script.py"
    real_script.write_text(
        textwrap.dedent(
            """
            print("__SCHEDULER_RESULT__:" + '{"rows": 5}')
            """
        ).strip(),
        encoding="utf-8",
    )
    try:
        spec = PluginSpec(
            name="test-dummy-ok",
            category="test",
            cron="0 0 * * *",
            script_path="src/data/pipeline/tests/_dummy_ok_script.py",
            timeout_seconds=30,
        )
        result = run_one(spec)
        assert result.success is True
        assert result.exit_code == 0
        assert result.script_result == {"rows": 5}
    finally:
        real_script.unlink(missing_ok=True)


def test_run_one_nonzero_exit_is_failure():
    from src.data.pipeline.base_plugin import PROJECT_ROOT

    real_script = PROJECT_ROOT / "src" / "data" / "pipeline" / "tests" / "_dummy_fail_script.py"
    real_script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    try:
        spec = PluginSpec(
            name="test-dummy-fail",
            category="test",
            cron="0 0 * * *",
            script_path="src/data/pipeline/tests/_dummy_fail_script.py",
            timeout_seconds=30,
        )
        result = run_one(spec)
        assert result.success is False
        assert result.exit_code == 1
    finally:
        real_script.unlink(missing_ok=True)
