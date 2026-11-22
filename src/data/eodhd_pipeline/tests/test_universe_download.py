"""Tests for the universe, the id registry and the resumable download."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.eodhd_pipeline import config
from src.data.eodhd_pipeline.client import EodhdError
from src.data.eodhd_pipeline.download import (
    PlanLimited,
    download_histories,
    fetch_security,
    history_path,
    load_progress,
    plan_download,
    read_history,
)
from src.data.eodhd_pipeline.tests.helpers import DELISTED, LIVE, FakeEodhd, make_client
from src.data.eodhd_pipeline.universe import IdRegistry, base_symbol, build_universe, listing_frame, load_universe


def _universe(root: Path, fake: FakeEodhd | None = None) -> pd.DataFrame:
    return build_universe(make_client(fake or FakeEodhd()), root)


def test_base_symbol_strips_the_reuse_suffix():
    assert [base_symbol(c) for c in ("XYZ_old", "aac_old2", "BRK-B", "A")] == ["XYZ", "AAC", "BRK-B", "A"]


def test_listing_frame_filters_types_and_venues_and_orders_delisted_first():
    frame = listing_frame(pd.DataFrame(LIVE), pd.DataFrame(DELISTED))
    assert frame["eodhd_code"].tolist() == ["GONE", "XYZ_old", "ABC"]
    assert frame["listed"].tolist() == [False, False, True] and frame["symbol"].tolist() == ["GONE", "XYZ", "ABC"]
    assert frame["isin"].isna().tolist() == [True, True, False] and frame["isin"].iloc[2] == "US0000000101"
    wide = listing_frame(pd.DataFrame(LIVE), pd.DataFrame(DELISTED), types=("Common Stock", "Preferred Stock"), exchanges=None)
    assert set(wide["eodhd_code"]) == {"GONE", "XYZ_old", "ABC", "ABC-P", "OTCZ"}
    both = listing_frame(pd.DataFrame(LIVE[:1]), pd.DataFrame(LIVE[:1]))
    assert both["listed"].tolist() == [True]
    assert listing_frame(pd.DataFrame(), pd.DataFrame()).empty


def test_id_registry_is_stable_and_only_adds(tmp_path: Path):
    registry = IdRegistry.load(tmp_path)
    assert registry.assign(["B", "A"]) == [config.ID_OFFSET + 1, config.ID_OFFSET + 2]
    registry.save(tmp_path)
    again = IdRegistry.load(tmp_path)
    assert again.assign(["A", "C", "B"]) == [config.ID_OFFSET + 2, config.ID_OFFSET + 3, config.ID_OFFSET + 1]


def test_build_universe_keeps_the_lists_and_ids(tmp_path: Path):
    fake = FakeEodhd()
    universe = _universe(tmp_path, fake)
    assert len(fake.requests) == 2 and len(universe) == 3
    assert len(list((tmp_path / config.SYMBOLS_DIR).glob("*.json"))) == 2
    first_ids = dict(zip(universe["eodhd_code"], universe["security_id"]))
    assert dict(zip(*[_universe(tmp_path)[c] for c in ("eodhd_code", "security_id")])) == first_ids
    assert load_universe(tmp_path).equals(universe)
    with pytest.raises(FileNotFoundError):
        load_universe(tmp_path / "nothing")
    with pytest.raises(EodhdError):
        build_universe(make_client(lambda r: __import__("httpx").Response(200, json=[])), tmp_path)


def test_download_writes_one_file_per_security_and_a_progress_line(tmp_path: Path):
    universe = _universe(tmp_path)
    fake = FakeEodhd()
    client = make_client(fake)
    summary = download_histories(client, tmp_path, universe, workers=2)
    assert (summary.done, summary.missing, summary.failed, summary.stop_reason) == (2, 1, 0, None)
    assert summary.calls == 7  # GONE: one call (404); ABC and XYZ_old: three each
    ids = dict(zip(universe["eodhd_code"], universe["security_id"]))
    abc = read_history(tmp_path, ids["ABC"])
    assert abc is not None and len(abc["eod"]) == 6 and abc["splits"][0]["split"] == "2.000000/1.000000"
    assert history_path(tmp_path, ids["ABC"]).name == f"{ids['ABC']}.json.gz" and read_history(tmp_path, ids["GONE"]) is None
    progress = load_progress(tmp_path)
    assert {code: r["status"] for code, r in progress.items()} == {"ABC": "ok", "XYZ_old": "ok", "GONE": "missing"}
    again = download_histories(make_client(fake), tmp_path, universe)
    assert again.planned == 0 and again.skipped_done == 3 and again.calls == 0
    forced = download_histories(make_client(FakeEodhd()), tmp_path, universe, force=True, codes=["ABC"])
    assert forced.done == 1 and forced.calls == 3


def test_failed_codes_are_retried_on_the_next_run(tmp_path: Path):
    universe = _universe(tmp_path)
    first = download_histories(make_client(FakeEodhd(fail_codes=("XYZ_old",))), tmp_path, universe)
    assert first.failed == 1 and first.done == 1 and "XYZ_old" in first.errors[0]
    assert load_progress(tmp_path)["XYZ_old"]["status"] == "error"
    second = download_histories(make_client(FakeEodhd()), tmp_path, universe)
    assert second.planned == 1 and second.done == 1


def test_budget_and_daily_quota_stop_the_run_cleanly(tmp_path: Path):
    universe = _universe(tmp_path)
    # 4 calls: GONE costs 1 (404), which leaves room for XYZ_old (3); ABC is not reached
    capped = download_histories(make_client(FakeEodhd()), tmp_path, universe, max_calls=4, workers=1)
    assert capped.stop_reason == "budget" and (capped.done, capped.missing, capped.calls) == (1, 1, 4)
    reserve = download_histories(make_client(FakeEodhd(used=99_978)), tmp_path, universe, max_calls=2)
    assert reserve.stop_reason == "budget" and reserve.calls == 0  # 2 calls are not enough for ABC's 3
    quota = download_histories(make_client(FakeEodhd(quota_after=0)), tmp_path, universe, workers=1)
    assert quota.stop_reason == "quota" and quota.done == 0
    assert set(load_progress(tmp_path)) == {"GONE", "XYZ_old"}  # nothing recorded for the code the quota stopped
    small = download_histories(make_client(FakeEodhd(used=99_990)), tmp_path, universe, reserve=200)
    assert small.done == 1 and small.calls == 3  # 10 left: the reserve is capped at 1, so ABC still fits


def test_a_free_plan_stops_the_download_unless_allowed(tmp_path: Path):
    universe = _universe(tmp_path)
    free = FakeEodhd(free=True)
    stopped = download_histories(make_client(free), tmp_path, universe, workers=1)
    assert stopped.stop_reason == "plan_limited" and stopped.plan_limited and stopped.done == 0 and free.requests == []
    no_warning = FakeEodhd(free=True)
    no_warning.free = False  # bars and events served, but /user still says free: marked limited anyway
    record = fetch_security(make_client(no_warning), tmp_path, config.ID_OFFSET + 99, "XYZ_old", allow_limited=True, free_plan=True)
    assert record["plan_limited"] is True
    with pytest.raises(PlanLimited):
        fetch_security(make_client(no_warning), tmp_path, config.ID_OFFSET + 99, "XYZ_old", free_plan=True)
    dry = download_histories(make_client(FakeEodhd(free=True)), tmp_path, universe, allow_limited=True)
    assert dry.done == 2 and dry.plan_limited and dry.stop_reason is None
    ids = dict(zip(universe["eodhd_code"], universe["security_id"]))
    abc = read_history(tmp_path, ids["ABC"])
    assert abc is not None and abc["events_restricted"] and abc["splits"] == [] and abc["plan_limited"]


def test_restricted_events_alone_stop_a_strict_download(tmp_path: Path):
    class EventsOnlyRestricted(FakeEodhd):
        def __call__(self, request):  # bars without the free-plan warning, events still 403
            response = super().__call__(request)
            if "/eod/" in request.url.path and response.status_code == 200:
                return __import__("httpx").Response(200, json=[row for row in response.json() if "date" in row])
            return response

    with pytest.raises(PlanLimited):
        fetch_security(make_client(EventsOnlyRestricted(free=True)), tmp_path, config.ID_OFFSET + 1, "ABC")


def test_plan_download_orders_and_limits(tmp_path: Path):
    universe = _universe(tmp_path)
    rows, skipped = plan_download(universe, {"GONE": {"code": "GONE", "status": "missing"}}, limit=1)
    assert rows["eodhd_code"].tolist() == ["XYZ_old"] and skipped == 1
    rows, _ = plan_download(universe, {}, codes=["ABC", "NOPE"])
    assert rows["eodhd_code"].tolist() == ["ABC"]


def test_progress_survives_a_cut_line(tmp_path: Path):
    path = tmp_path / config.PROGRESS_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"code": "A", "status": "ok"}) + "\n\n{\"code\": \"B\", \"sta", encoding="utf-8")
    assert list(load_progress(tmp_path)) == ["A"]
