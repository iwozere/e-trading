"""
Trading Strategy Pack (SP-1..SP-6) — plugin specs.

Ported verbatim from ``bin/scheduler/insert_strategy_pack_schedules.sql``
(SP-1..SP-4: monthly/daily/weekly/quarterly) and
``insert_strategy_pack_intraday_schedules.sql`` (SP-5, SP-6: intraday/
bar-close), which this module supersedes. No `notification_rules`,
deliberately: the pack sends its own notifications via
`NotificationServiceClient` (dedup via `DedupStore`) — adding scheduler-level
rules would double-notify.

Note: the original SQL rows used `target='src.strategy_pack'` (the package,
not the actual `run.py` module) for every row. `PluginSpec.module_target`
derives `'src.strategy_pack.run'` from `script_path` instead — `target` is
metadata only (execution uses `task_params.script_path`, per
`SchedulerService._execute_data_processing_job`), so `register_jobs.py`'s
dry-run will show a one-time, harmless `target` normalization for these 9
rows, the same class of change already made for P21's rows.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

_SCRIPT = "src/strategy_pack/run.py"


def _args(strategy_num: str, variant: str, config_name: str) -> list[str]:
    return ["run", "-s", strategy_num, "-v", variant, "-c", f"config/strategy_pack/schedules/{config_name}"]


SPECS: List[PluginSpec] = [
    PluginSpec(
        name="Strategy Pack SP-1 Monthly Momentum",
        category="strategy_pack",
        cron="0 1 1 * *",
        script_path=_SCRIPT,
        script_args=_args("1", "A", "sp1_monthly.json"),
        timeout_seconds=1800,
    ),
    PluginSpec(
        name="Strategy Pack SP-2 Daily Trend",
        category="strategy_pack",
        cron="30 22 * * *",
        script_path=_SCRIPT,
        script_args=_args("2", "A", "sp2_daily.json"),
        timeout_seconds=900,
    ),
    PluginSpec(
        name="Strategy Pack SP-3 Weekly Lazy",
        category="strategy_pack",
        cron="0 22 * * 0",
        script_path=_SCRIPT,
        script_args=_args("3", "A", "sp3_weekly.json"),
        timeout_seconds=900,
    ),
    PluginSpec(
        name="Strategy Pack SP-4 Quarterly Rebalance",
        category="strategy_pack",
        cron="0 2 1 1,4,7,10 *",
        script_path=_SCRIPT,
        script_args=_args("4", "A", "sp4_quarterly.json"),
        timeout_seconds=1800,
    ),
    PluginSpec(
        name="Strategy Pack SP-5 Swing BTC 4h",
        category="strategy_pack",
        cron="2 0,4,8,12,16,20 * * *",
        script_path=_SCRIPT,
        script_args=_args("5", "A", "sp5_btc_4h.json"),
        timeout_seconds=600,
    ),
    PluginSpec(
        name="Strategy Pack SP-5 Swing BTC 1h",
        category="strategy_pack",
        cron="2 * * * *",
        script_path=_SCRIPT,
        script_args=_args("5", "A", "sp5_btc_1h.json"),
        timeout_seconds=600,
    ),
    PluginSpec(
        name="Strategy Pack SP-5 Swing SPY 1h",
        category="strategy_pack",
        cron="2 14-22 * * 1-5",
        script_path=_SCRIPT,
        script_args=_args("5", "A", "sp5_spy_1h.json"),
        timeout_seconds=600,
        description="14-22 UTC covers US RTH across both DST regimes.",
    ),
    PluginSpec(
        name="Strategy Pack SP-6 EMA+SuperTrend BTC 4h",
        category="strategy_pack",
        cron="2 0,4,8,12,16,20 * * *",
        script_path=_SCRIPT,
        script_args=_args("6", "A", "sp6_btc_4h.json"),
        timeout_seconds=600,
    ),
    PluginSpec(
        name="Strategy Pack SP-6 EMA+SuperTrend BTC 1d",
        category="strategy_pack",
        cron="5 0 * * *",
        script_path=_SCRIPT,
        script_args=_args("6", "A", "sp6_btc_1d.json"),
        timeout_seconds=600,
        description="Crypto 1d bar closes at 00:00 UTC; fires at 00:05 UTC for a clean right-after-close run.",
    ),
]
