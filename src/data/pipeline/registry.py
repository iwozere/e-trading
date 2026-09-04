"""
The plugin registry — one list of every data-download/ingest job in scope.

To add a new data source: write its `run_*.py` script (or reuse an existing
one), add a `PluginSpec` for it to the relevant `specs/*_specs.py` group (or a
new group file), and append that group's `SPECS` list to `_GROUPS` below.
Nothing else needs to change — `register_jobs.py` and `runner.py` both derive
everything from `PLUGIN_REGISTRY`.

**Migration status** (see docs/Tasks.md for the up-to-date checklist): every
pipeline previously registered via `bin/scheduler/insert_*.sql`, a
per-pipeline `jobs/register_jobs.py`, or found only as a live DB row with no
file source at all (`specs/p15_specs.py`'s two bundle jobs) is now ported
here — this registry supersedes all 11 SQL seed files and the 3 Python
`register_jobs.py` modules (P20/P21/P22). Those files are kept as an archived
reference until a `register_jobs.py --dry-run` against production confirms
an empty diff (see docs/Tasks.md); they are not deleted yet.

`specs/providers.py` adds the only 3 provider downloaders confirmed to have
no existing caller anywhere (FRED, AAII, Fear & Greed's missing weekly
rebuild) — genuinely new schedule rows. `cboe`, `wikipedia`/index_changes,
and `russell3000` are deliberately NOT separately scheduled: P15's daily/
weekly bundle jobs already refresh those caches, and `openfigi` has no
"download everything" concept (on-demand CUSIP resolution only) — see
`specs/providers.py`'s docstring.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from src.data.pipeline.base_plugin import PluginSpec
from src.data.pipeline.specs.core_specs import SPECS as _CORE_SPECS
from src.data.pipeline.specs.p05_specs import SPECS as _P05_SPECS
from src.data.pipeline.specs.p10_specs import SPECS as _P10_SPECS
from src.data.pipeline.specs.p15_specs import SPECS as _P15_SPECS
from src.data.pipeline.specs.p17_specs import SPECS as _P17_SPECS
from src.data.pipeline.specs.p18_specs import SPECS as _P18_SPECS
from src.data.pipeline.specs.p19_specs import SPECS as _P19_SPECS
from src.data.pipeline.specs.p20_specs import SPECS as _P20_SPECS
from src.data.pipeline.specs.p21_specs import SPECS as _P21_SPECS
from src.data.pipeline.specs.p22_specs import SPECS as _P22_SPECS
from src.data.pipeline.specs.providers import SPECS as _PROVIDER_SPECS
from src.data.pipeline.specs.screener_specs import SPECS as _SCREENER_SPECS
from src.data.pipeline.specs.strategy_pack_specs import SPECS as _STRATEGY_PACK_SPECS

_GROUPS: List[List[PluginSpec]] = [
    _CORE_SPECS,
    _P05_SPECS,
    _P10_SPECS,
    _P15_SPECS,
    _P17_SPECS,
    _P18_SPECS,
    _P19_SPECS,
    _P20_SPECS,
    _P21_SPECS,
    _P22_SPECS,
    _PROVIDER_SPECS,
    _SCREENER_SPECS,
    _STRATEGY_PACK_SPECS,
]

PLUGIN_REGISTRY: List[PluginSpec] = [spec for group in _GROUPS for spec in group]


def _check_unique_names(specs: List[PluginSpec]) -> None:
    counts = Counter(spec.name for spec in specs)
    dupes = [name for name, n in counts.items() if n > 1]
    if dupes:
        raise ValueError(f"Duplicate PluginSpec name(s) in registry: {dupes}")


def _check_dependencies_exist(specs: List[PluginSpec]) -> None:
    """Every `depends_on` entry must name another plugin actually in the registry —
    catches a typo'd or renamed-but-not-updated dependency at import time rather
    than silently no-op-ing the completion gate at runtime (see `dependency_status.py`)."""
    names = {spec.name for spec in specs}
    bad: List[str] = []
    for spec in specs:
        for dep in spec.depends_on:
            if dep not in names:
                bad.append(f"{spec.name!r} depends_on unknown plugin {dep!r}")
    if bad:
        raise ValueError("Unresolvable depends_on reference(s):\n" + "\n".join(bad))


_check_unique_names(PLUGIN_REGISTRY)
_check_dependencies_exist(PLUGIN_REGISTRY)

_BY_NAME: Dict[str, PluginSpec] = {spec.name: spec for spec in PLUGIN_REGISTRY}


def get_by_name(name: str) -> PluginSpec | None:
    """Return the `PluginSpec` with this exact `name`, or None."""
    return _BY_NAME.get(name)


def get_by_category(category: str) -> List[PluginSpec]:
    """Return all specs tagged with this `category` (e.g. "p20", "p22")."""
    return [spec for spec in PLUGIN_REGISTRY if spec.category == category]


def list_categories() -> List[str]:
    """Return all distinct categories currently registered, sorted."""
    return sorted({spec.category for spec in PLUGIN_REGISTRY})
