"""Intentionally empty.

The strategy-pack CLI entrypoint lives in ``src/strategy_pack/run.py`` and is
invoked directly by the scheduler (``python src/strategy_pack/run.py ...``)
or from the shell. This module is kept empty so there is no duplicated entry
logic between ``__main__.py`` and ``run.py``.
"""
