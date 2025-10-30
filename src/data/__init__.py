# src/data/__init__.py  — minimal & safe
"""
Public namespace for the data package.

Deliberately side-effect free: do not import heavy submodules here.
Import from subpackages instead, e.g.:
  from src.data.downloader import BinanceDataDownloader
  from src.data.db.models.model_users import User
"""

__all__: list[str] = []  # keep empty on purpose