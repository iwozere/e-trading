"""
GDELT Data Downloaders (v1.0 and v2.0)

Two downloader classes cover the full GDELT history:

``Gdelt1Downloader`` — GDELT 1.0  (2013-04-01 to 2015-02-17)
    Downloads daily GKG 1.0 zip files and caches them as-is.
    One file per day, no aggregation.

``GdeltDownloader`` — GDELT 2.0  (2015-02-18 onward)
    Downloads 15-minute GKG and Events zip files, aggregates them to daily
    Parquet files (theme-level for GKG, EventCode-level for Events).

Cache layout:
    DATA_CACHE_DIR/gdelt/
        gkg/
            YYYYMMDD.gkg.csv.zip   ← GDELT 1.0 raw daily GKG zip
            YYYYMMDD.gkg.csv.gz    ← GDELT 2.0 aggregated GKG by theme (one file per day)
        events/
            YYYYMMDD.events.csv.gz ← GDELT 2.0 aggregated Events by EventCode (one file per day)

Classes:
- Gdelt1Downloader: GDELT 1.0 daily GKG downloader (raw zip caching)
- GdeltDownloader: GDELT 2.0 15-minute GKG + Events downloader (Parquet)


  To get 2013-04-01 → 2015-02-17 you need the v1 downloader:
  python src/data/downloader/gdelt_downloader.py v1-gkg-range --start 2013-04-01 --end 2015-02-17

  And for 2015-02-18 specifically, force full coverage:
  python src/data/downloader/gdelt_downloader.py gkg-day --date 2015-02-18 --files-per-day 96

"""

import io
import json
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import requests

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

# GDELT 1.0 — daily files
_GDELT_1_GKG_BASE = "http://data.gdeltproject.org/gkg/"
_GDELT_1_GKG_START = datetime(2013, 4, 1)
# GDELT 1.0 GKG has no end date — it continues to be published daily alongside v2.

# GKG 1.0 tab-separated columns (no header in the file)
_GKG1_COLS = [
    "DATE", "NUMARTS", "COUNTS", "THEMES", "LOCATIONS",
    "PERSONS", "ORGANIZATIONS", "TONE", "CAMEOCOUNTRIES", "GCAM",
]

# GDELT 2.0 — 15-minute files
_GDELT_BASE = "http://data.gdeltproject.org/gdeltv2/"
_GDELT_2_START = datetime(2015, 2, 18)

# Polite rate limiting — no official limit documented for GDELT HTTP
_DEFAULT_REQUEST_DELAY = 0.5  # seconds between requests
_DEFAULT_FILES_PER_DAY = 4    # one per ~6 hours; use 96 for full coverage

# All 96 HHMMSS timeslots in a day (every 15 minutes)
_ALL_TIMESLOTS: List[str] = [
    f"{h:02d}{m:02d}00"
    for h in range(24)
    for m in range(0, 60, 15)
]

_GKG_COLS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier",
    "SourceCommonName", "DocumentIdentifier", "Counts",
    "V2Counts", "Themes", "V2Themes", "Locations",
    "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "V2Tone", "Dates", "GCAM",
    "SharingImage", "RelatedImages", "SocialImageEmbeds",
    "SocialVideoEmbeds", "Quotations", "AllNames",
    "Amounts", "TranslationInfo", "Extras",
]

_EVENTS_COLS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]


class GdeltDownloader(BaseDataDownloader):
    """
    GDELT 2.0 Events and GKG Downloader.

    Downloads 15-minute GDELT CSV files from the GDELT project server,
    aggregates them to daily Parquet files, and caches them locally under
    DATA_CACHE_DIR/gdelt/.

    GKG aggregation: one row per (date, theme) matching the ``gdelt_daily`` schema.
    Events aggregation: one row per (date, EventCode).

    GDELT 2.0 is available from 2015-02-18 onwards.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        request_delay: float = _DEFAULT_REQUEST_DELAY,
        files_per_day: int = _DEFAULT_FILES_PER_DAY,
    ):
        """
        Initialize the GDELT downloader.

        Args:
            cache_dir: Root cache directory. Defaults to DATA_CACHE_DIR.
                       GDELT files are stored under <cache_dir>/gdelt/.
            request_delay: Minimum seconds between HTTP requests. Default: 0.5.
            files_per_day: 15-minute files to fetch per day.
                           96 = full coverage, 4 = one per 6 h (default), 1 = midnight only.
        """
        super().__init__()
        root = Path(cache_dir) if cache_dir else Path(DATA_CACHE_DIR)
        self._gdelt_dir = root / "gdelt"
        self._events_dir = self._gdelt_dir / "events"
        self._gkg_dir = self._gdelt_dir / "gkg"
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "e-trading-research gdelt-downloader akossyrev@gmail.com",
        })
        self._request_delay = request_delay
        self._last_request_time: float = 0.0
        self._timeslots = _select_timeslots(files_per_day)
        _logger.info(
            "GdeltDownloader initialised: files_per_day=%d timeslots=%s",
            files_per_day, len(self._timeslots),
        )

    # ------------------------------------------------------------------
    # BaseDataDownloader interface
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """Return the canonical provider name."""
        return "gdelt"

    def get_supported_intervals(self) -> List[str]:
        """GDELT provides news/event data — no OHLCV intervals."""
        return []

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Not supported by GDELT — returns an empty DataFrame.

        Args:
            symbol: Unused.
            interval: Unused.
            start_date: Unused.
            end_date: Unused.
            **kwargs: Unused.

        Returns:
            Empty DataFrame.
        """
        del symbol, interval, start_date, end_date, kwargs
        _logger.warning("GDELT does not provide OHLCV data. Use download_gkg_day() or download_events_day().")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # GKG downloads
    # ------------------------------------------------------------------

    def download_gkg_day(self, date: datetime, force: bool = False) -> Optional[Path]:
        """
        Download, aggregate, and cache GKG data for a single calendar day.

        Fetches up to ``files_per_day`` 15-minute GKG CSV zip files, parses
        V2Themes and V2Tone, and produces a per-theme daily summary saved to
        DATA_CACHE_DIR/gdelt/gkg/YYYYMMDD.gkg.csv.gz.

        Schema (DatetimeIndex = date):
            theme, article_count, avg_tone, positive_avg, negative_avg, polarity_avg

        Args:
            date: Calendar day to download. Time component is ignored.
            force: If True, re-download even when the cache already contains this date.

        Returns:
            Path to the day's ``YYYYMMDD.gkg.csv.gz`` file, or None if no data
            could be retrieved.
        """
        date_str = date.strftime("%Y%m%d")
        day_file = self._gkg_dir / f"{date_str}.gkg.csv.gz"

        if day_file.exists() and not force:
            _logger.debug("GKG %s already cached at %s", date.date(), day_file)
            return day_file

        if date < _GDELT_2_START:
            _logger.warning("GDELT 2.0 data starts 2015-02-18; skipping %s", date.date())
            return None

        frames: List[pd.DataFrame] = []

        for slot in self._timeslots:
            url = f"{_GDELT_BASE}{date_str}{slot}.gkg.csv.zip"
            df = self._download_gkg_file(url)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            _logger.warning("No GKG data retrieved for %s", date.date())
            return None

        raw = pd.concat(frames, ignore_index=True)
        aggregated = _aggregate_gkg(raw, date)

        if aggregated.empty:
            _logger.warning("GKG aggregation produced no rows for %s", date.date())
            return None

        self._gkg_dir.mkdir(parents=True, exist_ok=True)
        aggregated.set_index("date").to_csv(day_file, compression="gzip")
        _logger.info("Saved GKG %s: %d theme-rows → %s", date.date(), len(aggregated), day_file)
        return day_file

    def download_gkg_range(
        self,
        start_date: datetime,
        end_date: datetime,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Download GKG data for a date range (inclusive on both ends).

        Args:
            start_date: First day to download.
            end_date: Last day to download.
            force: If True, re-download already-cached days.

        Returns:
            Summary dict: ``total``, ``downloaded``, ``skipped``, ``errors``.
        """
        return self._range_download(
            start_date, end_date,
            self.download_gkg_day, self._gkg_dir,
            force, "GKG", ".gkg.csv.gz",
        )

    def load_gkg_day(self, date: datetime, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load cached GKG data for a single day, downloading if absent.

        Args:
            date: Calendar day to load.
            force_refresh: If True, re-download before loading.

        Returns:
            DataFrame with DatetimeIndex (date) and columns: theme, article_count,
            avg_tone, positive_avg, negative_avg, polarity_avg.
            Returns an empty DataFrame if the data cannot be retrieved.
        """
        path = self.download_gkg_day(date, force=force_refresh)
        if path is None or not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True, compression="gzip")
        except Exception:
            _logger.exception("Failed to read GKG data for %s", date.date())
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Events downloads
    # ------------------------------------------------------------------

    def download_events_day(self, date: datetime, force: bool = False) -> Optional[Path]:
        """
        Download, aggregate, and cache Events data for a single calendar day.

        Fetches up to ``files_per_day`` 15-minute Events CSV zip files and
        aggregates them to a per-EventCode daily summary saved to
        DATA_CACHE_DIR/gdelt/events/YYYYMMDD.events.csv.gz.

        Schema (DatetimeIndex = date):
            event_code, event_root_code, quad_class, num_events,
            num_mentions, num_articles, avg_tone, goldstein_scale_avg

        Args:
            date: Calendar day to download. Time component is ignored.
            force: If True, re-download even when the cache already contains this date.

        Returns:
            Path to the day's ``YYYYMMDD.events.csv.gz`` file, or None if no data
            could be retrieved.
        """
        date_str = date.strftime("%Y%m%d")
        day_file = self._events_dir / f"{date_str}.events.csv.gz"

        if day_file.exists() and not force:
            _logger.debug("Events %s already cached at %s", date.date(), day_file)
            return day_file

        if date < _GDELT_2_START:
            _logger.warning("GDELT 2.0 data starts 2015-02-18; skipping %s", date.date())
            return None

        frames: List[pd.DataFrame] = []

        for slot in self._timeslots:
            url = f"{_GDELT_BASE}{date_str}{slot}.export.CSV.zip"
            df = self._download_events_file(url)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            _logger.warning("No Events data retrieved for %s", date.date())
            return None

        raw = pd.concat(frames, ignore_index=True)
        aggregated = _aggregate_events(raw, date)

        if aggregated.empty:
            _logger.warning("Events aggregation produced no rows for %s", date.date())
            return None

        self._events_dir.mkdir(parents=True, exist_ok=True)
        aggregated.set_index("date").to_csv(day_file, compression="gzip")
        _logger.info("Saved Events %s: %d event-code rows → %s", date.date(), len(aggregated), day_file)
        return day_file

    def download_events_range(
        self,
        start_date: datetime,
        end_date: datetime,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Download Events data for a date range (inclusive on both ends).

        Args:
            start_date: First day to download.
            end_date: Last day to download.
            force: If True, re-download already-cached days.

        Returns:
            Summary dict: ``total``, ``downloaded``, ``skipped``, ``errors``.
        """
        return self._range_download(
            start_date, end_date,
            self.download_events_day, self._events_dir,
            force, "Events", ".events.csv.gz",
        )

    def load_events_day(self, date: datetime, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load cached Events data for a single day, downloading if absent.

        Args:
            date: Calendar day to load.
            force_refresh: If True, re-download before loading.

        Returns:
            DataFrame with DatetimeIndex (date) and columns: event_code,
            event_root_code, quad_class, num_events, num_mentions, num_articles,
            avg_tone, goldstein_scale_avg.
            Returns an empty DataFrame if the data cannot be retrieved.
        """
        path = self.download_events_day(date, force=force_refresh)
        if path is None or not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True, compression="gzip")
        except Exception:
            _logger.exception("Failed to read Events data for %s", date.date())
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_gkg_file(self, url: str) -> Optional[pd.DataFrame]:
        """Download and parse a single 15-minute GKG CSV zip file."""
        content = self._fetch_zip(url)
        if content is None:
            return None
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                fname = z.namelist()[0]
                df = pd.read_csv(
                    z.open(fname),
                    sep="\t",
                    header=None,
                    names=_GKG_COLS,
                    on_bad_lines="skip",
                    low_memory=False,
                    dtype=str,
                    encoding="latin1",
                )
            return df
        except Exception:
            _logger.exception("Failed to parse GKG file from %s", url)
            return None

    def _download_events_file(self, url: str) -> Optional[pd.DataFrame]:
        """Download and parse a single 15-minute Events CSV zip file."""
        content = self._fetch_zip(url)
        if content is None:
            return None
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                fname = z.namelist()[0]
                df = pd.read_csv(
                    z.open(fname),
                    sep="\t",
                    header=None,
                    names=_EVENTS_COLS,
                    on_bad_lines="skip",
                    low_memory=False,
                    dtype=str,
                    encoding="latin1",
                )
            return df
        except Exception:
            _logger.exception("Failed to parse Events file from %s", url)
            return None

    def _fetch_zip(self, url: str) -> Optional[bytes]:
        """Rate-limited GET returning raw zip bytes, or None on error/404."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)

        try:
            _logger.debug("GET %s", url)
            response = self._session.get(url, timeout=60)
            self._last_request_time = time.monotonic()
            if response.status_code == 404:
                _logger.debug("404 (no file) for %s", url)
                return None
            response.raise_for_status()
            return response.content
        except requests.HTTPError as exc:
            _logger.debug("HTTP error for %s: %s", url, exc)
            return None
        except Exception:
            _logger.exception("Failed to fetch %s", url)
            return None

    def _range_download(
        self,
        start_date: datetime,
        end_date: datetime,
        download_fn: Callable[[datetime, bool], Optional[Path]],
        dest_dir: Path,
        force: bool,
        label: str,
        file_suffix: str = ".csv.gz",
    ) -> Dict[str, Any]:
        """Generic date-range bulk download loop."""
        total = downloaded = skipped = errors = 0
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

        # Build set of already-cached dates by scanning YYYYMMDD{file_suffix} filenames
        if not force and dest_dir.exists():
            cached = set()
            for f in dest_dir.glob(f"????????{file_suffix}"):
                try:
                    cached.add(
                        datetime.strptime(f.name[:8], "%Y%m%d").date()
                    )
                except ValueError:
                    pass
        else:
            cached = set()

        _logger.info(
            "Starting %s range download: %s → %s",
            label, current.date(), end.date(),
        )

        while current <= end:
            total += 1

            if current.date() in cached:
                skipped += 1
            else:
                result = download_fn(current, force)
                if result is None:
                    errors += 1
                else:
                    downloaded += 1
                    cached.add(current.date())  # keep in-memory set consistent

            if total % 30 == 0:
                _logger.info(
                    "%s progress: %d days processed — downloaded=%d skipped=%d errors=%d",
                    label, total, downloaded, skipped, errors,
                )

            current += timedelta(days=1)

        _logger.info(
            "%s range complete: downloaded=%d skipped=%d errors=%d / %d total",
            label, downloaded, skipped, errors, total,
        )
        return {"total": total, "downloaded": downloaded, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _select_timeslots(files_per_day: int) -> List[str]:
    """
    Select evenly-spaced 15-minute timeslots for the requested files per day.

    Args:
        files_per_day: Number of files to select. Clamped to [1, 96].

    Returns:
        List of HHMMSS strings (e.g. ``["000000", "060000", "120000", "180000"]``).
    """
    n = max(1, min(files_per_day, 96))
    if n >= 96:
        return list(_ALL_TIMESLOTS)
    step = 96 // n
    return [_ALL_TIMESLOTS[i * step] for i in range(n)]


def _parse_tone(tone_str: Any) -> Dict[str, Optional[float]]:
    """
    Parse a GDELT V2Tone string into tone components.

    V2Tone format: ``Tone,Positive,Negative,Polarity,ActivityRefDensity,SelfGroupDensity,WordCount``

    Args:
        tone_str: Raw V2Tone string from the GKG CSV.

    Returns:
        Dict with keys: ``tone``, ``positive``, ``negative``, ``polarity``.
        All values are None if the string cannot be parsed.
    """
    empty: Dict[str, Optional[float]] = {
        "tone": None, "positive": None, "negative": None, "polarity": None,
    }
    if not isinstance(tone_str, str):
        return empty
    parts = tone_str.split(",")
    if len(parts) < 4:
        return empty
    try:
        return {
            "tone":     float(parts[0]),
            "positive": float(parts[1]),
            "negative": float(parts[2]),
            "polarity": float(parts[3]),
        }
    except ValueError:
        return empty


def _parse_themes(themes_str: Any) -> List[str]:
    """
    Parse a GDELT V2Themes string into a list of theme codes.

    V2Themes format: ``THEME_CODE,charoffset;THEME_CODE2,charoffset2;...``

    Args:
        themes_str: Raw V2Themes string from the GKG CSV.

    Returns:
        List of theme code strings (e.g. ``["ECON_INTEREST_RATE", "FED_RESERVE"]``).
    """
    if not isinstance(themes_str, str) or not themes_str.strip():
        return []
    themes: List[str] = []
    for entry in themes_str.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        code = entry.split(",")[0].strip()
        if code:
            themes.append(code)
    return themes


def _aggregate_gkg(raw: pd.DataFrame, date: datetime) -> pd.DataFrame:
    """
    Aggregate raw GKG records to a per-theme daily summary.

    Expands V2Themes (semicolon-separated) so each theme gets its own row,
    then groups by theme and computes mean tone metrics.

    Args:
        raw: Concatenated raw GKG DataFrames for the day.
        date: Calendar day being aggregated (used to populate the ``date`` column).

    Returns:
        DataFrame with columns: date, theme, article_count,
        avg_tone, positive_avg, negative_avg, polarity_avg.
    """
    if raw.empty:
        return pd.DataFrame()

    # Parse tone into separate numeric columns
    tone_df = raw["V2Tone"].apply(_parse_tone).apply(pd.Series)
    raw = pd.concat([raw.reset_index(drop=True), tone_df.reset_index(drop=True)], axis=1)
    raw = raw.dropna(subset=["tone"])

    if raw.empty:
        return pd.DataFrame()

    # Expand V2Themes — one row per (article × theme)
    raw = raw.copy()
    raw["theme"] = raw["V2Themes"].apply(_parse_themes)
    raw = raw.explode("theme")
    raw = raw.loc[raw["theme"].notna() & (raw["theme"] != "")].copy()

    if raw.empty:
        return pd.DataFrame()

    agg = (
        raw.groupby("theme", sort=False)
        .agg(
            article_count=("tone", "count"),
            avg_tone=("tone", "mean"),
            positive_avg=("positive", "mean"),
            negative_avg=("negative", "mean"),
            polarity_avg=("polarity", "mean"),
        )
        .reset_index()
    )

    agg["date"] = date.strftime("%Y-%m-%d")
    agg = pd.DataFrame(agg[["date"] + [c for c in agg.columns if c != "date"]])
    agg["date"] = pd.to_datetime(agg["date"])
    return agg


def _aggregate_events(raw: pd.DataFrame, date: datetime) -> pd.DataFrame:
    """
    Aggregate raw Events records to a per-EventCode daily summary.

    Groups events by (EventCode, EventRootCode, QuadClass) and sums/averages
    the key quantitative fields.

    Args:
        raw: Concatenated raw Events DataFrames for the day.
        date: Calendar day being aggregated (used to populate the ``date`` column).

    Returns:
        DataFrame with columns: date, event_code, event_root_code, quad_class,
        num_events, num_mentions, num_articles, avg_tone, goldstein_scale_avg.
    """
    if raw.empty:
        return pd.DataFrame()

    for col in ["GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["EventCode"])

    if raw.empty:
        return pd.DataFrame()

    agg = (
        raw.groupby(["EventCode", "EventRootCode", "QuadClass"], sort=False)
        .agg(
            num_events=("GLOBALEVENTID", "count"),
            num_mentions=("NumMentions", "sum"),
            num_articles=("NumArticles", "sum"),
            avg_tone=("AvgTone", "mean"),
            goldstein_scale_avg=("GoldsteinScale", "mean"),
        )
        .reset_index()
    )

    agg = agg.rename(columns={
        "EventCode": "event_code",
        "EventRootCode": "event_root_code",
        "QuadClass": "quad_class",
    })
    quad_numeric: pd.Series = pd.to_numeric(agg["quad_class"], errors="coerce")  # type: ignore[assignment]
    agg["quad_class"] = quad_numeric.astype("Int64")
    agg["date"] = date.strftime("%Y-%m-%d")
    agg = pd.DataFrame(agg[["date"] + [c for c in agg.columns if c != "date"]])
    agg["date"] = pd.to_datetime(agg["date"])
    return agg


# ---------------------------------------------------------------------------
# GDELT 1.0 downloader
# ---------------------------------------------------------------------------

class Gdelt1Downloader(BaseDataDownloader):
    """
    GDELT 1.0 GKG Downloader.

    Downloads daily GKG 1.0 zip files from the GDELT project and caches them
    as-is under DATA_CACHE_DIR/gdelt/gkg/YYYYMMDD.gkg.csv.zip.

    GDELT 1.0 GKG is available from 2013-04-01 to 2015-02-17 (the day before
    GDELT 2.0 went live).  One file per day; no aggregation is performed.

    GKG 1.0 file format: tab-separated, no header, 10 columns:
        DATE, NUMARTS, COUNTS, THEMES, LOCATIONS, PERSONS,
        ORGANIZATIONS, TONE, CAMEOCOUNTRIES, GCAM
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        request_delay: float = _DEFAULT_REQUEST_DELAY,
    ):
        """
        Initialize the GDELT 1.0 downloader.

        Args:
            cache_dir: Root cache directory. Defaults to DATA_CACHE_DIR.
                       GKG zip files are stored under <cache_dir>/gdelt/gkg/.
            request_delay: Minimum seconds between HTTP requests. Default: 0.5.
        """
        super().__init__()
        root = Path(cache_dir) if cache_dir else Path(DATA_CACHE_DIR)
        self._gkg_dir = root / "gdelt" / "gkg"
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "e-trading-research gdelt1-downloader akossyrev@gmail.com",
        })
        self._request_delay = request_delay
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # BaseDataDownloader interface
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """Return the canonical provider name."""
        return "gdelt1"

    def get_supported_intervals(self) -> List[str]:
        """GDELT 1.0 provides news/event data — no OHLCV intervals."""
        return []

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Not supported by GDELT 1.0 — returns an empty DataFrame.

        Args:
            symbol: Unused.
            interval: Unused.
            start_date: Unused.
            end_date: Unused.
            **kwargs: Unused.

        Returns:
            Empty DataFrame.
        """
        del symbol, interval, start_date, end_date, kwargs
        _logger.warning("GDELT 1.0 does not provide OHLCV data. Use download_gkg_day() instead.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # GKG downloads
    # ------------------------------------------------------------------

    def download_gkg_day(self, date: datetime, force: bool = False) -> Optional[Path]:
        """
        Download and cache the GDELT 1.0 GKG zip file for a single calendar day.

        The file is saved as-is (no parsing or aggregation) to
        DATA_CACHE_DIR/gdelt/gkg/YYYYMMDD.gkg.csv.zip.

        Args:
            date: Calendar day to download. Time component is ignored.
            force: If True, re-download even when the cache file already exists.

        Returns:
            Path to the cached zip file, or None if the download failed.
        """
        date_str = date.strftime("%Y%m%d")
        dest = self._gkg_dir / f"{date_str}.gkg.csv.zip"

        if dest.exists() and not force:
            _logger.debug("GKG 1.0 %s already cached at %s", date.date(), dest)
            return dest

        if date < _GDELT_1_GKG_START:
            _logger.warning(
                "GDELT 1.0 GKG starts 2013-04-01; %s is before that date",
                date.date(),
            )
            return None

        url = f"{_GDELT_1_GKG_BASE}{date_str}.gkg.csv.zip"
        content = self._fetch_zip(url)
        if content is None:
            return None

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        _logger.info("Saved GKG 1.0 %s (%d bytes) → %s", date.date(), len(content), dest)
        return dest

    def download_gkg_range(
        self,
        start_date: datetime,
        end_date: datetime,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Download GDELT 1.0 GKG zip files for a date range (inclusive).

        Dates outside 2013-04-01 to 2015-02-17 are skipped with a warning.

        Args:
            start_date: First day to download.
            end_date: Last day to download.
            force: If True, re-download already-cached files.

        Returns:
            Summary dict: ``total``, ``downloaded``, ``skipped``, ``errors``.
        """
        total = downloaded = skipped = errors = 0
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

        _logger.info("Starting GKG 1.0 range download: %s → %s", current.date(), end.date())

        while current <= end:
            total += 1
            dest = self._gkg_dir / f"{current.strftime('%Y%m%d')}.gkg.csv.zip"

            if dest.exists() and not force:
                skipped += 1
            else:
                result = self.download_gkg_day(current, force=force)
                if result is None:
                    errors += 1
                else:
                    downloaded += 1

            if total % 30 == 0:
                _logger.info(
                    "GKG 1.0 progress: %d days — downloaded=%d skipped=%d errors=%d",
                    total, downloaded, skipped, errors,
                )

            current += timedelta(days=1)

        _logger.info(
            "GKG 1.0 range complete: downloaded=%d skipped=%d errors=%d / %d total",
            downloaded, skipped, errors, total,
        )
        return {"total": total, "downloaded": downloaded, "skipped": skipped, "errors": errors}

    def load_gkg_day(self, date: datetime, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load cached GKG 1.0 data for a single day, downloading if absent.

        Reads the zip file and returns a DataFrame with the 10 GKG 1.0 columns.

        Args:
            date: Calendar day to load.
            force_refresh: If True, re-download before loading.

        Returns:
            DataFrame with columns: DATE, NUMARTS, COUNTS, THEMES, LOCATIONS,
            PERSONS, ORGANIZATIONS, TONE, CAMEOCOUNTRIES, GCAM.
            Returns an empty DataFrame if the file cannot be retrieved.
        """
        path = self.download_gkg_day(date, force=force_refresh)
        if path is None or not path.exists():
            return pd.DataFrame()
        try:
            with zipfile.ZipFile(path) as z:
                fname = z.namelist()[0]
                df = pd.read_csv(
                    z.open(fname),
                    sep="\t",
                    header=None,
                    names=_GKG1_COLS,
                    on_bad_lines="skip",
                    low_memory=False,
                    dtype=str,
                )
            return df
        except Exception:
            _logger.exception("Failed to read GKG 1.0 zip from %s", path)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_zip(self, url: str) -> Optional[bytes]:
        """Rate-limited GET returning raw zip bytes, or None on error/404."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)

        try:
            _logger.debug("GET %s", url)
            response = self._session.get(url, timeout=60)
            self._last_request_time = time.monotonic()
            if response.status_code == 404:
                _logger.debug("404 (no file) for %s", url)
                return None
            response.raise_for_status()
            return response.content
        except requests.HTTPError as exc:
            _logger.debug("HTTP error for %s: %s", url, exc)
            return None
        except Exception:
            _logger.exception("Failed to fetch %s", url)
            return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download GDELT data to local cache (v1.0 and v2.0).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "GDELT 1.0 commands (prefix 'v1-'):\n"
            "  v1-gkg-day    Download one day of GKG 1.0 raw zip\n"
            "  v1-gkg-range  Download a date range of GKG 1.0 raw zips\n\n"
            "GDELT 2.0 commands:\n"
            "  gkg-day       Download one day of GKG 2.0 (aggregated Parquet)\n"
            "  gkg-range     Download a date range of GKG 2.0\n"
            "  events-day    Download one day of Events 2.0 (aggregated Parquet)\n"
            "  events-range  Download a date range of Events 2.0\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- GDELT 1.0 subcommands --
    p_v1_gkg_day = subparsers.add_parser("v1-gkg-day", help="[v1.0] Download GKG raw zip for one day")
    _v1_date_group = p_v1_gkg_day.add_mutually_exclusive_group(required=True)
    _v1_date_group.add_argument("--date", type=str, help="ISO date, e.g. 2013-04-01")
    _v1_date_group.add_argument("--yesterday", action="store_true", help="Use yesterday's UTC date")
    p_v1_gkg_day.add_argument("--force", action="store_true", help="Re-download even if cached")

    p_v1_gkg_range = subparsers.add_parser("v1-gkg-range", help="[v1.0] Download GKG raw zips for a date range")
    p_v1_gkg_range.add_argument("--start", type=str, required=True, help="Start date, e.g. 2013-04-01")
    p_v1_gkg_range.add_argument("--end", type=str, required=True, help="End date, e.g. 2015-02-17")
    p_v1_gkg_range.add_argument("--force", action="store_true")

    # -- GDELT 2.0 subcommands --
    p_gkg_day = subparsers.add_parser("gkg-day", help="[v2.0] Download GKG Parquet for one day")
    p_gkg_day.add_argument("--date", type=str, required=True, help="ISO date, e.g. 2020-01-15")
    p_gkg_day.add_argument("--force", action="store_true", help="Re-download even if cached")

    p_gkg_range = subparsers.add_parser("gkg-range", help="[v2.0] Download GKG Parquet for a date range")
    p_gkg_range.add_argument("--start", type=str, required=True, help="Start date, e.g. 2020-01-01")
    p_gkg_range.add_argument("--end", type=str, required=True, help="End date, e.g. 2020-12-31")
    p_gkg_range.add_argument("--force", action="store_true")

    p_events_day = subparsers.add_parser("events-day", help="[v2.0] Download Events Parquet for one day")
    p_events_day.add_argument("--date", type=str, required=True, help="ISO date, e.g. 2020-01-15")
    p_events_day.add_argument("--force", action="store_true")

    p_events_range = subparsers.add_parser("events-range", help="[v2.0] Download Events Parquet for a date range")
    p_events_range.add_argument("--start", type=str, required=True, help="Start date, e.g. 2020-01-01")
    p_events_range.add_argument("--end", type=str, required=True, help="End date, e.g. 2020-12-31")
    p_events_range.add_argument("--force", action="store_true")

    # -- shared options --
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help=f"Cache root directory (default: {DATA_CACHE_DIR})",
    )
    parser.add_argument(
        "--files-per-day", type=int, default=_DEFAULT_FILES_PER_DAY,
        help="[v2.0 only] 15-minute files per day (1=midnight, 4=default, 96=full)",
    )
    parser.add_argument(
        "--request-delay", type=float, default=_DEFAULT_REQUEST_DELAY,
        help="Seconds between HTTP requests (default: 0.5)",
    )

    args = parser.parse_args()

    if args.command.startswith("v1-"):
        dl1 = Gdelt1Downloader(cache_dir=args.cache_dir, request_delay=args.request_delay)

        if args.command == "v1-gkg-day":
            if args.yesterday:
                date = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0, tzinfo=None
                ) - timedelta(days=1)
            else:
                date = datetime.strptime(args.date, "%Y-%m-%d")
            path = dl1.download_gkg_day(date, force=args.force)
            result = {"success": path is not None, "path": str(path) if path else None}
            print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

        elif args.command == "v1-gkg-range":
            start = datetime.strptime(args.start, "%Y-%m-%d")
            end = datetime.strptime(args.end, "%Y-%m-%d")
            summary = dl1.download_gkg_range(start, end, force=args.force)
            print(f"__SCHEDULER_RESULT__:{json.dumps({'success': True, **summary})}")

    else:
        dl2 = GdeltDownloader(
            cache_dir=args.cache_dir,
            files_per_day=args.files_per_day,
            request_delay=args.request_delay,
        )

        if args.command == "gkg-day":
            date = datetime.strptime(args.date, "%Y-%m-%d")
            path = dl2.download_gkg_day(date, force=args.force)
            result = {"success": path is not None, "path": str(path) if path else None}
            print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

        elif args.command == "gkg-range":
            start = datetime.strptime(args.start, "%Y-%m-%d")
            end = datetime.strptime(args.end, "%Y-%m-%d")
            summary = dl2.download_gkg_range(start, end, force=args.force)
            print(f"__SCHEDULER_RESULT__:{json.dumps({'success': True, **summary})}")

        elif args.command == "events-day":
            date = datetime.strptime(args.date, "%Y-%m-%d")
            path = dl2.download_events_day(date, force=args.force)
            result = {"success": path is not None, "path": str(path) if path else None}
            print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

        elif args.command == "events-range":
            start = datetime.strptime(args.start, "%Y-%m-%d")
            end = datetime.strptime(args.end, "%Y-%m-%d")
            summary = dl2.download_events_range(start, end, force=args.force)
            print(f"__SCHEDULER_RESULT__:{json.dumps({'success': True, **summary})}")
