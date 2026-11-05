"""Discover IRAC AORs and retrieve SHA Level 1 products.

IRSA SIA v2 (``spitzer_sha`` collection) is the discovery backend. SIA returns
one record per archive file, matching files whose footprints overlap the search
circle. A positional query identifies AORs (``obs_id`` is the hexadecimal
AORKEY); each selected AOR is then re-queried over its full observation span
(from AOR-level archive metadata when available) so every file of that AOR
covering the target is retained. Archive URLs
(``.../r<AOR>/chN/<bcd|cal|raw|pbcd>/<file>``) define product type and layout.
IRSA truncates at MAXREC without an overflow flag, so queries reaching the limit
are split in time. SIA offers no AOR-level inventory, so completeness is never
asserted; `compare_inventory` checks a plan against the archive's per-AOR
directory index and quality report, or against SHA ZIP downloads.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse
from zipfile import BadZipFile, ZipFile

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

LOGGER = logging.getLogger(__name__)
LIMITATION = (
    "Archive completeness is unverified unless compare_inventory reports a match: "
    "SIA provides no authoritative per-AOR file inventory, so only files whose "
    "footprints overlap the expansion circle during the AOR are found (e.g. frames "
    "of the adjacent field imaged by the other channel pair, ch1/ch3 vs ch2/ch4, "
    "that never cover the target are not retrieved)."
)
# SIA POS matches footprint overlap, so this keeps every frame covering the target.
DEFAULT_EXPANSION_RADIUS = 0.5 * u.arcmin
# IRSA returns exactly MAXREC rows without an overflow flag when truncating, and
# drops very large responses, so queries are split in time to stay below this.
SIA_MAXREC = 20000
# Per-file provenance written between inventory checkpoints (see download_plan).
JOURNAL_NAME = "download_journal.jsonl"
MISSION_MJD = (52880.0, 58900.0)
_MIN_WINDOW = 1e-4  # days
# AOR time coverage (days). Discovered frames are padded by AOR_PAD; archive AOR
# bounds get AOR_EDGE of slack; without bounds, the search grows until AOR_GAP
# days beyond the first/last frame contain no frames, up to MAX_AOR_SPAN.
AOR_PAD = 0.5
AOR_EDGE = 0.01
AOR_GAP = 0.5
MAX_AOR_SPAN = 10.0
TIME_COVERAGE = {
    "aor_metadata": "AOR start/end from SIA AOR-level (DOC/README) records",
    "qa_report": "AOR start/end from the archive QualityAnalysis README",
    "full_mission": "unrestricted time search",
    "gap_search": f"heuristic: no AOR frames within {AOR_GAP} d of its first/last",
    "unbounded": f"AOR frames span more than {MAX_AOR_SPAN} d; search stopped",
}
# Archive subdirectories retained (bcd) or retained as ancillary (cal); raw is
# Level 0 and pbcd is Level 2, both excluded and reported.
INCLUDED_SUBDIRS = ("bcd", "cal")
EXCLUDED_SUBDIRS = {"raw": "level0_raw", "pbcd": "level2_pbcd"}
_ANCILLARY = ("uncertainty", "mask", "ancillary")
OBS_COLUMNS = ("aorkey", "pid", "channel", "date", "mode")
DISCOVERY_COLUMNS = OBS_COLUMNS + (
    "science_bcds",
    "uncertainty_files",
    "mask_files",
    "other_ancillary_files",
    "excluded_files",
    "estimated_bytes",
    "unknown_sizes",
    "ra",
    "dec",
    "mjd_min",
    "mjd_max",
    "time_coverage",
    "completeness",
)
PLAN_COLUMNS = OBS_COLUMNS + (
    "filename",
    "url",
    "product",
    "estimated_bytes",
    "local_path",
    "status",
    "error",
    "completeness",
)
INVENTORY_COLUMNS = OBS_COLUMNS + (
    "url",
    "local_path",
    "product",
    "bytes",
    "sha256",
    "status",
)


class ArchiveError(RuntimeError):
    """An archive query, inventory, or transfer could not be verified."""


class SiaOverflow(ArchiveError):
    """An SIA query reached MAXREC, so its result may be truncated."""


def _text(row, *names: str) -> str:
    columns = {name.lower(): name for name in row.colnames}
    for name in names:
        key = columns.get(name.lower())
        if key is not None:
            value = row[key]
            if value is not None and not np.ma.is_masked(value):
                result = str(value).strip()
                if result.lower() not in ("", "none", "null", "nan", "--"):
                    return result
    return ""


def _identifier(value: object) -> str:
    match = re.fullmatch(r"r?([0-9]+)", str(value).strip())
    if not match or int(match[1]) <= 0:
        raise ValueError(f"Invalid AORKEY/PID: {value!r}")
    return str(int(match[1]))


def _ids(values: Sequence[object] | None) -> set[str]:
    return {_identifier(value) for value in values} if values is not None else set()


def _channels(values: Sequence[int]) -> set[int]:
    result = set(values)
    if not result or not result <= {1, 2, 3, 4}:
        raise ValueError("Select one or more IRAC channels from 1, 2, 3, 4")
    return result


def _url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ArchiveError(f"Missing or unsupported archive product URL: {value!r}")
    return value


def _archive_path(url: str) -> tuple[str, int, str, str] | None:
    """Split an SHA product URL ending in ``r<AOR>/chN/<subdir>/<file>``."""
    match = re.search(
        r"/r([0-9]+)/ch([1-4])/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$",
        unquote(urlparse(url).path),
    )
    if not match:
        return None
    return _identifier(match[1]), int(match[2]), match[3], match[4]


def _aor(row) -> str:
    explicit = _text(row, "reqkey", "aorkey")
    if explicit:
        return _identifier(explicit)
    # spitzer_sha obs_id is the AORKEY in 12-digit hexadecimal. cal/ rows point
    # into another AOR's directory, so obs_id takes precedence over the URL.
    obs_id = _text(row, "obs_id")
    if re.fullmatch(r"[0-9A-Fa-f]{12}", obs_id) and "spitzer_sha" in _text(
        row, "obs_collection", "obs_publisher_did"
    ):
        return _identifier(int(obs_id, 16))
    parsed = _archive_path(_text(row, "access_url", "accessurl"))
    if parsed:
        return parsed[0]
    # SIA identifiers vary; accept only an unambiguous archive AOR path/name.
    candidates = set()
    for name in ("obs_id", "access_url", "obs_publisher_did"):
        candidates.update(
            re.findall(r"(?:^|[/_:])r([0-9]+)(?=$|[/_:])", _text(row, name))
        )
    if len(candidates) != 1:
        raise ArchiveError(
            "Cannot identify one AORKEY in SIA metadata; no AORs were silently omitted."
        )
    return _identifier(candidates.pop())


def identify_channel(row) -> int | None:
    """Identify an IRAC channel from channel, path, or bandpass metadata.

    Returns
    -------
    int or None
        Channel 1--4, or None for explicitly non-IRAC records.

    Raises
    ------
    ArchiveError
        IRAC metadata are ambiguous, unknown, or contradictory.
    """
    instrument = _text(row, "instrument_name", "instrument", "modedisplayname")
    if instrument and not re.search(r"\birac\b", instrument, re.IGNORECASE):
        return None
    candidates = set()
    channel = _text(row, "channel")
    if channel:
        match = re.fullmatch(r"(?:ch)?([1-4])", channel, re.IGNORECASE)
        if not match:
            raise ArchiveError(f"Unknown IRAC channel: {channel!r}")
        candidates.add(int(match[1]))
    for name in ("externalname", "accessurl", "access_url"):
        candidates.update(
            int(value)
            for value in re.findall(r"/ch([1-4])/", _text(row, name), re.IGNORECASE)
        )
    band = _text(row, "wavelength", "energy_bandpassname")
    if band:
        match = re.fullmatch(r"(?:IRAC[\s_-]*|I)([1-4])", band, re.IGNORECASE)
        if match:
            candidates.add(int(match[1]))
        else:
            match = re.fullmatch(
                r"(3\.6|4\.5|5\.8|8(?:\.0)?)(?:\s*(?:um|microns?))?", band
            )
            if match:
                candidates.add({3.6: 1, 4.5: 2, 5.8: 3, 8.0: 4}[float(match[1])])
    if len(candidates) != 1:
        raise ArchiveError(f"Cannot identify a unique IRAC channel (band={band!r})")
    return candidates.pop()


def _float(row, *names: str) -> float:
    value = _text(row, *names)
    try:
        return float(value) if value else np.nan
    except ValueError:
        return np.nan


def _mode(row) -> str:
    mode = _text(row, "modedisplayname", "mode")
    if mode:
        return mode
    parts = re.findall(
        r'"reqmode"\s*:\s*\[\s*"([^"]+)"', _text(row, "instrument_keywords")
    )
    # IRAC images are 32x32 (subarray) or 256x256 (full array) pixels.
    width = _float(row, "s_xel1")
    fov = _float(row, "s_fov")
    if width == 32 or (not np.isfinite(width) and fov < 0.03):
        parts.append("subarray")
    elif width == 256 or (not np.isfinite(width) and fov >= 0.03):
        parts.append("full array")
    return ", ".join(parts)


def _observation(row, channel: int) -> tuple:
    date = _text(row, "scet", "begintime", "date")
    if not date and _text(row, "t_min"):
        date = Time(float(_text(row, "t_min")), format="mjd").isot
    pid = _text(row, "pid", "programid", "program_id", "proposal_id")
    # SIA proposal_id looks like "WASP_C5_50759"; the PID is the final number.
    match = re.search(r"(?:^|_)r?([0-9]+)$", pid)
    return (
        _aor(row),
        _identifier(match[1]) if match else "",
        channel,
        date,
        _mode(row),
    )


def _category(subdir: str, filename: str) -> str:
    if subdir in EXCLUDED_SUBDIRS:
        return EXCLUDED_SUBDIRS[subdir]
    if filename.lower().endswith(".log"):
        return "pipeline_log"
    if subdir not in INCLUDED_SUBDIRS or not filename.lower().endswith(".fits"):
        return "unsupported"
    if subdir == "bcd" and filename.endswith("_bcd.fits"):
        return "science"
    return _product(filename)


def _record(row) -> dict | None:
    """Normalize one SIA row; None for non-IRAC records.

    AOR-level files (documentation/README without a channel directory) return a
    record with ``channel=None`` and the AOR's time span, so callers can use
    them as AOR bounds and report them as excluded.
    """
    url = _text(row, "access_url", "accessurl")
    path = unquote(urlparse(url).path)
    if re.search(r"/r[0-9]+/[^/]+$", path):
        return {
            "channel": None,
            "category": "aor_level",
            "aorkey": _aor(row),
            "url": url,
            "filename": PurePosixPath(path).name,
            "ra": np.nan,
            "dec": np.nan,
            "mjd_min": _float(row, "t_min"),
            "mjd_max": _float(row, "t_max"),
        }
    channel = identify_channel(row)
    if channel is None:
        return None
    observation = _observation(row, channel)
    mjd = _float(row, "t_min")
    if not np.isfinite(mjd) and observation[3]:
        mjd = Time(observation[3]).mjd
    record = dict(
        zip(OBS_COLUMNS, observation),
        url=url,
        ra=_float(row, "s_ra", "ra"),
        dec=_float(row, "s_dec", "dec"),
        mjd_min=mjd,
        mjd_max=_float(row, "t_max") if _text(row, "t_max") else mjd,
    )
    parsed = _archive_path(url) if url else None
    # Dark/flat cal files live in the calibration AOR's directory; locally they
    # belong to the science AOR, as in SHA ZIP downloads.
    if parsed and (
        parsed[1] != channel or (parsed[0] != observation[0] and parsed[2] != "cal")
    ):
        raise ArchiveError(f"Archive URL disagrees with AOR/channel metadata: {url}")
    filename = parsed[3] if parsed else PurePosixPath(urlparse(url).path).name
    subdir = parsed[2] if parsed else ""
    # ObsCore access_estsize is an estimate in kilobytes, not an exact size.
    estimate = _float(row, "access_estsize")
    record.update(
        filename=filename,
        subdir=subdir,
        category=_category(subdir, filename) if parsed else "unsupported",
        estimated_bytes=int(estimate * 1024) if np.isfinite(estimate) else -1,
    )
    return record


def _records(table: Table, excluded: dict) -> list:
    """Normalize rows, counting non-IRAC rows in ``excluded``.

    AOR-level records (``channel=None``) are returned for callers to report.
    """
    records = []
    for row in table:
        record = _record(row)
        if record is None:
            excluded["non_irac"] = excluded.get("non_irac", 0) + 1
        else:
            records.append(record)
    return records


def _merge_record(merged: dict, record: dict) -> None:
    """Deduplicate a product within its science AOR/channel, checking metadata."""
    key = (
        record["aorkey"],
        record["channel"],
        record["url"] or record["filename"],
    )
    previous = merged.get(key)
    if previous is not None:
        for name, value in record.items():
            old = previous[name]
            if old == value or (
                isinstance(old, float)
                and isinstance(value, float)
                and np.isnan(old)
                and np.isnan(value)
            ):
                continue
            raise ArchiveError(
                f"Conflicting archive metadata for {record['url'] or record['filename']} "
                f"in AOR {record['aorkey']}/ch{record['channel']}: "
                f"{name} ({old!r} != {value!r})"
            )
    else:
        merged[key] = record


def _merge(records: list) -> list:
    """Deduplicate by science AOR/channel/URL; conflicting metadata is an error."""
    merged = {}
    for record in records:
        _merge_record(merged, record)
    return sorted(
        merged.values(),
        key=lambda r: (r["aorkey"], r["channel"], r["subdir"], r["filename"], r["url"]),
    )


def _hint(records: list, fallback: SkyCoord | None) -> tuple:
    """Return an AOR's mean position and MJD range from its records."""
    ra = np.array([r["ra"] for r in records], dtype=float)
    dec = np.array([r["dec"] for r in records], dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec)
    if good.any():
        # Average unit vectors to handle RA wrapping.
        points = SkyCoord(ra[good], dec[good], unit="deg").cartesian
        center = SkyCoord(points.mean(), representation_type="cartesian").spherical
        ra0, dec0 = center.lon.to_value(u.deg), center.lat.to_value(u.deg)
    elif fallback is not None:
        ra0, dec0 = fallback.icrs.ra.deg, fallback.icrs.dec.deg
    else:
        ra0 = dec0 = np.nan
    times = np.array(
        [t for r in records for t in (r["mjd_min"], r["mjd_max"])], dtype=float
    )
    times = times[np.isfinite(times)]
    return (
        ra0,
        dec0,
        times.min() if times.size else np.nan,
        times.max() if times.size else np.nan,
    )


def _empty(columns: Sequence[str]) -> Table:
    return Table(names=columns, dtype=["U1"] * len(columns))


def _table(rows: list, columns: Sequence[str]) -> Table:
    return Table(rows=rows, names=columns) if rows else _empty(columns)


def _session(timeout: float, retries: int):
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:
        raise ImportError(
            'Install archive dependencies: pip install ".[download]"'
        ) from exc

    class TimeoutAdapter(HTTPAdapter):
        def send(self, request, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = timeout
            return super().send(request, **kwargs)

    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=False,
    )
    for scheme in ("http://", "https://"):
        session.mount(scheme, TimeoutAdapter(max_retries=retry))
    return session


class ArchiveClient:
    """Access IRSA SIA discovery and SHA archive files over HTTP.

    Parameters
    ----------
    timeout : float
        Connect/read timeout in seconds for each network request.
    retries : int
        Maximum retries after transient failures.
    """

    def __init__(self, timeout: float = 60, retries: int = 3):
        if not np.isfinite(timeout) or timeout <= 0 or retries < 0:
            raise ValueError("timeout must be positive and retries nonnegative")
        self.timeout = timeout
        self.retries = retries
        self.session = _session(timeout, retries)

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self.session.close()

    def query_sia(
        self,
        position: SkyCoord,
        radius: u.Quantity,
        time: tuple[float, float] | None = None,
        maxrec: int = SIA_MAXREC,
    ) -> Table:
        """Query IRAC ``spitzer_sha`` SIA records; levels are not inferred from CALIB.

        Parameters
        ----------
        position : SkyCoord
            Circle center.
        radius : Quantity
            Circle radius; matches files whose footprints overlap the circle.
        time : tuple of float, optional
            Inclusive MJD interval passed as the SIA ``TIME`` constraint.
        maxrec : int
            Row limit. Reaching it raises `SiaOverflow`, because IRSA truncates
            at MAXREC without an overflow flag.
        """
        import requests
        from astroquery.ipac.irsa import IrsaClass
        from pyvo.dal.exceptions import DALAccessError, DALOverflowWarning

        service = IrsaClass()
        # Astroquery forwards this session to PyVO but exposes no public setter.
        original_session = service._session
        service._session = self.session
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DALOverflowWarning)
                kwargs = {}
                if time is not None:
                    kwargs["time"] = (
                        Time(time[0], format="mjd"),
                        Time(time[1], format="mjd"),
                    )
                result = service.query_sia(
                    pos=(position, radius),
                    collection="spitzer_sha",
                    instrument="IRAC",
                    maxrec=maxrec,
                    **kwargs,
                )
        except DALOverflowWarning as exc:
            raise SiaOverflow(f"SIA result truncated: {exc}") from exc
        except (DALAccessError, requests.RequestException) as exc:
            raise ArchiveError(f"SIA query failed: {exc}") from exc
        finally:
            service._session = original_session
            original_session.close()
        if len(result) >= maxrec:
            raise SiaOverflow(f"SIA reached the {maxrec}-row limit")
        return result

    def fetch_text(self, url: str, max_bytes: int = 20_000_000) -> str:
        """GET a small text resource (quality report or directory index).

        Raises
        ------
        ArchiveError
            HTTP failure, or a response larger than ``max_bytes``.
        """
        import requests

        try:
            with self.session.get(
                _url(url), stream=True, timeout=self.timeout
            ) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    content.extend(chunk)
                    if len(content) > max_bytes:
                        raise ArchiveError(f"Response exceeds {max_bytes} bytes: {url}")
        except requests.RequestException as exc:
            raise ArchiveError(f"Archive request failed: {url}: {exc}") from exc
        return bytes(content).decode("utf-8", errors="replace")


def _sia_search(
    client: ArchiveClient,
    position: SkyCoord,
    radius: u.Quantity,
    time: tuple[float, float] | None = None,
) -> list:
    """Run SIA queries, halving the time window until none is truncated.

    Returns a list of tables; overlapping window edges may duplicate rows,
    which `_merge` removes by URL.
    """
    try:
        return [client.query_sia(position, radius, time=time)]
    except SiaOverflow:
        low, high = time if time is not None else MISSION_MJD
        if high - low < _MIN_WINDOW:
            raise ArchiveError(
                f"SIA results remain truncated within MJD {low}-{high}; reduce "
                "the search radius"
            ) from None
        middle = (low + high) / 2
        LOGGER.info("Splitting SIA query at MJD %.5f to avoid truncation", middle)
        return _sia_search(client, position, radius, (low, middle)) + _sia_search(
            client, position, radius, (middle, high)
        )


def _aor_directory(url: str, aor: str) -> str | None:
    """Return the archive ``.../r<AOR>/`` directory URL contained in a file URL."""
    match = re.match(rf"(.+/r{aor}/)", url)
    return match[1] if match else None


def parse_quality_report(text: str) -> dict:
    """Parse an SHA ``QualityAnalysis_<AOR>.README`` file.

    Returns
    -------
    dict
        ``start``/``end`` (UTC MJD, when present) and ``processed``: channel ->
        number of successfully pipeline-processed DCEs (one science BCD each).
        Channels reported with no received DCEs are 0; unreported channels are
        absent.
    """
    report = {"processed": {}}
    for key in ("start", "end"):
        match = re.search(
            rf"Observation {key}:\s*([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}[ T][0-9:.]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            report[key] = Time(match[1].replace(" ", "T"), scale="utc").mjd
    for channel, count in re.findall(
        r"Number of received channel-([1-4]) DCEs:\s*([0-9]+)", text
    ):
        if int(count) == 0:
            report["processed"][int(channel)] = 0
    for channel, count in re.findall(
        r"Number of successfully pipeline-processed channel-([1-4]) DCEs:\s*([0-9]+)",
        text,
    ):
        report["processed"][int(channel)] = int(count)
    return report


def _quality_report(client: ArchiveClient, aor: str, urls: Iterable[str]) -> dict:
    """Fetch and parse an AOR's quality report; empty when unavailable."""
    directory = next((d for d in (_aor_directory(url, aor) for url in urls) if d), None)
    if directory is None:
        return {}
    url = urljoin(directory, f"QualityAnalysis_{aor}.README")
    try:
        text = client.fetch_text(url, max_bytes=1_000_000)
    except ArchiveError as exc:
        LOGGER.info("AOR %s: quality report unavailable (%s)", aor, exc)
        return {}
    return parse_quality_report(text) if isinstance(text, str) else {}


def _aor_records(
    client: ArchiveClient,
    aor: str,
    hint: tuple,
    expansion_radius: u.Quantity,
    excluded: dict | None = None,
    aor_level: Sequence[dict] = (),
) -> tuple[list, str]:
    """Return every file record found for one AOR, and its time coverage.

    The SIA search starts `AOR_PAD` days around the hinted frame times and is
    extended to the AOR's archive start/end: from AOR-level SIA records when
    available, else from the archive QualityAnalysis README. Without either,
    it grows until `AOR_GAP` days beyond the first/last frame contain no
    frames of the AOR, so date-filtered discovery cannot truncate long AORs.

    Returns
    -------
    records : list
        Channel records (AOR-level records are counted in ``excluded``).
    coverage : str
        Key of `TIME_COVERAGE` describing how the AOR span was established.
    """
    ra, dec, mjd_min, mjd_max = hint
    if not (np.isfinite(ra) and np.isfinite(dec)):
        raise ArchiveError(
            f"AOR {aor}: no position available for SIA expansion; rerun "
            "discovery with coordinates"
        )
    excluded = {} if excluded is None else excluded
    center = SkyCoord(ra, dec, unit="deg", frame="icrs")
    found = {}
    level = {record["url"]: record for record in aor_level}

    def absorb(window):
        for table in _sia_search(client, center, expansion_radius, window):
            for record in _records(table, excluded):
                if record["aorkey"] == aor:
                    if record["channel"] is None:
                        level[record["url"]] = record
                    else:
                        _merge_record(found, record)

    if not (np.isfinite(mjd_min) and np.isfinite(mjd_max)):
        absorb(None)
        coverage = "full_mission"
    else:
        low, high = mjd_min - AOR_PAD, mjd_max + AOR_PAD
        absorb((low, high))
        report = None
        while True:
            spans = [
                (r["mjd_min"], r["mjd_max"])
                for r in level.values()
                if np.isfinite(r["mjd_min"]) and np.isfinite(r["mjd_max"])
            ]
            source = "aor_metadata"
            if not spans:
                if report is None:
                    report = _quality_report(
                        client, aor, sorted(record["url"] for record in found.values())
                    )
                if "start" in report and "end" in report:
                    spans = [(report["start"], report["end"])]
                    source = "qa_report"
            if spans:
                coverage = source
                wanted = (
                    min(s[0] for s in spans) - AOR_EDGE,
                    max(s[1] for s in spans) + AOR_EDGE,
                )
            else:
                coverage = "gap_search"
                _, _, first, last = _hint(list(found.values()), None)
                first = np.nanmin([first, mjd_min])
                last = np.nanmax([last, mjd_max])
                wanted = (first - AOR_GAP, last + AOR_GAP)
            segments = [(wanted[0], low)] if wanted[0] < low else []
            segments += [(high, wanted[1])] if wanted[1] > high else []
            if not segments:
                break
            if max(high, wanted[1]) - min(low, wanted[0]) > MAX_AOR_SPAN:
                coverage = "unbounded"
                break
            for segment in segments:
                absorb(segment)
            low, high = min(low, wanted[0]), max(high, wanted[1])
    if coverage in ("gap_search", "unbounded"):
        LOGGER.warning(
            "AOR %s: archive AOR start/end unavailable; time coverage is %s (%s)",
            aor,
            coverage,
            TIME_COVERAGE[coverage],
        )
    if level:
        excluded["aor_level"] = excluded.get("aor_level", 0) + len(level)
    return list(found.values()), coverage


def _pair_counts(records: list) -> dict:
    counts = dict.fromkeys(
        ("science", "uncertainty", "mask", "ancillary", "excluded"), 0
    )
    for record in records:
        category = record["category"]
        counts[category if category in counts else "excluded"] += 1
    return counts


def discover_observations(
    *,
    position: SkyCoord | None = None,
    target: str | None = None,
    radius: u.Quantity = 0.5 * u.arcmin,
    pids: Sequence[object] | None = None,
    aorkeys: Sequence[object] | None = None,
    channels: Sequence[int] = (1, 2, 3, 4),
    start: str | None = None,
    end: str | None = None,
    expansion_radius: u.Quantity = DEFAULT_EXPANSION_RADIUS,
    client: ArchiveClient | None = None,
) -> Table:
    """Discover AOR/channel candidates without downloading FITS data.

    Parameters
    ----------
    position : SkyCoord, optional
        Explicit ICRS position; bypasses name resolution even with a target label.
    target : str, optional
        Name to resolve only when position is absent.
    radius : Quantity
        Positive angular search radius used to identify AORs. SIA matches
        files whose footprints overlap the circle, not only frame centers.
    pids, aorkeys : sequence, optional
        Program IDs and AORKEYs. Criteria intersect; IDs within a list are ORed.
        They filter positional results (SIA has no PID/AOR parameter).
    channels : sequence of int
        IRAC channels to retain.
    start, end : str, optional
        Inclusive UTC bounds on AOR/channel date-range overlap. Date-only end
        bounds mean midnight; use an explicit timestamp for a different bound.
        With SIA they also constrain the server query. All exposures of
        selected AOR/channels are retained, not clipped: each AOR is
        re-queried over its full archive start/end (see `TIME_COVERAGE`).
    expansion_radius : Quantity
        SIA radius around each AOR's discovered frames, queried over the AOR's
        time span to gather all of its files (default 30 arcsec). Frames that
        never overlap this circle, such as the adjacent field imaged by the
        other channel pair (ch1/ch3 vs ch2/ch4), are not retrieved.
    client : ArchiveClient, optional
        Injectable archive client for tests or reuse.

    Returns
    -------
    Table
        One row per matching AOR/channel with product counts and size estimates.
        Counts describe returned metadata, not an authoritative inventory.
    """
    wanted_channels = _channels(channels)
    wanted_pids, wanted_aors = _ids(pids), _ids(aorkeys)
    for name, value in (("radius", radius), ("expansion_radius", expansion_radius)):
        degrees = value.to_value(u.deg)
        if not np.isfinite(degrees) or degrees <= 0:
            raise ValueError(f"{name} must be a positive finite angle")
    lower = Time(start).mjd if start else -np.inf
    upper = Time(end).mjd if end else np.inf
    if lower > upper:
        raise ValueError("start must not be after end")
    if position is None and target:
        from astropy.coordinates import name_resolve
        from astropy.utils.data import conf

        try:
            with conf.set_temp("remote_timeout", 30):
                position = SkyCoord.from_name(target)
        except name_resolve.NameResolveError as exc:
            raise ArchiveError(
                f"Cannot resolve {target!r}; supply explicit RA/Dec coordinates. "
                "Coordinate searches need no resolver."
            ) from exc
    if position is None:
        raise ValueError(
            "Discovery requires coordinates or a target name (IRSA SIA is "
            "positional); PIDs/AORKEYs then filter the positional results."
        )
    owned = client is None
    client = client or ArchiveClient()
    excluded = {}
    coverage = {}
    try:
        window = None
        # Date bounds are applied server-side to reduce result volume; whole
        # AORs are still retrieved by the expansion below.
        if start or end:
            window = (max(lower, MISSION_MJD[0]), min(upper, MISSION_MJD[1]))
        tables = _sia_search(client, position.icrs, radius, window)
        found = [record for table in tables for record in _records(table, excluded)]
        candidates = {record["aorkey"] for record in found}
        if wanted_aors:
            candidates &= wanted_aors
        # Re-query each AOR over its full time span before filtering
        # channels/dates, so the summary describes whole AORs. Unnecessary
        # when the unwindowed search already covered the expansion circle.
        expand = window is not None or expansion_radius > radius
        records = []
        for aor in sorted(candidates):
            hits = [
                record
                for record in found
                if record["aorkey"] == aor and record["channel"] is not None
            ]
            levels = [
                record
                for record in found
                if record["aorkey"] == aor and record["channel"] is None
            ]
            if expand:
                # The expansion re-finds the initial files and counts AOR-level
                # files itself, so they are not counted twice.
                more, coverage[aor] = _aor_records(
                    client,
                    aor,
                    _hint(hits + levels, position),
                    expansion_radius,
                    excluded,
                    levels,
                )
                hits += more
            else:
                coverage[aor] = "full_mission"
                if levels:
                    excluded["aor_level"] = excluded.get("aor_level", 0) + len(
                        {record["url"] for record in levels}
                    )
            records += hits
        records = _merge(
            [
                record
                for record in records
                if record["channel"] in wanted_channels
                and (not wanted_aors or record["aorkey"] in wanted_aors)
            ]
        )
        by_aor = {}
        for record in records:
            by_aor.setdefault(record["aorkey"], []).append(record)
        rows = []
        for aor, aor_records in sorted(by_aor.items()):
            ra, dec, mjd_min, mjd_max = _hint(aor_records, None)
            for channel in sorted({record["channel"] for record in aor_records}):
                group = [r for r in aor_records if r["channel"] == channel]
                programs = {r["pid"] for r in group if r["pid"]}
                if len(programs) > 1:
                    raise ArchiveError(f"Conflicting PIDs for AOR {aor}: {programs}")
                if wanted_pids:
                    if not programs:
                        raise ArchiveError(
                            f"Missing PID for AOR {aor}; cannot apply PID filter. "
                            "Filter by AORKEY instead."
                        )
                    if not programs & wanted_pids:
                        continue
                dates = [Time(r["date"]).mjd for r in group if r["date"]]
                if start or end:
                    if len(dates) != len(group):
                        raise ArchiveError(
                            f"Missing date for AOR {aor}; cannot apply date filter"
                        )
                    if max(dates) < lower or min(dates) > upper:
                        continue
                counts = _pair_counts(group)
                if not counts["science"]:
                    LOGGER.warning(
                        "AOR %s/ch%d: archive metadata lists no science BCDs",
                        aor,
                        channel,
                    )
                for record in group:
                    if record["category"] not in (
                        "science",
                        "uncertainty",
                        "mask",
                        "ancillary",
                    ):
                        excluded[record["category"]] = (
                            excluded.get(record["category"], 0) + 1
                        )
                kept = [r for r in group if r["category"] in ("science",) + _ANCILLARY]
                rows.append(
                    (
                        aor,
                        next(iter(programs), ""),
                        channel,
                        Time(min(dates), format="mjd").isot if dates else "",
                        ", ".join(sorted({r["mode"] for r in group if r["mode"]})),
                        counts["science"],
                        counts["uncertainty"],
                        counts["mask"],
                        counts["ancillary"],
                        counts["excluded"],
                        sum(max(0, r["estimated_bytes"]) for r in kept),
                        sum(r["estimated_bytes"] < 0 for r in kept),
                        ra,
                        dec,
                        mjd_min,
                        mjd_max,
                        coverage.get(aor, "full_mission"),
                        "unverified",
                    )
                )
        result = _table(rows, DISCOVERY_COLUMNS)
        result.meta.update(
            expansion_radius_deg=float(expansion_radius.to_value(u.deg)),
            excluded_products=dict(sorted(excluded.items())),
            time_coverage_meaning=TIME_COVERAGE,
            completeness="unverified",
            limitation=LIMITATION,
        )
        missing_aors = sorted(wanted_aors - {str(row["aorkey"]) for row in result})
        result.meta["requested_aors_without_matches"] = missing_aors
        if missing_aors:
            LOGGER.warning(
                "Requested AORs without matches after filters: %s",
                ", ".join(missing_aors),
            )
        return result
    finally:
        if owned:
            client.close()


def plan_download(
    observations: Table,
    output: Path,
    *,
    aorkeys: Sequence[object] | None = None,
    channels: Sequence[int] = (1, 2, 3, 4),
    ancillary: bool = True,
    expansion_radius: u.Quantity | None = None,
    client: ArchiveClient | None = None,
) -> Table:
    """Enumerate AOR-wide Level 1 transfers, independent of discovery footprints.

    Parameters
    ----------
    observations : Table
        Discovery result containing AORKEY/channel pairs and the
        ``ra``/``dec``/``mjd_min``/``mjd_max`` expansion hints.
    output : Path
        Campaign root, not shared between independent light curves.
    aorkeys : sequence, optional
        Subset of discovered AORs; unknown requested AORs are errors.
    channels : sequence of int
        Channels to plan. Requested channels absent from discovery are reported
        informationally, not as incomplete observations: cryogenic IRAC channel
        pairs 1/3 and 2/4 view different fields. Discovered AOR/channel pairs
        without science BCD links remain errors.
    ancillary : bool
        Include uncertainty, mask, and other ``bcd/``/``cal/`` FITS products.
        False retrieves only science ``*_bcd.fits`` files.
    expansion_radius : Quantity, optional
        SIA expansion radius; defaults to the discovery value.
    client : ArchiveClient, optional
        Archive client to reuse.

    Returns
    -------
    Table
        Plan rows. Level 0 (``raw``), Level 2 (``pbcd``), unsupported, and
        not-requested products are counted in ``meta['excluded_products']``;
        ``meta['time_coverage']`` records how each AOR's span was established.
    """
    wanted_channels = _channels(channels)
    if expansion_radius is None:
        expansion_radius = (
            observations.meta.get(
                "expansion_radius_deg", DEFAULT_EXPANSION_RADIUS.to_value(u.deg)
            )
            * u.deg
        )
    known = {_identifier(row["aorkey"]) for row in observations}
    selected = _ids(aorkeys) if aorkeys is not None else known
    if selected - known:
        raise ValueError(f"AORs not discovered: {sorted(selected - known)}")
    pairs = {(aor, channel) for aor in selected for channel in wanted_channels}
    known_pairs = {
        (_identifier(row["aorkey"]), int(row["channel"]))
        for row in observations
        if _identifier(row["aorkey"]) in selected
        and int(row["channel"]) in wanted_channels
    }
    if not pairs:
        raise ValueError("No AOR/channel pairs selected")
    if "ra" not in observations.colnames:
        raise ArchiveError("Discovery table lacks SIA expansion hints; rerun discovery")
    owned = client is None
    client = client or ArchiveClient()
    rows = []
    seen = {}
    found = set()
    excluded = {}
    coverage = {}
    try:
        for aor in sorted(selected):
            group = [row for row in observations if _identifier(row["aorkey"]) == aor]
            hint = (
                np.nanmean([float(row["ra"]) for row in group]),
                np.nanmean([float(row["dec"]) for row in group]),
                np.nanmin([float(row["mjd_min"]) for row in group]),
                np.nanmax([float(row["mjd_max"]) for row in group]),
            )
            records, coverage[aor] = _aor_records(
                client, aor, hint, expansion_radius, excluded
            )
            for record in _merge(records):
                channel = record["channel"]
                if (aor, channel) not in pairs:
                    continue
                category = record["category"]
                if category == "science" or (ancillary and category in _ANCILLARY):
                    pass
                else:
                    reason = category if category not in _ANCILLARY else "not_requested"
                    excluded[reason] = excluded.get(reason, 0) + 1
                    continue
                filename = record["filename"]
                if not re.fullmatch(r"[A-Za-z0-9_.-]+\.fits", filename):
                    raise ArchiveError(f"Unsafe archive filename: {filename!r}")
                url = _url(record["url"])
                local = (
                    PurePosixPath("data")
                    / f"ch{channel}"
                    / f"r{aor}"
                    / f"ch{channel}"
                    / record["subdir"]
                    / filename
                ).as_posix()
                if local in seen:
                    if seen[local] != url:
                        raise ArchiveError(f"Conflicting archive URLs for {local}")
                    continue
                seen[local] = url
                if category == "science":
                    found.add((aor, channel))
                rows.append(
                    tuple(record[name] for name in OBS_COLUMNS)
                    + (
                        filename,
                        url,
                        category,
                        record["estimated_bytes"],
                        local,
                        "planned",
                        "",
                        "unverified",
                    )
                )
        if known_pairs - found:
            raise ArchiveError(
                "No science BCD links for discovered AOR/channels: "
                f"{sorted(known_pairs - found)}"
            )
        if not rows:
            raise ArchiveError("No BCD links for the selected AORs/channels")
        result = _table(rows, PLAN_COLUMNS)
        result.sort(["aorkey", "channel", "local_path"])
        missing = [f"{aor}/ch{channel}" for aor, channel in sorted(pairs - found)]
        result.meta.update(
            output=str(Path(output).resolve()),
            ancillary=bool(ancillary),
            excluded_products=dict(sorted(excluded.items())),
            channels_without_bcd_links=missing,
            time_coverage=dict(sorted(coverage.items())),
            time_coverage_meaning=TIME_COVERAGE,
            completeness="unverified",
            limitation=LIMITATION,
        )
        if missing:
            LOGGER.info(
                "Requested AOR/channels without discovered BCDs: %s. "
                "These channels may observe different fields; absence alone "
                "is not a completeness warning.",
                ", ".join(missing),
            )
        if excluded:
            LOGGER.info("Products not planned (by reason): %s", excluded)
        return result
    finally:
        if owned:
            client.close()


def _associations(paths: dict) -> tuple[int, int]:
    """Count science BCDs lacking a same-stem uncertainty or mask file."""
    names = {PurePosixPath(path).name for path in paths}
    missing_uncertainty = missing_mask = 0
    for path, product in paths.items():
        if product != "science":
            continue
        stem = PurePosixPath(path).name.removesuffix("_bcd.fits")
        missing_uncertainty += not any(
            stem + suffix in names for suffix in ("_bunc.fits", "_cbunc.fits")
        )
        missing_mask += not any(
            stem + suffix in names
            for suffix in ("_bimsk.fits", "_imsk.fits", "_brmsk.fits")
        )
    return missing_uncertainty, missing_mask


def _estimated_size(plan: Table) -> int:
    """Sum archive size estimates (unknown sizes count as zero)."""
    return sum(max(0, int(value)) for value in plan["estimated_bytes"])


def _groups(table: Table) -> dict:
    """Group table rows by (AORKEY, channel) in one pass."""
    groups = {}
    for row in table:
        groups.setdefault((str(row["aorkey"]), int(row["channel"])), []).append(row)
    return dict(sorted(groups.items()))


def observation_summary(plan: Table) -> Table:
    """Summarize planned files/sizes per AOR/channel; never assert completeness."""
    rows = []
    for (aor, channel), group in _groups(plan).items():
        products = [str(row["product"]) for row in group]
        associations = _associations(
            {str(row["local_path"]): str(row["product"]) for row in group}
        )
        sizes = [int(row["estimated_bytes"]) for row in group]
        rows.append(
            (
                aor,
                str(group[0]["pid"]),
                channel,
                min(
                    (str(row["date"]) for row in group if str(row["date"])), default=""
                ),
                ", ".join(sorted({str(row["mode"]) for row in group} - {""})),
                products.count("science"),
                products.count("uncertainty"),
                products.count("mask"),
                products.count("ancillary"),
                *associations,
                sum(max(0, size) for size in sizes),
                sum(size < 0 for size in sizes),
                "unverified",
            )
        )
    return _table(
        rows,
        OBS_COLUMNS
        + (
            "science_bcds",
            "uncertainty_files",
            "mask_files",
            "other_ancillary_files",
            "bcds_without_uncertainty",
            "bcds_without_mask",
            "estimated_bytes",
            "unknown_sizes",
            "completeness",
        ),
    )


def _write_table(table: Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    table.write(temporary, format="ascii.ecsv", overwrite=True)
    _fsync(temporary)
    temporary.replace(path)
    _fsync(path.parent)


def _fsync(path: Path) -> None:
    """Flush a file or directory to stable storage where the OS allows it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_plan(plan: Table, path: Path | None = None) -> Path:
    """Persist an ECSV download plan without transferring any FITS data.

    Parameters
    ----------
    plan : Table
        Planned transfers with output-directory metadata.
    path : Path, optional
        ECSV destination; defaults to output/download_plan.ecsv.

    Returns
    -------
    Path
        Saved plan, reusable with CLI --manifest.
    """
    path = (
        Path(path)
        if path is not None
        else Path(plan.meta["output"]) / "download_plan.ecsv"
    )
    _write_table(plan, path)
    return path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _destination(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ArchiveError(f"Unsafe output path: {relative}")
    destination = root.joinpath(*path.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ArchiveError(f"Output path escapes campaign root: {relative}")
    return destination


def _stream(
    client: ArchiveClient,
    url: str,
    destination: Path,
    replace: bool = False,
    before_replace: Callable[[int, str], None] | None = None,
) -> tuple[int, str]:
    """Stream one file atomically; return its size and SHA-256 digest.

    An existing destination is replaced only when ``replace`` is true.
    ``before_replace(size, sha256)`` runs after the complete file is durably
    written but before it is renamed into place (used for provenance records).
    """
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        for attempt in range(client.retries + 1):
            try:
                with client.session.get(
                    _url(url),
                    stream=True,
                    timeout=client.timeout,
                    headers={"Accept-Encoding": "identity"},
                ) as response:
                    response.raise_for_status()
                    length = response.headers.get("Content-Length")
                    total = int(length) if length is not None else -1
                    count = 0
                    digest = hashlib.sha256()
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                                digest.update(chunk)
                                count += len(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if total >= 0 and count != total:
                        raise ArchiveError(
                            f"Incomplete transfer for {url}: received {count} of "
                            f"{total} bytes"
                        )
                    if count == 0:
                        raise ArchiveError(f"Empty download from {url}")
                    if destination.name.endswith(".fits"):
                        _validate_fits(temporary)
                    if not replace and (
                        destination.exists() or destination.is_symlink()
                    ):
                        # OSError, so it is not retried as a transfer failure.
                        raise FileExistsError(
                            f"Refusing to overwrite existing file: {destination}"
                        )
                    if before_replace is not None:
                        before_replace(count, digest.hexdigest())
                    temporary.replace(destination)
                    return count, digest.hexdigest()
            except (requests.RequestException, ArchiveError) as exc:
                if (
                    isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    and exc.response.status_code not in (429, 500, 502, 503, 504)
                ):
                    raise ArchiveError(f"HTTP download failed: {url}: {exc}") from exc
                if attempt == client.retries:
                    raise ArchiveError(
                        f"Download failed after {attempt + 1} attempts: {url}: {exc}"
                    ) from exc
                LOGGER.warning("Retrying interrupted download: %s (%s)", url, exc)
                time.sleep(min(2**attempt, 8))
    finally:
        temporary.unlink(missing_ok=True)
    raise ArchiveError(f"No download attempt made for {url}")


def _validate_fits(path: Path) -> None:
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyError

    try:
        with fits.open(path, mode="readonly", memmap=True) as hdus:
            hdus.verify("exception")
            for hdu in hdus:
                info = hdu.fileinfo()
                if info and path.stat().st_size < info["datLoc"] + info["datSpan"]:
                    raise ArchiveError(f"Truncated FITS data: {path}")
    except (OSError, ValueError, VerifyError) as exc:
        raise ArchiveError(f"Invalid FITS product: {path}: {exc}") from exc


def _product(name: str) -> str:
    if name.endswith("_bcd.fits"):
        return "science"
    if name.endswith(("_bunc.fits", "_cbunc.fits")):
        return "uncertainty"
    if name.endswith(("_bimsk.fits", "_imsk.fits", "_brmsk.fits")):
        return "mask"
    return "ancillary"


def _verified(row, root: Path) -> bool:
    path = _destination(root, str(row["local_path"]))
    return (
        path.is_file()
        and path.stat().st_size == int(row["bytes"])
        and _hash(path) == str(row["sha256"])
    )


def _validate_plan(plan: Table) -> None:
    missing = set(PLAN_COLUMNS) - set(plan.colnames)
    if missing or not isinstance(plan.meta.get("output"), str):
        raise ArchiveError(
            "Invalid download plan: expected output-directory metadata and columns "
            f"{sorted(missing)}"
        )
    if not len(plan):
        raise ArchiveError("Download plan is empty")
    root = Path(plan.meta["output"])
    paths = {}
    for row in plan:
        aor = _identifier(row["aorkey"])
        channel = int(row["channel"])
        _channels([channel])
        filename = str(row["filename"])
        parts = PurePosixPath(str(row["local_path"])).parts
        subdir = parts[4] if len(parts) == 6 else ""
        expected = (
            PurePosixPath("data")
            / f"ch{channel}"
            / f"r{aor}"
            / f"ch{channel}"
            / subdir
            / filename
        ).as_posix()
        _destination(root, str(row["local_path"]))
        science = row["product"] == "science"
        if (
            str(row["local_path"]) != expected
            or subdir not in INCLUDED_SUBDIRS
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.fits", filename)
            or science != (subdir == "bcd" and filename.endswith("_bcd.fits"))
            or row["product"] not in ("science",) + _ANCILLARY
        ):
            raise ArchiveError(f"Invalid Level 1 output layout: {row['local_path']}")
        _url(str(row["url"]))
        value = row["estimated_bytes"]
        if int(value) != value or int(value) < -1:
            raise ArchiveError(f"Invalid estimated_bytes: {value!r}")
        if expected in paths and paths[expected] != str(row["url"]):
            raise ArchiveError(f"Conflicting plan entries for {expected}")
        paths[expected] = str(row["url"])


def download_plan(
    plan: Table,
    *,
    client: ArchiveClient | None = None,
    checkpoint_files: int = 100,
    checkpoint_seconds: float = 30.0,
    overwrite_untracked: bool = False,
) -> Table:
    """Explicitly transfer a plan, persisting manifests and local file hashes.

    Parameters
    ----------
    plan : Table
        Result of `plan_download` or an ECSV plan read from disk. Its output
        metadata identifies the campaign directory. Rerunning skips only files
        verified against the existing inventory's size and local SHA-256.
    client : ArchiveClient, optional
        HTTP client to reuse.
    checkpoint_files, checkpoint_seconds : int, float
        The manifest and inventory are atomically rewritten after this many
        processed files or seconds (whichever comes first), and always on
        completion, failure, or interruption. Between checkpoints, each
        completed file is first recorded (path, source URL, size, local SHA-256)
        in the append-only ``download_journal.jsonl``. After a hard kill or
        power failure, the next run adopts journaled files whose size and
        SHA-256 still match into the inventory, so they are neither downloaded
        again nor treated as untracked.
    overwrite_untracked : bool
        Existing files absent from ``download_inventory.ecsv`` and from a
        matching journal record (e.g. placed there by hand or another tool) are
        never overwritten by default: they are marked ``conflict`` in the
        manifest, the remaining files are still processed, and an
        `ArchiveError` is then raised. Set True to replace them. Files
        previously recorded in the inventory that fail verification are always
        repaired.

    Returns
    -------
    Table
        File inventory. Local hashes (computed while streaming) detect changes
        on restarts; they are not archive-supplied checksums. Its
        ``meta['ancillary_selection']`` records, per ``AOR/chN``, whether
        uncertainty/mask/ancillary files were requested.

    Raises
    ------
    ArchiveError
        A transfer fails, or untracked files block the plan. The manifest
        status/error is persisted before raising.

    Notes
    -----
    Separate campaigns require separate output directories. This function does
    not launch photometry or claim complete AOR/channel archive inventories.
    """
    _validate_plan(plan)
    if checkpoint_files < 1 or not checkpoint_seconds >= 0:
        raise ValueError("checkpoint_files must be >= 1 and checkpoint_seconds >= 0")
    from tqdm.auto import tqdm

    root = Path(plan.meta["output"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download_manifest.ecsv"
    inventory_path = root / "download_inventory.ecsv"
    inventory = {}
    selection = {}
    if inventory_path.exists():
        existing = Table.read(inventory_path, format="ascii.ecsv")
        selection.update(existing.meta.get("ancillary_selection", {}))
        for item in existing:
            inventory[str(item["local_path"])] = {
                name: item[name] for name in INVENTORY_COLUMNS
            }
    ancillary = plan.meta.get("ancillary")
    planned_keys = {_selection_key(row["aorkey"], row["channel"]) for row in plan}
    changed = False
    # Plans without the selection leave it unrecorded; compare_inventory then
    # requires it explicitly instead of assuming one.
    if ancillary is not None:
        changed = any(selection.get(k) != bool(ancillary) for k in planned_keys)
        selection.update(dict.fromkeys(planned_keys, bool(ancillary)))
    else:
        changed = bool(planned_keys & set(selection))
        for key in planned_keys:
            selection.pop(key, None)
    journal_path = root / JOURNAL_NAME
    recovered = _recover_journal(journal_path, root, inventory)
    if recovered:
        LOGGER.info("Recovered %d journaled file(s) into the inventory", recovered)
        changed = True
    journal = None
    manifest = Table(
        rows=[tuple(row[name] for name in PLAN_COLUMNS) for row in plan],
        names=PLAN_COLUMNS,
        meta=plan.meta.copy(),
    )
    statuses = ["planned"] * len(plan)
    errors = [""] * len(plan)
    conflicts = []

    def persist() -> None:
        nonlocal changed
        manifest["status"] = statuses
        manifest["error"] = errors
        _write_table(manifest, manifest_path)
        if changed:
            _write_table(_inventory_table(inventory, selection), inventory_path)
            changed = False
            # Journal entries are now in the durable inventory.
            if journal is not None and journal.tell():
                # truncate() keeps the offset; rewind so later appends start at
                # zero instead of after a run of NUL bytes.
                journal.seek(0)
                journal.truncate()
                journal.flush()
                os.fsync(journal.fileno())

    def journal_file(row, size: int, digest: str) -> None:
        values = _inventory_record(row, size, digest)
        line = json.dumps({key: _json_value(value) for key, value in values.items()})
        journal.write(line + "\n")
        journal.flush()
        os.fsync(journal.fileno())

    persist()
    # Every verifiable journal record is now in the inventory; drop the rest.
    journal = journal_path.open("w", encoding="utf-8")
    owned = client is None
    client = client or ArchiveClient()
    index = None
    pending = transferred = 0
    last = time.monotonic()
    try:
        with tqdm(total=len(plan), unit="file", desc="SHA download") as progress:
            for index, row in enumerate(plan):
                local, url = str(row["local_path"]), str(row["url"])
                item = inventory.get(local)
                if (
                    item is not None
                    and str(item["url"]) == url
                    and _verified(item, root)
                ):
                    statuses[index] = "skipped"
                elif (
                    item is None
                    and not overwrite_untracked
                    and (
                        _destination(root, local).exists()
                        or _destination(root, local).is_symlink()
                    )
                ):
                    statuses[index] = "conflict"
                    errors[index] = (
                        "Existing file is not in download_inventory.ecsv; not "
                        "overwritten (use overwrite_untracked=True to replace it)"
                    )
                    conflicts.append(local)
                else:
                    statuses[index] = "downloading"
                    try:
                        size, digest = _stream(
                            client,
                            url,
                            _destination(root, local),
                            replace=item is not None or overwrite_untracked,
                            before_replace=partial(journal_file, row),
                        )
                    except (OSError, ValueError, ArchiveError) as exc:
                        statuses[index] = "failed"
                        errors[index] = str(exc)
                        raise ArchiveError(
                            f"AOR {row['aorkey']}/ch{row['channel']}: {exc}"
                        ) from exc
                    inventory[local] = _inventory_record(row, size, digest)
                    statuses[index] = "downloaded"
                    changed = True
                    transferred += size
                    progress.set_postfix_str(
                        tqdm.format_sizeof(transferred, "B", 1024), refresh=False
                    )
                progress.update(1)
                pending += 1
                if (
                    pending >= checkpoint_files
                    or time.monotonic() - last >= checkpoint_seconds
                ):
                    persist()
                    pending, last = 0, time.monotonic()
        index = None
        table = _inventory_table(inventory, selection)
        summary = inventory_summary(table)
        summary.meta["limitation"] = LIMITATION
        _write_table(summary, root / "download_completeness.ecsv")
        if conflicts:
            raise ArchiveError(
                f"{len(conflicts)} existing untracked file(s) were not overwritten "
                f"(first: {conflicts[0]}); move them aside or pass "
                "overwrite_untracked=True (CLI: --overwrite-untracked). See "
                f"{manifest_path}"
            )
        return table
    except BaseException:
        if index is not None and statuses[index] == "downloading":
            statuses[index] = "interrupted"
            errors[index] = "Interrupted before the transfer completed"
        raise
    finally:
        try:
            persist()
        finally:
            journal.close()
            if not journal_path.stat().st_size:
                journal_path.unlink()
            if owned:
                client.close()


def _inventory_record(row, size: int, digest: str) -> dict:
    values = tuple(row[name] for name in OBS_COLUMNS) + (
        str(row["url"]),
        str(row["local_path"]),
        str(row["product"]),
        size,
        digest,
        "verified",
    )
    return dict(zip(INVENTORY_COLUMNS, values))


def _json_value(value):
    if np.ma.is_masked(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _recover_journal(path: Path, root: Path, inventory: dict) -> int:
    """Adopt journaled downloads whose local size and SHA-256 still match.

    Records are written by `download_plan` before each completed file is
    renamed into place, so a record proves provenance only together with
    matching content; anything else stays untracked. Returns the number of
    adopted files.
    """
    if not path.exists():
        return 0
    adopted = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                values = {name: record[name] for name in INVENTORY_COLUMNS}
                local = str(values["local_path"])
                current = inventory.get(local)
                if current is not None and (
                    str(current["url"]),
                    str(current["sha256"]),
                ) == (str(values["url"]), str(values["sha256"])):
                    continue
                if _verified(values, root):
                    inventory[local] = values
                    adopted += 1
            # A torn final line from an interrupted append is ignored.
            except (ValueError, KeyError, TypeError, ArchiveError):
                continue
    return adopted


def _selection_key(aorkey, channel) -> str:
    return f"{_identifier(aorkey)}/ch{int(channel)}"


def _inventory_table(inventory: dict, selection: dict | None = None) -> Table:
    rows = [
        tuple(inventory[path][name] for name in INVENTORY_COLUMNS)
        for path in sorted(inventory)
    ]
    table = _table(rows, INVENTORY_COLUMNS)
    if selection:
        table.meta["ancillary_selection"] = {
            key: bool(value) for key, value in sorted(selection.items())
        }
    return table


def inventory_summary(inventory: Table) -> Table:
    """Count local science, uncertainty and masks per AOR/channel.

    Notes
    -----
    Counts describe only verified local files, not authoritative archive totals.
    Even matching science/uncertainty/mask counts do not prove completeness.
    """
    rows = []
    for (aor, channel), group in _groups(inventory).items():
        paths = {str(row["local_path"]): str(row["product"]) for row in group}
        counts = [
            list(paths.values()).count(kind)
            for kind in ("science", "uncertainty", "mask", "ancillary")
        ]
        missing_uncertainty, missing_mask = _associations(paths)
        rows.append(
            (aor, channel, *counts, missing_uncertainty, missing_mask, "unverified")
        )
    return _table(
        rows,
        (
            "aorkey",
            "channel",
            "science",
            "uncertainty",
            "mask",
            "ancillary",
            "bcds_without_uncertainty",
            "bcds_without_mask",
            "completeness",
        ),
    )


_REFERENCE_PATH = re.compile(r"(?:^|/)r([0-9]+)/ch([1-4])/bcd/([A-Za-z0-9_.-]+)$")
_HREF = re.compile(r'href="([^"?#/]+)"', re.IGNORECASE)
COMPARISON_COLUMNS = (
    "aorkey",
    "channel",
    "reference",
    "reference_files",
    "reference_bcds",
    "qa_processed_dces",
    "planned_bcds",
    "expected_files",
    "matched_files",
    "missing_files",
    "extra_files",
    "excluded_files",
    "agreement",
)


def _archive_listing(client: ArchiveClient, directory: str) -> set[str]:
    """List filenames in an archive directory from its HTML index page."""
    text = client.fetch_text(directory)
    names = {unquote(name) for name in _HREF.findall(text)}
    names = {name for name in names if not name.startswith(".")}
    if not names:
        raise ArchiveError(f"No files found in archive directory index: {directory}")
    return names


def _reference_names(reference) -> dict:
    """Map (AORKEY, channel) to ``bcd/`` filenames in ZIPs, trees, or paths."""
    items = [reference] if isinstance(reference, (str, Path)) else list(reference)
    members = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
            members += [p.relative_to(path).as_posix() for p in files]
            zips = [p for p in files if p.suffix.lower() == ".zip"]
        elif str(item).lower().endswith(".zip"):
            zips = [path]
        else:
            members.append(str(item))
            zips = []
        for archive_path in zips:
            try:
                with ZipFile(archive_path) as archive:
                    members += archive.namelist()
            except (OSError, BadZipFile) as exc:
                raise ArchiveError(
                    f"Cannot read SHA ZIP {archive_path}: {exc}"
                ) from exc
    names = {}
    for member in members:
        match = _REFERENCE_PATH.search(member.replace("\\", "/"))
        if match:
            key = (_identifier(match[1]), int(match[2]))
            names.setdefault(key, set()).add(match[3])
    if not names:
        raise ArchiveError("Reference contains no r<AOR>/chN/bcd/<file> entries")
    return names


def compare_inventory(
    plan: Table,
    reference=None,
    *,
    ancillary: bool | None = None,
    client: ArchiveClient | None = None,
) -> Table:
    """Compare planned or downloaded ``bcd/`` filenames with an independent list.

    Parameters
    ----------
    plan : Table
        Download plan or inventory (``aorkey``, ``channel``, ``local_path`` and
        ``url`` columns).
    reference : path or sequence, optional
        None (default) reads, for each planned AOR/channel, the IRSA archive's
        HTML directory index of ``r<AOR>/chN/bcd/`` (the directory holding the
        planned product URLs) and the AOR's ``QualityAnalysis_<AOR>.README``
        processed-DCE count. Otherwise SHA ZIP files, extracted directories
        (searched recursively, including ZIPs), or relative paths containing
        ``r<AOR>/chN/bcd/<file>``.
    ancillary : bool, optional
        Whether uncertainty/mask/ancillary ``bcd/`` files were requested. By
        default it is read from the table: a download inventory's per-AOR/channel
        ``meta['ancillary_selection']``, else a plan's ``meta['ancillary']``.
        If neither is recorded, an `ArchiveError` is raised rather than
        guessing, so BCD-only downloads are not reported as missing files.
    client : ArchiveClient, optional
        Client for archive references.

    Returns
    -------
    Table
        One row per planned AOR/channel. ``missing_files`` are reference files
        the plan should contain but does not; ``extra_files`` are planned but
        absent from the reference; ``excluded_files`` are intentionally not
        planned (logs, unsupported, or not-requested ancillary files).
        ``agreement`` is ``match`` only if nothing is missing or extra and the
        quality-report count (when known) equals the reference BCD count.
        ``meta['details']`` lists the filenames for each category.

    Notes
    -----
    Only ``bcd/`` files are compared: ``cal/`` files are served from separate
    calibration AORs. The archive directory index is HTML intended for
    browsing, not a formally versioned API; parse failures are errors.
    """
    recorded = plan.meta.get("ancillary_selection", {})
    planned, directories, selections = {}, {}, {}
    for row in plan:
        key = (_identifier(row["aorkey"]), int(row["channel"]))
        parts = PurePosixPath(str(row["local_path"])).parts
        planned.setdefault(key, set())
        if len(parts) >= 2 and parts[-2] == "bcd":
            planned[key].add(parts[-1])
            directories.setdefault(key, str(row["url"]).rsplit("/", 1)[0] + "/")
    if not planned:
        raise ArchiveError("Nothing to compare: the plan/inventory is empty")
    for key in planned:
        choice = ancillary
        if choice is None:
            choice = recorded.get(_selection_key(*key), plan.meta.get("ancillary"))
        if choice is None:
            raise ArchiveError(
                f"Ancillary selection for {key[0]}/ch{key[1]} is not recorded; "
                "pass ancillary=True or False to compare_inventory"
            )
        selections[key] = bool(choice)
    reports = {}
    if reference is None:
        owned = client is None
        client = client or ArchiveClient()
        try:
            names = {}
            for key in sorted(planned):
                if key not in directories:
                    raise ArchiveError(f"No bcd/ URL planned for {key[0]}/ch{key[1]}")
                names[key] = _archive_listing(client, directories[key])
                if key[0] not in reports:
                    reports[key[0]] = _quality_report(
                        client, key[0], [directories[key]]
                    )
        finally:
            if owned:
                client.close()
        source = "irsa_archive_index"
    else:
        names = _reference_names(reference)
        source = "local_reference"
    rows, details = [], {}
    for key in sorted(planned):
        aor, channel = key
        available = names.get(key, set())
        expected, excluded = set(), {}
        for name in available:
            category = _category("bcd", name)
            if category == "science" or (selections[key] and category in _ANCILLARY):
                expected.add(name)
            else:
                reason = "not_requested" if category in _ANCILLARY else category
                excluded.setdefault(reason, []).append(name)
        have = planned[key]
        missing, extra = sorted(expected - have), sorted(have - available)
        reference_bcds = sum(_category("bcd", n) == "science" for n in available)
        qa = reports.get(aor, {}).get("processed", {}).get(channel, -1)
        if not available:
            agreement = "not_in_reference"
        elif missing or extra or (qa >= 0 and qa != reference_bcds):
            agreement = "mismatch"
        else:
            agreement = "match" if qa >= 0 else "match_without_qa_count"
        rows.append(
            (
                aor,
                channel,
                source,
                len(available),
                reference_bcds,
                qa,
                sum(_category("bcd", n) == "science" for n in have),
                len(expected),
                len(expected & have),
                len(missing),
                len(extra),
                sum(len(v) for v in excluded.values()),
                agreement,
            )
        )
        details[f"{aor}/ch{channel}"] = {
            "missing": missing,
            "extra": extra,
            "excluded": {k: sorted(v) for k, v in sorted(excluded.items())},
        }
        if agreement != "match":
            LOGGER.warning(
                "AOR %s/ch%d vs %s: %s (%d missing, %d extra, QA BCDs %d, "
                "reference BCDs %d)",
                aor,
                channel,
                source,
                agreement,
                len(missing),
                len(extra),
                qa,
                reference_bcds,
            )
    aors = {aor for aor, _ in planned}
    unplanned = {key for key in names if key[0] in aors and key not in planned}
    for aor, report in reports.items():
        unplanned |= {
            (aor, channel)
            for channel, count in report.get("processed", {}).items()
            if count > 0 and (aor, channel) not in planned
        }
    result = _table(rows, COMPARISON_COLUMNS)
    result.meta.update(
        reference=source,
        details=details,
        unplanned_channels_with_data=[f"{a}/ch{c}" for a, c in sorted(unplanned)],
        note="Only bcd/ files are compared; cal/ files come from calibration AORs.",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run non-interactive discovery/preview or explicitly requested downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="Name to resolve only if RA/Dec are absent")
    parser.add_argument("--ra", help="ICRS RA: decimal degrees or sexagesimal hours")
    parser.add_argument(
        "--dec", help="ICRS Dec: decimal degrees or sexagesimal degrees"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.5,
        help="Search radius in arcminutes; SIA matches overlapping footprints",
    )
    parser.add_argument(
        "--pid", nargs="+", help="Program ID(s), intersected with position/AOR filters"
    )
    parser.add_argument("--aor", nargs="+", help="AORKEY(s)")
    parser.add_argument(
        "--channels", nargs="+", type=int, help="IRAC channels; default: 1 2 3 4"
    )
    parser.add_argument("--start", help="Inclusive UTC date/time lower bound")
    parser.add_argument(
        "--end", help="Inclusive UTC upper bound (date-only = midnight)"
    )
    parser.add_argument(
        "--output", type=Path, help="Campaign directory; default: Spitzer_download"
    )
    parser.add_argument(
        "--expansion-radius",
        type=float,
        default=DEFAULT_EXPANSION_RADIUS.to_value(u.arcmin),
        help="SIA radius (arcmin) around each AOR's frames; files whose footprints "
        "overlap it during the AOR are retrieved",
    )
    parser.add_argument(
        "--bcd-only",
        action="store_true",
        help="Retrieve only science *_bcd.fits, not uncertainty/mask/cal files",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Enumerate full AOR BCD links and save an ECSV plan",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly download the planned BCD products",
    )
    parser.add_argument(
        "--overwrite-untracked",
        action="store_true",
        help="With --download, replace existing files that are not recorded in "
        "download_inventory.ecsv (by default they are left untouched)",
    )
    parser.add_argument(
        "--manifest", type=Path, help="Reuse an existing ECSV plan without discovery"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare the planned bcd/ files with the IRSA archive directory index "
        "and QualityAnalysis README of each AOR (metadata only)",
    )
    parser.add_argument(
        "--reference",
        nargs="+",
        type=Path,
        help="Compare the plan with SHA ZIP files or extracted directories instead",
    )
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)
    channels = args.channels if args.channels is not None else [1, 2, 3, 4]
    output = args.output or Path("Spitzer_download")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if bool(args.ra) != bool(args.dec):
        parser.error("--ra and --dec must be supplied together")
    compare = args.compare or args.reference
    LOGGER.warning(LIMITATION)
    client = None
    try:
        if args.manifest:
            selectors = (
                args.target,
                args.ra,
                args.pid,
                args.aor,
                args.start,
                args.end,
                args.bcd_only,
                args.channels,
                args.output,
            )
            if any(selectors):
                parser.error(
                    "--manifest cannot be combined with discovery/product selectors"
                )
            plan = Table.read(args.manifest, format="ascii.ecsv")
            _validate_plan(plan)
        else:
            position = None
            if args.ra:
                unit = (
                    (u.hourangle, u.deg)
                    if ":" in args.ra or "h" in args.ra
                    else (u.deg, u.deg)
                )
                position = SkyCoord(args.ra, args.dec, unit=unit, frame="icrs")
            client = ArchiveClient(args.timeout, args.retries)
            observations = discover_observations(
                position=position,
                target=args.target,
                radius=args.radius * u.arcmin,
                pids=args.pid,
                aorkeys=args.aor,
                channels=channels,
                start=args.start,
                end=args.end,
                expansion_radius=args.expansion_radius * u.arcmin,
                client=client,
            )
            observations.pprint(max_lines=-1, max_width=-1)
            _write_table(observations, output / "discovered_observations.ecsv")
            if not args.preview and not args.download and not compare:
                return 0
            if not len(observations):
                raise ArchiveError("No matching observations; no download was planned")
            plan = plan_download(
                observations,
                output,
                channels=channels,
                ancillary=not args.bcd_only,
                client=client,
            )
        observation_summary(plan).pprint(max_lines=-1, max_width=-1)
        print(
            "Estimated size:",
            _estimated_size(plan),
            "bytes; files of unknown size:",
            sum(int(value) < 0 for value in plan["estimated_bytes"]),
        )
        print("Products not planned (by reason):", plan.meta["excluded_products"])
        print("AOR time coverage:", plan.meta.get("time_coverage", {}))
        _write_table(plan, Path(plan.meta["output"]) / "download_plan.ecsv")
        if compare:
            comparison = compare_inventory(plan, args.reference, client=client)
            comparison.pprint(max_lines=-1, max_width=-1)
            print(
                "Channels with data not in the plan:",
                comparison.meta["unplanned_channels_with_data"],
            )
            _write_table(
                comparison, Path(plan.meta["output"]) / "inventory_comparison.ecsv"
            )
        if args.download:
            print(
                inventory_summary(
                    download_plan(
                        plan,
                        client=client,
                        overwrite_untracked=args.overwrite_untracked,
                    )
                )
            )
            from SPCA.directory_structure import make_directory_structure

            print("AOR snippet:", make_directory_structure(plan.meta["output"]))
        else:
            print("Preview only; use --download to retrieve FITS data.")
        return 0
    except (ArchiveError, ValueError, OSError, ImportError) as exc:
        LOGGER.error("%s", exc)
        return 1
    finally:
        if client is not None:
            client.close()
