"""Offline tests for SIA discovery, AOR planning, and byte-preserving transfers."""

import importlib.util
import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch
from zipfile import ZipFile

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import MaskedColumn, Table, vstack

from SPCA import spitzer_download
from SPCA.directory_structure import make_directory_structure
from SPCA.spitzer_download import (
    MAX_AOR_SPAN,
    ArchiveClient,
    ArchiveError,
    compare_inventory,
    discover_observations,
    download_plan,
    identify_channel,
    inventory_summary,
    main,
    observation_summary,
    parse_quality_report,
    plan_download,
    save_plan,
)

HAS_REQUESTS = importlib.util.find_spec("requests") is not None
HAS_ASTROQUERY = importlib.util.find_spec("astroquery") is not None
ROOT = Path(__file__).resolve().parents[1]
BASE = "https://irsa.example/data/SPITZER/SHA/archive/proc/IRAC049600"
T0 = 54768.0  # 2008-10-29T00:00:00 UTC


def image_bytes(shape=(4, 4)):
    stream = BytesIO()
    fits.PrimaryHDU(np.arange(np.prod(shape), dtype=np.float32).reshape(shape)).writeto(
        stream
    )
    return stream.getvalue()


def sia_products(
    entries, aor="1234501", mjd=54768.04, ra=97.6366, dec=29.6723, pid="50517"
):
    """Build rows shaped like live IRSA ``spitzer_sha`` SIA results.

    Entries are ``(channel, subdir, filename[, t_min[, t_max]])``; channel None
    gives an AOR-level file, and a ``cal`` subdir points into another
    (calibration) AOR directory.
    """
    bands = {None: "", 1: "IRAC1", 2: "IRAC2", 3: "IRAC3", 4: "IRAC4"}
    rows = []
    for channel, subdir, name, *times in entries:
        t_min = times[0] if times else mjd
        t_max = times[1] if len(times) > 1 else t_min + 0.001
        directory = "68573184" if subdir == "cal" else aor
        path = f"r{directory}/ch{channel}/{subdir}" if channel else f"r{aor}"
        rows.append(
            (
                f"{int(aor):012X}",
                "spitzer_sha",
                "IRAC",
                bands[channel],
                f"{BASE}/{path}/{name}",
                len(image_bytes()) / 1024,
                ra,
                dec,
                0.01075,
                32,
                '{"reqmode": ["IracMapPC"]}',
                f"WASP_C5_{pid}",
                t_min,
                t_max,
            )
        )
    names = (
        "obs_id",
        "obs_collection",
        "instrument_name",
        "energy_bandpassname",
        "access_url",
        "access_estsize",
        "s_ra",
        "s_dec",
        "s_fov",
        "s_xel1",
        "instrument_keywords",
        "proposal_id",
        "t_min",
        "t_max",
    )
    if not rows:
        return sia_products([(1, "bcd", "x_bcd.fits")], aor)[:0]
    return Table(rows=rows, names=names)


def time_filtered(table):
    """Mock ``query_sia`` honoring the TIME constraint, like IRSA SIA."""

    def query(position, radius, time=None):
        if time is None:
            return table
        low, high = time
        return table[(table["t_max"] >= low) & (table["t_min"] <= high)]

    return query


def long_aor(step=0.05, span=1.5, aor_level=False):
    """A synthetic AOR lasting ``span`` days (default 36 h) of channel 1 frames."""
    times = np.arange(0, span + 1e-9, step)
    entries = [(1, "bcd", f"f{i:04d}_bcd.fits", T0 + t) for i, t in enumerate(times)]
    if aor_level:
        entries.append((None, "", "SPITZER_I_1234501_DOC.fits", T0, T0 + span))
    return sia_products(entries), len(times)


def quality_report(start="2008-10-29 00:00:00.000", end="2008-10-30 12:00:00.000"):
    return (
        "AOR Quality Analysis\n"
        f"Observation Start:     {start}\n"
        f"Observation End:       {end}\n"
        "Number of received channel-1 DCEs:   2\n"
        "Number of successfully pipeline-processed channel-1 DCEs:   2\n"
        "Number of received channel-2 DCEs:   5\n"
        "Number of successfully pipeline-processed channel-2 DCEs:   5\n"
        "Number of received channel-3 DCEs:   0\n"
    )


def response(data, *, status=200, length=True):
    result = Mock()
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.status_code = status
    result.headers = {"Content-Length": str(len(data))} if length else {}
    result.content = data
    result.iter_content.return_value = [data[:100], b"", data[100:]]
    if status != 200:
        import requests

        result.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status}", response=result
        )
    return result


class Fixture(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.client = Mock(timeout=10, retries=0)
        self.client.query_sia.return_value = sia_products(
            [(1, "bcd", "first_bcd.fits")]
        )
        self.client.fetch_text.side_effect = ArchiveError("no quality report")
        self.position = SkyCoord("06h30m32.797s +29d40m20.26s")

    def observations(self, mjd_min=54768.04, mjd_max=54768.041):
        return Table(
            rows=[
                (
                    "1234501",
                    "50517",
                    1,
                    "2008-10-29",
                    "IRAC Subarray",
                    97.6366,
                    29.6723,
                    mjd_min,
                    mjd_max,
                )
            ],
            names=(
                "aorkey",
                "pid",
                "channel",
                "date",
                "mode",
                "ra",
                "dec",
                "mjd_min",
                "mjd_max",
            ),
        )

    def plan(self, ancillary=False):
        return plan_download(
            self.observations(),
            self.root,
            channels=[1],
            ancillary=ancillary,
            client=self.client,
        )

    def discover(self, **kwargs):
        return discover_observations(
            position=self.position, client=self.client, **kwargs
        )


class DiscoveryTests(Fixture):
    def test_explicit_coordinates_bypass_resolver(self):
        with patch.object(
            SkyCoord, "from_name", side_effect=AssertionError("resolver called")
        ):
            found = discover_observations(
                position=self.position, target="WASP-12b", client=self.client
            )
        self.assertEqual(list(found["aorkey"]), ["1234501"])
        self.assertEqual(
            self.client.query_sia.call_args_list,
            [call(self.position.icrs, 0.5 * u.arcmin, time=None)],
        )
        self.assertEqual(found["time_coverage"][0], "full_mission")

    def test_name_resolution_failure_is_actionable(self):
        from astropy.coordinates.name_resolve import NameResolveError

        failure = NameResolveError("unavailable")
        resolver = patch.object(SkyCoord, "from_name", side_effect=failure)
        with resolver, self.assertRaisesRegex(ArchiveError, "explicit RA/Dec"):
            discover_observations(target="WASP-12b", client=self.client)

    def test_sia_requires_coordinates(self):
        with self.assertRaisesRegex(ValueError, "coordinates"):
            discover_observations(aorkeys=[1234501], client=self.client)
        self.client.query_sia.assert_not_called()

    def test_pid_and_aor_lists_filter_positional_results(self):
        self.client.query_sia.return_value = vstack(
            [
                sia_products([(1, "bcd", "a_bcd.fits")]),
                sia_products([(1, "bcd", "b_bcd.fits")], aor="999", pid="70060"),
            ]
        )
        found = self.discover(aorkeys=["1234501", 1234501])
        self.assertEqual(list(found["aorkey"]), ["1234501"])
        found = self.discover(pids=[70060])
        self.assertEqual(list(found["aorkey"]), ["999"])
        self.assertEqual(list(found["pid"]), ["70060"])

    def test_sia_expands_aor_with_time_window_and_finds_other_channels(self):
        initial = sia_products([(1, "bcd", "a_bcd.fits")])
        expanded = sia_products(
            [
                (1, "bcd", "a_bcd.fits"),
                (1, "bcd", "a_bunc.fits"),
                (1, "bcd", "a_bimsk.fits"),
                (3, "bcd", "b_bcd.fits"),
                (3, "cal", "b_sdark.fits"),
                (3, "raw", "b_dce.fits"),
                (3, "pbcd", "b_maic.fits"),
                (None, "", "r_DOC.fits"),
            ]
        )
        expanded = vstack([expanded, sia_products([(1, "bcd", "x_bcd.fits")], "77")])
        self.client.query_sia.side_effect = [initial, expanded]
        found = self.discover(channels=[3], expansion_radius=5 * u.arcmin)
        self.assertEqual(list(found["aorkey"]), ["1234501"])
        self.assertEqual(list(found["channel"]), [3])
        row = found[0]
        self.assertEqual(row["pid"], "50517")
        self.assertEqual(row["science_bcds"], 1)
        self.assertEqual(row["other_ancillary_files"], 1)
        self.assertEqual(row["excluded_files"], 2)
        self.assertEqual(row["estimated_bytes"], 2 * len(image_bytes()))
        self.assertEqual(row["mode"], "IracMapPC, subarray")
        self.assertEqual(row["time_coverage"], "aor_metadata")
        self.assertTrue(str(row["date"]).startswith("2008-10-29"))
        expansion = self.client.query_sia.call_args_list[1]
        self.assertEqual(expansion.args[1], 5 * u.arcmin)
        low, high = expansion.kwargs["time"]
        self.assertAlmostEqual(low, 54768.04 - spitzer_download.AOR_PAD)
        self.assertAlmostEqual(high, 54768.041 + spitzer_download.AOR_PAD)
        self.assertEqual(
            found.meta["excluded_products"],
            {"aor_level": 1, "level0_raw": 1, "level2_pbcd": 1},
        )
        self.assertNotIn("service", found.meta)

    def test_sia_aor_comes_from_hex_obs_id_not_cal_directory(self):
        self.client.query_sia.return_value = sia_products(
            [(2, "bcd", "a_bcd.fits"), (2, "cal", "SPITZER_I2_68573184_sdark.fits")],
            aor="66692096",
        )
        found = self.discover()
        self.assertEqual(list(found["aorkey"]), ["66692096"])
        self.assertEqual(found["other_ancillary_files"][0], 1)

    def test_truncated_sia_queries_are_split_in_time(self):
        from SPCA.spitzer_download import MISSION_MJD, SiaOverflow

        middle = sum(MISSION_MJD) / 2
        early = sia_products([(1, "bcd", "a_bcd.fits")], aor="111")
        late = sia_products([(2, "bcd", "b_bcd.fits")], aor="222")
        self.client.query_sia.side_effect = [SiaOverflow("limit"), early, late]
        found = self.discover()
        self.assertEqual(list(found["aorkey"]), ["111", "222"])
        windows = [c.kwargs["time"] for c in self.client.query_sia.call_args_list]
        self.assertEqual(
            windows, [None, (MISSION_MJD[0], middle), (middle, MISSION_MJD[1])]
        )

    def test_unsplittable_truncation_is_an_error(self):
        from SPCA.spitzer_download import SiaOverflow

        self.client.query_sia.side_effect = SiaOverflow("limit")
        with self.assertRaisesRegex(ArchiveError, "remain truncated"):
            self.discover(start="2008-10-29T00:00:00", end="2008-10-29T00:00:05")

    def test_date_bounds_constrain_sia_server_query(self):
        self.client.query_sia.return_value = sia_products([(3, "bcd", "a_bcd.fits")])
        self.discover(start="2008-01-01", end="2008-12-31")
        low, high = self.client.query_sia.call_args_list[0].kwargs["time"]
        self.assertAlmostEqual(low, 54466.0)
        self.assertAlmostEqual(high, 54831.0)
        # Whole AORs are re-queried after the date-windowed search.
        self.assertEqual(self.client.query_sia.call_count, 2)

    def test_filters_intersect_and_dates_select_whole_aor(self):
        self.client.query_sia.return_value = vstack(
            [
                sia_products([(1, "bcd", "before_bcd.fits", 54466.5)]),
                sia_products([(1, "bcd", "after_bcd.fits", 54801.5)]),
                sia_products([(1, "bcd", "o_bcd.fits")], aor="999", pid="70060"),
            ]
        )
        found = self.discover(
            pids=["50517"],
            aorkeys=["1234501"],
            channels=[1],
            start="2008-06-01",
            end="2008-07-01",
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found["science_bcds"][0], 2)
        self.assertTrue(str(found["date"][0]).startswith("2008-01-01"))

    def test_unmatched_filters_return_empty_metadata_table(self):
        found = self.discover(pids=[70060])
        self.assertEqual(len(found), 0)
        self.assertEqual(found.meta["completeness"], "unverified")

    def test_missing_pid_required_by_filter_is_error(self):
        table = sia_products([(1, "bcd", "a_bcd.fits")])
        table["proposal_id"] = MaskedColumn(table["proposal_id"], mask=[True])
        self.client.query_sia.return_value = table
        with self.assertRaisesRegex(ArchiveError, "Missing PID"):
            self.discover(pids=[50517])

    def test_missing_optional_metadata_is_not_stringified_mask(self):
        table = sia_products([(1, "bcd", "a_bcd.fits")])
        for name in ("proposal_id", "instrument_keywords"):
            table[name] = MaskedColumn(table[name], mask=[True])
        self.client.query_sia.return_value = table
        found = self.discover(aorkeys=[1234501])
        self.assertEqual(str(found["pid"][0]), "")
        self.assertEqual(str(found["mode"][0]), "subarray")

    def test_missing_dates_required_by_filter_is_error(self):
        table = sia_products([(1, "bcd", "a_bcd.fits")])
        table.remove_columns(["t_min", "t_max"])
        self.client.query_sia.return_value = table
        with self.assertRaisesRegex(ArchiveError, "Missing date"):
            self.discover(aorkeys=[1234501], start="2008-01-01")

    def test_channel_identification(self):
        for band, channel in [("IRAC1", 1), ("4.5", 2), ("5.8 um", 3), ("8.0", 4)]:
            row = Table(
                rows=[("IRAC", band)], names=("instrument_name", "energy_bandpassname")
            )[0]
            self.assertEqual(identify_channel(row), channel)
        row = Table(
            rows=[("MIPS", "24")], names=("instrument_name", "energy_bandpassname")
        )[0]
        self.assertIsNone(identify_channel(row))

    def test_conflicting_channels_fail(self):
        row = Table(
            rows=[("IRAC", "IRAC1", "3")],
            names=("instrument_name", "energy_bandpassname", "channel"),
        )[0]
        with self.assertRaisesRegex(ArchiveError, "unique IRAC channel"):
            identify_channel(row)

    def test_sia_aor_can_be_identified_from_archive_path(self):
        table = sia_products([(1, "bcd", "a_bcd.fits")])
        table.remove_column("obs_id")
        self.client.query_sia.return_value = table
        self.assertEqual(list(self.discover()["aorkey"]), ["1234501"])

    def test_sia_without_aor_does_not_silently_omit(self):
        self.client.query_sia.return_value = Table(
            rows=[("IRAC", "IRAC1")], names=("instrument_name", "energy_bandpassname")
        )
        with self.assertRaisesRegex(ArchiveError, "Cannot identify one AORKEY"):
            self.discover()

    def test_invalid_selection_rejected_before_network(self):
        for kwargs in (
            {"channels": [5]},
            {"radius": -1 * u.arcmin},
            {"start": "2009-01-01", "end": "2008-01-01"},
            {"aorkeys": ["bad"]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.discover(**kwargs)
        self.client.query_sia.assert_not_called()

    def test_requested_aor_without_match_is_reported(self):
        with self.assertLogs("SPCA.spitzer_download", level="WARNING"):
            found = self.discover(aorkeys=[1234501, 9999999])
        self.assertEqual(found.meta["requested_aors_without_matches"], ["9999999"])


class AorTimeCoverageTests(Fixture):
    """A date filter that catches only the start of a >32 h AOR."""

    def run_discovery(self, table):
        self.client.query_sia.side_effect = time_filtered(table)
        return self.discover(
            channels=[1], start="2008-10-29T00:00:00", end="2008-10-29T03:00:00"
        )

    def test_date_filter_only_catches_start(self):
        table, count = long_aor()
        first = time_filtered(table)(None, None, (T0, T0 + 0.125))
        self.assertLess(len(first), count)

    def test_aor_level_record_bounds_whole_aor(self):
        table, count = long_aor(aor_level=True)
        found = self.run_discovery(table)
        self.assertEqual(found["science_bcds"][0], count)
        self.assertEqual(found["time_coverage"][0], "aor_metadata")
        self.assertAlmostEqual(found["mjd_max"][0], T0 + 1.5 + 0.001)
        self.client.fetch_text.assert_not_called()

    def test_quality_report_bounds_whole_aor(self):
        table, count = long_aor()
        self.client.fetch_text.side_effect = None
        self.client.fetch_text.return_value = quality_report()
        found = self.run_discovery(table)
        self.assertEqual(found["science_bcds"][0], count)
        self.assertEqual(found["time_coverage"][0], "qa_report")
        self.client.fetch_text.assert_called_once()
        self.assertEqual(
            self.client.fetch_text.call_args.args[0],
            f"{BASE}/r1234501/QualityAnalysis_1234501.README",
        )

    def test_gap_search_recovers_whole_aor_and_warns(self):
        table, count = long_aor()
        with self.assertLogs("SPCA.spitzer_download", level="WARNING") as logs:
            found = self.run_discovery(table)
        self.assertEqual(found["science_bcds"][0], count)
        self.assertEqual(found["time_coverage"][0], "gap_search")
        self.assertTrue(any("gap_search" in line for line in logs.output))
        last = max(c.kwargs["time"][1] for c in self.client.query_sia.call_args_list)
        self.assertGreater(last, T0 + 1.5 + spitzer_download.AOR_GAP - 0.01)

    def test_planning_from_truncated_hint_recovers_whole_aor(self):
        table, count = long_aor(aor_level=True)
        self.client.query_sia.side_effect = time_filtered(table)
        plan = plan_download(
            self.observations(mjd_min=T0, mjd_max=T0 + 0.1),
            self.root,
            channels=[1],
            client=self.client,
        )
        self.assertEqual(len(plan), count)
        self.assertEqual(plan.meta["time_coverage"], {"1234501": "aor_metadata"})
        self.assertEqual(plan.meta["excluded_products"], {"aor_level": 1})

    def test_unbounded_aor_is_reported_not_silently_complete(self):
        table, _ = long_aor(step=0.4, span=3 * MAX_AOR_SPAN)
        with self.assertLogs("SPCA.spitzer_download", level="WARNING") as logs:
            found = self.run_discovery(table)
        self.assertEqual(found["time_coverage"][0], "unbounded")
        self.assertTrue(any("unbounded" in line for line in logs.output))
        self.assertLessEqual(found["mjd_max"][0] - found["mjd_min"][0], 12)

    def test_parse_quality_report(self):
        report = parse_quality_report(quality_report())
        self.assertAlmostEqual(report["start"], T0)
        self.assertAlmostEqual(report["end"], T0 + 1.5)
        self.assertEqual(report["processed"], {1: 2, 2: 5, 3: 0})
        self.assertEqual(parse_quality_report("nothing useful"), {"processed": {}})


class PlanningTests(Fixture):
    entries = (
        (1, "bcd", "a_bcd.fits"),
        (1, "bcd", "a_bunc.fits"),
        (1, "bcd", "a_bimsk.fits"),
        (1, "cal", "a_sdark.fits"),
        (1, "raw", "a_dce.fits"),
        (1, "pbcd", "a_maic.fits"),
        (1, "bcd", "a_bcd.txt"),
        (1, "bcd", "a_ptn.log"),
    )

    def setUp(self):
        super().setUp()
        self.client.query_sia.return_value = sia_products(self.entries)

    def sia_plan(self, **kwargs):
        found = self.discover()
        return plan_download(
            found, self.root, channels=[1], client=self.client, **kwargs
        )

    def test_per_file_plan_keeps_cal_and_reports_exclusions(self):
        plan = self.sia_plan()
        self.assertEqual(
            list(plan["local_path"]),
            [
                "data/ch1/r1234501/ch1/bcd/a_bcd.fits",
                "data/ch1/r1234501/ch1/bcd/a_bimsk.fits",
                "data/ch1/r1234501/ch1/bcd/a_bunc.fits",
                "data/ch1/r1234501/ch1/cal/a_sdark.fits",
            ],
        )
        self.assertEqual(
            plan.meta["excluded_products"],
            {
                "level0_raw": 1,
                "level2_pbcd": 1,
                "pipeline_log": 1,
                "unsupported": 1,
            },
        )
        self.assertEqual(set(plan["estimated_bytes"]), {len(image_bytes())})
        self.assertEqual(set(plan["completeness"]), {"unverified"})
        self.assertIn(plan.meta["time_coverage"]["1234501"], ("gap_search",))
        summary = observation_summary(plan)
        self.assertEqual(summary["estimated_bytes"][0], 4 * len(image_bytes()))
        self.assertEqual(summary["bcds_without_uncertainty"][0], 0)
        self.assertEqual(summary["bcds_without_mask"][0], 0)
        self.assertEqual(summary["completeness"][0], "unverified")

    def test_plan_requeries_aor_time_window(self):
        self.plan()
        expansion = self.client.query_sia.call_args_list[0]
        low, high = expansion.kwargs["time"]
        self.assertAlmostEqual(low, 54768.04 - spitzer_download.AOR_PAD)
        self.assertAlmostEqual(high, 54768.041 + spitzer_download.AOR_PAD)

    def test_bcd_only_excludes_ancillary_as_not_requested(self):
        plan = self.sia_plan(ancillary=False)
        self.assertEqual(list(plan["product"]), ["science"])
        self.assertEqual(plan.meta["excluded_products"]["not_requested"], 3)

    def test_missing_size_estimate_is_unknown(self):
        table = sia_products(self.entries[:1])
        table["access_estsize"] = MaskedColumn(table["access_estsize"], mask=[True])
        self.client.query_sia.return_value = table
        summary = observation_summary(self.sia_plan())
        self.assertEqual(summary["unknown_sizes"][0], 1)
        self.assertEqual(summary["estimated_bytes"][0], 0)

    def test_shared_calibration_is_retained_for_each_science_aor(self):
        entries = [(2, "bcd", "a_bcd.fits"), (2, "cal", "a_sdark.fits")]
        table = vstack(
            [
                sia_products(entries, aor="1234501", pid="60003"),
                sia_products(
                    entries, aor="1234502", pid="60003", mjd=54769.04, ra=97.64
                ),
            ]
        )
        # Repeated SIA rows in each science AOR must still be deduplicated.
        self.client.query_sia.return_value = vstack([table, table])
        found = self.discover(pids=[60003], channels=[2])
        self.assertEqual(list(found["aorkey"]), ["1234501", "1234502"])
        self.assertEqual(list(found["science_bcds"]), [1, 1])
        self.assertEqual(list(found["other_ancillary_files"]), [1, 1])
        plan = plan_download(found, self.root, channels=[2], client=self.client)
        self.assertEqual(len(plan), 4)
        calibration = plan[plan["product"] == "ancillary"]
        self.assertEqual(len(set(calibration["url"])), 1)
        self.assertEqual(
            list(calibration["local_path"]),
            [
                "data/ch2/r1234501/ch2/cal/a_sdark.fits",
                "data/ch2/r1234502/ch2/cal/a_sdark.fits",
            ],
        )
        self.assertEqual(list(calibration["aorkey"]), ["1234501", "1234502"])
        self.assertEqual(list(calibration["channel"]), [2, 2])

    def test_conflicting_same_aor_url_metadata_is_rejected(self):
        for field, value in (
            ("proposal_id", "WASP_C5_60003"),
            ("access_estsize", 999),
            ("t_min", 54768.05),
        ):
            with self.subTest(field=field):
                first = sia_products([(1, "bcd", "a_bcd.fits")])
                second = first.copy()
                second[field][0] = value
                self.client.query_sia.return_value = vstack([first, second])
                with self.assertRaisesRegex(
                    ArchiveError, "Conflicting archive metadata"
                ):
                    self.discover()
                # Expansion must not hide conflicts by keeping only the last row.
                with self.assertRaisesRegex(
                    ArchiveError, "Conflicting archive metadata"
                ):
                    self.plan()

    def test_duplicate_same_aor_url_with_missing_metadata_is_deduplicated(self):
        table = sia_products([(1, "bcd", "a_bcd.fits")])
        table.remove_columns(["s_ra", "s_dec", "access_estsize"])
        self.client.query_sia.return_value = vstack([table, table])
        found = self.discover()
        self.assertEqual(list(found["science_bcds"]), [1])
        self.assertEqual(len(self.plan()), 1)

    def test_plan_without_expansion_hints_is_rejected(self):
        observations = self.observations()
        observations.remove_columns(["ra", "dec", "mjd_min", "mjd_max"])
        with self.assertRaisesRegex(ArchiveError, "expansion hints"):
            plan_download(observations, self.root, client=self.client)

    def test_empty_inventory_is_not_complete(self):
        self.client.query_sia.return_value = sia_products([])
        with self.assertRaisesRegex(ArchiveError, "No science BCD"):
            self.plan()

    def test_unavailable_requested_channels_are_reported(self):
        self.client.fetch_text.side_effect = None
        self.client.fetch_text.return_value = quality_report()
        with self.assertLogs("SPCA.spitzer_download", level="INFO") as logs:
            plan = plan_download(
                self.observations(),
                self.root,
                channels=[1, 4],
                ancillary=False,
                client=self.client,
            )
        self.assertEqual(plan.meta["channels_without_bcd_links"], ["1234501/ch4"])
        self.assertFalse(any(record.levelname == "WARNING" for record in logs.records))

    def test_discovered_channel_losing_bcd_links_remains_error(self):
        observations = self.observations()
        observations["channel"] = [4]
        with self.assertRaisesRegex(
            ArchiveError, "No science BCD links for discovered"
        ):
            plan_download(observations, self.root, client=self.client)

    def test_unknown_aor_rejected(self):
        with self.assertRaisesRegex(ValueError, "not discovered"):
            plan_download(
                self.observations(), self.root, aorkeys=[999], client=self.client
            )

    def test_conflicting_duplicate_paths_rejected(self):
        first = sia_products([(1, "bcd", "a_bcd.fits")])
        second = first.copy()
        second["access_url"] = [first["access_url"][0].replace("049600", "049601")]
        self.client.query_sia.return_value = vstack([first, second])
        with self.assertRaisesRegex(ArchiveError, "Conflicting archive URLs"):
            self.plan()

    @unittest.skipUnless(HAS_REQUESTS, "Install .[download] for HTTP transfers")
    def test_saved_plan_downloads_files_compatible_with_directory_setup(self):
        cube = image_bytes((64, 4, 4))
        self.client.session.get.return_value = response(cube)
        plan = Table.read(save_plan(self.sia_plan()), format="ascii.ecsv")
        inventory = download_plan(plan, client=self.client)
        self.assertEqual(self.client.session.get.call_count, 4)
        self.assertEqual(
            sorted(inventory["product"]),
            ["ancillary", "mask", "science", "uncertainty"],
        )
        for row in plan:
            self.assertEqual((self.root / str(row["local_path"])).read_bytes(), cube)
        self.assertFalse(list(self.root.rglob("*.part")))
        self.assertEqual(make_directory_structure(self.root), "r1234")
        self.client.session.get.reset_mock()
        download_plan(plan, client=self.client)
        self.client.session.get.assert_not_called()


@unittest.skipUnless(HAS_REQUESTS, "Install .[download] for HTTP adapter tests")
class DownloadTests(Fixture):
    def manifest(self):
        return Table.read(self.root / "download_manifest.ecsv", format="ascii.ecsv")

    def test_byte_identical_fullframe_and_subarray(self):
        for shape in ((8, 8), (64, 4, 4)):
            with self.subTest(shape=shape):
                data = image_bytes(shape)
                self.client.session.get.return_value = response(data)
                plan = self.plan()
                inventory = download_plan(plan, client=self.client)
                path = self.root / str(plan["local_path"][0])
                self.assertEqual(path.read_bytes(), data)
                with fits.open(path) as hdus:
                    self.assertEqual(hdus[0].data.shape, shape)
                self.assertEqual(inventory["product"][0], "science")
                self.assertFalse(list(self.root.rglob("*.part")))
                self.assertEqual(make_directory_structure(self.root), "r1234")
                # Remove outputs between intentionally different test inputs.
                (self.root / "download_inventory.ecsv").unlink()
                path.unlink()

    def test_verified_files_skip_network_on_restart(self):
        self.client.session.get.return_value = response(image_bytes())
        plan = self.plan()
        download_plan(plan, client=self.client)
        self.client.session.get.reset_mock()
        inventory = download_plan(
            Table.read(save_plan(plan), format="ascii.ecsv"), client=self.client
        )
        self.client.session.get.assert_not_called()
        self.assertEqual(len(inventory), 1)
        self.assertEqual(self.manifest()["status"][0], "skipped")

    def test_corrupted_completed_file_is_downloaded_again(self):
        self.client.session.get.return_value = response(image_bytes())
        plan = self.plan()
        download_plan(plan, client=self.client)
        (self.root / str(plan["local_path"][0])).write_bytes(b"corrupt")
        self.client.session.get.reset_mock()
        download_plan(plan, client=self.client)
        self.client.session.get.assert_called_once()
        # A file recorded by the downloader is repaired without authorization.
        self.assertEqual(
            (self.root / str(plan["local_path"][0])).read_bytes(), image_bytes()
        )
        self.assertEqual(self.manifest()["status"][0], "downloaded")

    def test_untracked_existing_file_is_not_overwritten_by_default(self):
        self.client.query_sia.return_value = sia_products(
            [(1, "bcd", "a_bcd.fits"), (1, "bcd", "b_bcd.fits")]
        )
        plan = self.plan()
        mine = self.root / str(plan["local_path"][0])
        mine.parent.mkdir(parents=True)
        mine.write_bytes(b"user data")
        self.client.session.get.return_value = response(image_bytes())
        with self.assertRaisesRegex(ArchiveError, "1 existing untracked file"):
            download_plan(plan, client=self.client)
        self.assertEqual(mine.read_bytes(), b"user data")
        self.client.session.get.assert_called_once()
        manifest = self.manifest()
        self.assertEqual(list(manifest["status"]), ["conflict", "downloaded"])
        self.assertIn("not overwritten", manifest["error"][0])
        inventory = Table.read(
            self.root / "download_inventory.ecsv", format="ascii.ecsv"
        )
        self.assertNotIn(str(plan["local_path"][0]), list(inventory["local_path"]))

    def test_untracked_file_replaced_only_when_authorized(self):
        plan = self.plan()
        mine = self.root / str(plan["local_path"][0])
        mine.parent.mkdir(parents=True)
        mine.write_bytes(b"user data")
        data = image_bytes()
        self.client.session.get.return_value = response(data)
        download_plan(plan, client=self.client, overwrite_untracked=True)
        self.assertEqual(mine.read_bytes(), data)
        self.assertEqual(self.manifest()["status"][0], "downloaded")

    def test_stream_refuses_destination_created_during_transfer(self):
        from SPCA.spitzer_download import _stream

        destination = self.root / "x_bcd.fits"

        def appear(*args, **kwargs):
            destination.write_bytes(b"other")
            return response(image_bytes())

        self.client.session.get.side_effect = appear
        with self.assertRaises(FileExistsError):
            _stream(self.client, f"{BASE}/x_bcd.fits", destination)
        self.assertEqual(destination.read_bytes(), b"other")
        self.assertFalse(list(self.root.glob("*.part")))

    def test_inventory_records_ancillary_selection(self):
        self.client.session.get.return_value = response(image_bytes())
        inventory = download_plan(self.plan(ancillary=False), client=self.client)
        expected = {"1234501/ch1": False}
        self.assertEqual(inventory.meta["ancillary_selection"], expected)
        persisted = Table.read(
            self.root / "download_inventory.ecsv", format="ascii.ecsv"
        )
        self.assertEqual(persisted.meta["ancillary_selection"], expected)

    def killed_before_checkpoint(self):
        """Download 2 of 3 files, then die before their inventory checkpoint."""

        class Killed(BaseException):
            pass

        self.client.query_sia.return_value = sia_products(
            [(1, "bcd", f"{name}_bcd.fits") for name in "abc"]
        )
        plan = self.plan()
        self.client.session.get.return_value = response(image_bytes())
        original = spitzer_download._write_table

        def write(table, path):
            if path.name == "download_inventory.ecsv" and len(table):
                raise Killed  # the checkpoint never reaches the disk
            original(table, path)

        with (
            patch.object(spitzer_download, "_write_table", side_effect=write),
            self.assertRaises(Killed),
        ):
            download_plan(
                plan, client=self.client, checkpoint_files=2, checkpoint_seconds=1e9
            )
        inventory = Table.read(
            self.root / "download_inventory.ecsv", format="ascii.ecsv"
        )
        self.assertEqual(len(inventory), 0)
        journal = self.root / spitzer_download.JOURNAL_NAME
        with journal.open("a") as stream:
            stream.write('{"local_path": "data/ch1/r1234501')  # torn append
        paths = [self.root / str(path) for path in plan["local_path"]]
        self.assertTrue(paths[0].exists() and paths[1].exists())
        self.assertFalse(paths[2].exists())
        self.client.session.get.reset_mock()
        return plan, paths

    def test_journaled_downloads_recovered_after_lost_checkpoint(self):
        plan, paths = self.killed_before_checkpoint()
        inventory = download_plan(plan, client=self.client)
        # Only the never-downloaded file is fetched; no --overwrite-untracked.
        self.client.session.get.assert_called_once()
        self.assertEqual(
            list(self.manifest()["status"]), ["skipped", "skipped", "downloaded"]
        )
        self.assertEqual(len(inventory), 3)
        self.assertEqual(
            len(Table.read(self.root / "download_inventory.ecsv", format="ascii.ecsv")),
            3,
        )
        self.assertFalse((self.root / spitzer_download.JOURNAL_NAME).exists())
        for path in paths:
            self.assertEqual(path.read_bytes(), image_bytes())

    def test_journal_recovery_after_successful_checkpoint(self):
        class Killed(BaseException):
            pass

        names = "abcdefg"
        self.client.query_sia.return_value = sia_products(
            [(1, "bcd", f"{name}_bcd.fits") for name in names]
        )
        plan = self.plan()
        self.client.session.get.return_value = response(image_bytes())
        original = spitzer_download._write_table

        def write(table, path):
            # The 3-file checkpoint succeeds; the 6-file one is lost (SIGKILL).
            if path.name == "download_inventory.ecsv" and len(table) >= 6:
                raise Killed
            original(table, path)

        with (
            patch.object(spitzer_download, "_write_table", side_effect=write),
            self.assertRaises(Killed),
        ):
            download_plan(
                plan, client=self.client, checkpoint_files=3, checkpoint_seconds=1e9
            )
        inventory_path = self.root / "download_inventory.ecsv"
        self.assertEqual(len(Table.read(inventory_path, format="ascii.ecsv")), 3)
        journal = (self.root / spitzer_download.JOURNAL_NAME).read_bytes()
        self.assertFalse(journal.startswith(b"\0"))
        self.assertEqual(len(journal.splitlines()), 3)
        paths = [self.root / str(path) for path in plan["local_path"]]
        self.assertFalse(paths[6].exists())
        paths[6].write_bytes(b"manual file")
        self.client.session.get.reset_mock()
        with self.assertRaisesRegex(ArchiveError, "1 existing untracked file"):
            download_plan(plan, client=self.client)
        # Post-checkpoint files d-f are adopted from the journal, not refetched.
        self.client.session.get.assert_not_called()
        self.assertEqual(
            list(self.manifest()["status"]), ["skipped"] * 6 + ["conflict"]
        )
        self.assertEqual(paths[6].read_bytes(), b"manual file")
        inventory = Table.read(inventory_path, format="ascii.ecsv")
        self.assertEqual(
            list(inventory["local_path"]), [str(p) for p in plan["local_path"][:6]]
        )

    def test_journal_does_not_adopt_changed_or_manual_files(self):
        plan, paths = self.killed_before_checkpoint()
        # Same filename and journal record but different content, plus a file
        # placed by hand that the downloader never wrote.
        paths[0].write_bytes(b"replaced by user")
        paths[2].write_bytes(b"manual file")
        with self.assertRaisesRegex(ArchiveError, "2 existing untracked file"):
            download_plan(plan, client=self.client)
        self.client.session.get.assert_not_called()
        self.assertEqual(paths[0].read_bytes(), b"replaced by user")
        self.assertEqual(paths[2].read_bytes(), b"manual file")
        self.assertEqual(
            list(self.manifest()["status"]), ["conflict", "skipped", "conflict"]
        )
        inventory = Table.read(
            self.root / "download_inventory.ecsv", format="ascii.ecsv"
        )
        self.assertEqual(list(inventory["local_path"]), [str(plan["local_path"][1])])

    def test_missing_uncertainty_and_mask_are_reported(self):
        self.client.session.get.return_value = response(image_bytes())
        inventory = download_plan(self.plan(), client=self.client)
        summary = inventory_summary(inventory)
        self.assertEqual(summary["science"][0], 1)
        self.assertEqual(summary["mask"][0], 0)
        self.assertEqual(summary["bcds_without_uncertainty"][0], 1)
        self.assertEqual(summary["bcds_without_mask"][0], 1)
        persisted = Table.read(
            self.root / "download_completeness.ecsv", format="ascii.ecsv"
        )
        self.assertEqual(persisted["completeness"][0], "unverified")

    def test_failure_persists_completed_files_for_restart(self):
        import requests

        self.client.query_sia.return_value = sia_products(
            [(1, "bcd", f"{name}_bcd.fits") for name in "abc"]
        )
        plan = self.plan()
        self.client.session.get.side_effect = [
            response(image_bytes()),
            response(image_bytes()),
            requests.ConnectionError("network down"),
        ]
        with self.assertRaisesRegex(ArchiveError, "network down"):
            download_plan(plan, client=self.client)
        self.assertEqual(
            list(self.manifest()["status"]), ["downloaded", "downloaded", "failed"]
        )
        self.client.session.get.reset_mock(side_effect=True)
        self.client.session.get.return_value = response(image_bytes())
        download_plan(plan, client=self.client)
        self.client.session.get.assert_called_once()
        self.assertEqual(
            list(self.manifest()["status"]), ["skipped", "skipped", "downloaded"]
        )

    def test_interruption_is_recorded(self):
        self.client.session.get.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            download_plan(self.plan(), client=self.client)
        self.assertEqual(self.manifest()["status"][0], "interrupted")
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_bookkeeping_is_batched_for_large_aors(self):
        files = 2000
        self.client.query_sia.return_value = sia_products(
            [(1, "bcd", f"f{i:05d}_bcd.fits") for i in range(files)]
        )
        plan = self.plan()

        def fake_stream(client, url, destination, replace=False, before_replace=None):
            return 10, "0" * 64

        original = spitzer_download._write_table
        with (
            patch.object(spitzer_download, "_stream", side_effect=fake_stream),
            patch.object(
                spitzer_download, "_write_table", side_effect=original
            ) as writer,
        ):
            inventory = download_plan(plan, client=self.client, checkpoint_seconds=1e9)
        self.assertEqual(len(inventory), files)
        # Two tables per 100-file checkpoint plus initial/final/summary writes,
        # instead of two writes per file.
        self.assertLessEqual(writer.call_count, 2 * files // 100 + 4)
        self.assertEqual(set(self.manifest()["status"]), {"downloaded"})

    def test_invalid_checkpoint_settings_rejected(self):
        with self.assertRaises(ValueError):
            download_plan(self.plan(), client=self.client, checkpoint_files=0)

    def test_unsafe_symlink_destination_rejected(self):
        path = self.root / "data/ch1"
        path.parent.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        path.symlink_to(outside / ".." / "..", target_is_directory=True)
        with self.assertRaisesRegex(ArchiveError, "escapes campaign root"):
            download_plan(self.plan(), client=self.client)

    def test_invalid_fits_header_raises_archive_error(self):
        data = bytearray(image_bytes())
        # A nonstandard BITPIX is a FITS verification error, not an HTTP error.
        index = data.find(b"BITPIX  ")
        data[index : index + 80] = b"BITPIX  =                   99".ljust(80)
        self.client.session.get.return_value = response(bytes(data))
        with self.assertRaisesRegex(ArchiveError, "Invalid FITS"):
            download_plan(self.plan(), client=self.client)

    def test_content_length_mismatch_persists_failed_status(self):
        data = image_bytes()
        result = response(data)
        result.headers["Content-Length"] = str(len(data) + 10)
        self.client.session.get.return_value = result
        plan = self.plan()
        with self.assertRaisesRegex(ArchiveError, "Incomplete transfer"):
            download_plan(plan, client=self.client)
        manifest = self.manifest()
        self.assertEqual(manifest["status"][0], "failed")
        self.assertIn("Incomplete transfer", str(manifest["error"][0]))
        self.assertFalse((self.root / str(plan["local_path"][0])).exists())

    def test_truncated_unknown_length_transfer_not_promoted(self):
        data = image_bytes()
        result = response(data, length=False)
        result.iter_content.return_value = [data[:100]]
        self.client.session.get.return_value = result
        plan = self.plan()
        with self.assertRaisesRegex(ArchiveError, "Invalid FITS|Truncated"):
            download_plan(plan, client=self.client)
        self.assertFalse(list(self.root.rglob("*.part")))
        self.assertFalse((self.root / str(plan["local_path"][0])).exists())

    def test_invalid_fits_not_promoted_even_with_matching_length(self):
        data = b"x" * len(image_bytes())
        self.client.session.get.return_value = response(data)
        plan = self.plan()
        with self.assertRaisesRegex(ArchiveError, "Invalid FITS"):
            download_plan(plan, client=self.client)
        self.assertFalse((self.root / str(plan["local_path"][0])).exists())

    def test_interrupted_stream_retries(self):
        import requests

        first = response(image_bytes())
        first.iter_content.side_effect = requests.ConnectionError("interrupted")
        self.client.retries = 1
        self.client.session.get.side_effect = [first, response(image_bytes())]
        with patch("SPCA.spitzer_download.time.sleep"):
            inventory = download_plan(self.plan(), client=self.client)
        self.assertEqual(len(inventory), 1)

    def test_http404_persists_failure_without_retry(self):
        self.client.session.get.return_value = response(b"missing", status=404)
        self.client.retries = 3
        with self.assertRaisesRegex(ArchiveError, "HTTP download failed"):
            download_plan(self.plan(), client=self.client)
        self.assertEqual(self.client.session.get.call_count, 1)
        self.assertEqual(self.manifest()["status"][0], "failed")

    def test_timeout_persists_failure(self):
        import requests

        self.client.session.get.side_effect = requests.Timeout("read timed out")
        with self.assertRaisesRegex(ArchiveError, "timed out"):
            download_plan(self.plan(), client=self.client)
        self.assertEqual(self.manifest()["status"][0], "failed")

    def test_manifest_path_escape_is_rejected(self):
        plan = self.plan()
        plan["local_path"] = ["../escape_bcd.fits"]
        with self.assertRaisesRegex(ArchiveError, "Unsafe output path"):
            download_plan(plan, client=self.client)
        self.client.session.get.assert_not_called()


class CompareInventoryTests(Fixture):
    entries = (
        (1, "bcd", "a_bcd.fits"),
        (1, "bcd", "a_bunc.fits"),
        (1, "bcd", "a_bimsk.fits"),
        (1, "cal", "a_sdark.fits"),
    )

    def setUp(self):
        super().setUp()
        self.client.query_sia.return_value = sia_products(self.entries)

    def zip_reference(self, names):
        path = self.root / "sha.zip"
        with ZipFile(path, "w") as archive:
            for name in names:
                archive.writestr(name, b"")
        return path

    def test_archive_index_reports_missing_extra_excluded_and_unplanned(self):
        plan = self.plan(ancillary=True)
        index = "".join(
            f'<li><a href="{name}"> {name}</a></li>'
            for name in (
                "../",
                "?C=N;O=D",
                "a_bcd.fits",
                "a_bunc.fits",
                "a_bimsk.fits",
                "b_bcd.fits",
                "a_ptn.log",
            )
        )

        def fetch(url, max_bytes=None):
            if url.endswith("/bcd/"):
                return f"<html><ul>{index}</ul></html>"
            if url.endswith(".README"):
                return quality_report()
            raise ArchiveError(f"unexpected {url}")

        self.client.fetch_text.reset_mock()
        self.client.fetch_text.side_effect = fetch
        with self.assertLogs("SPCA.spitzer_download", level="WARNING"):
            result = compare_inventory(plan, client=self.client)
        row = result[0]
        self.assertEqual(row["reference"], "irsa_archive_index")
        self.assertEqual(row["reference_bcds"], 2)
        self.assertEqual(row["qa_processed_dces"], 2)
        self.assertEqual(row["planned_bcds"], 1)
        self.assertEqual(row["expected_files"], 4)
        self.assertEqual(row["matched_files"], 3)
        self.assertEqual(row["missing_files"], 1)
        self.assertEqual(row["extra_files"], 0)
        self.assertEqual(row["excluded_files"], 1)
        self.assertEqual(row["agreement"], "mismatch")
        detail = result.meta["details"]["1234501/ch1"]
        self.assertEqual(detail["missing"], ["b_bcd.fits"])
        self.assertEqual(detail["excluded"], {"pipeline_log": ["a_ptn.log"]})
        self.assertEqual(result.meta["unplanned_channels_with_data"], ["1234501/ch2"])
        self.assertEqual(
            self.client.fetch_text.call_args_list[0].args[0],
            f"{BASE}/r1234501/ch1/bcd/",
        )

    def test_quality_count_disagreement_is_mismatch(self):
        plan = self.plan()
        self.client.fetch_text.side_effect = lambda url, max_bytes=None: (
            '<a href="a_bcd.fits">' if url.endswith("/") else quality_report()
        )
        with self.assertLogs("SPCA.spitzer_download", level="WARNING"):
            result = compare_inventory(plan, client=self.client)
        self.assertEqual(result["missing_files"][0], 0)
        self.assertEqual(result["agreement"][0], "mismatch")

    def test_empty_archive_index_is_error(self):
        plan = self.plan()
        self.client.fetch_text.side_effect = None
        self.client.fetch_text.return_value = '<a href="../">Parent</a>'
        with self.assertRaisesRegex(ArchiveError, "No files found"):
            compare_inventory(plan, client=self.client)

    def test_sha_zip_reference_matches(self):
        plan = self.plan(ancillary=True)
        prefix = "r1234501/ch1"
        reference = self.zip_reference(
            [
                f"{prefix}/bcd/a_bcd.fits",
                f"{prefix}/bcd/a_bunc.fits",
                f"{prefix}/bcd/a_bimsk.fits",
                f"{prefix}/cal/a_sdark.fits",
                f"{prefix}/raw/a_dce.fits",
            ]
        )
        self.client.fetch_text.reset_mock()
        result = compare_inventory(plan, reference)
        self.assertEqual(result["reference"][0], "local_reference")
        self.assertEqual(result["agreement"][0], "match_without_qa_count")
        self.client.fetch_text.assert_not_called()

    def test_bcd_only_plan_reports_not_requested_and_extra(self):
        plan = self.plan(ancillary=False)
        reference = self.root / "extracted"
        (reference / "r1234501/ch1/bcd").mkdir(parents=True)
        (reference / "r1234501/ch1/bcd/a_bunc.fits").write_bytes(b"")
        with self.assertLogs("SPCA.spitzer_download", level="WARNING"):
            result = compare_inventory(plan, [reference])
        detail = result.meta["details"]["1234501/ch1"]
        self.assertEqual(detail["excluded"], {"not_requested": ["a_bunc.fits"]})
        self.assertEqual(detail["extra"], ["a_bcd.fits"])
        self.assertEqual(result["agreement"][0], "mismatch")

    def test_reference_without_bcd_entries_is_error(self):
        with self.assertRaisesRegex(ArchiveError, "no r<AOR>/chN/bcd"):
            compare_inventory(self.plan(), self.zip_reference(["README.txt"]))

    def bcd_only_reference(self):
        prefix = "r1234501/ch1/bcd"
        return self.zip_reference(
            [f"{prefix}/a_bcd.fits", f"{prefix}/a_bunc.fits", f"{prefix}/a_bimsk.fits"]
        )

    def test_downloaded_bcd_only_inventory_does_not_report_missing_ancillary(self):
        self.client.session.get.return_value = response(image_bytes())
        download_plan(self.plan(ancillary=False), client=self.client)
        # Read back from disk, as a user comparing a finished download would.
        inventory = Table.read(
            self.root / "download_inventory.ecsv", format="ascii.ecsv"
        )
        result = compare_inventory(inventory, self.bcd_only_reference())
        detail = result.meta["details"]["1234501/ch1"]
        self.assertEqual(detail["missing"], [])
        self.assertEqual(
            detail["excluded"], {"not_requested": ["a_bimsk.fits", "a_bunc.fits"]}
        )
        self.assertEqual(result["agreement"][0], "match_without_qa_count")

    def test_unrecorded_ancillary_selection_requires_explicit_choice(self):
        self.client.session.get.return_value = response(image_bytes())
        inventory = download_plan(self.plan(ancillary=False), client=self.client)
        inventory.meta.clear()
        with self.assertRaisesRegex(ArchiveError, "pass ancillary=True or False"):
            compare_inventory(inventory, self.bcd_only_reference())
        result = compare_inventory(
            inventory, self.bcd_only_reference(), ancillary=False
        )
        self.assertEqual(result["missing_files"][0], 0)


@unittest.skipUnless(HAS_REQUESTS, "Install .[download] for HTTP adapter tests")
class AdapterTests(Fixture):
    def test_fetch_text_streams_with_timeout(self):
        with patch("SPCA.spitzer_download._session") as session_factory:
            session_factory.return_value.get.return_value = response(b"report text")
            client = ArchiveClient(timeout=12, retries=1)
            self.assertEqual(client.fetch_text("https://x.example/a"), "report text")
        kwargs = client.session.get.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 12)
        self.assertTrue(kwargs["stream"])

    def test_fetch_text_http_error_and_size_limit(self):
        with patch("SPCA.spitzer_download._session") as session_factory:
            get = session_factory.return_value.get
            get.return_value = response(b"missing", status=404)
            client = ArchiveClient()
            with self.assertRaisesRegex(ArchiveError, "Archive request failed"):
                client.fetch_text("https://x.example/a")
            get.return_value = response(b"x" * 300)
            with self.assertRaisesRegex(ArchiveError, "exceeds"):
                client.fetch_text("https://x.example/a", max_bytes=100)

    @unittest.skipUnless(HAS_ASTROQUERY, "Install .[download] for astroquery adapter")
    def test_sia_uses_collection_not_calibration_level(self):
        from astroquery.ipac.irsa import IrsaClass

        table = sia_products([(1, "bcd", "a_bcd.fits")])
        with patch.object(IrsaClass, "query_sia", return_value=table) as query:
            client = ArchiveClient()
            self.addCleanup(client.close)
            result = client.query_sia(self.position, 1 * u.arcmin)
        self.assertEqual(len(result), 1)
        self.assertEqual(query.call_args.kwargs["collection"], "spitzer_sha")
        self.assertNotIn("calib_level", query.call_args.kwargs)

    @unittest.skipUnless(HAS_ASTROQUERY, "Install .[download] for astroquery adapter")
    def test_sia_overflow_warning_is_error(self):
        import warnings

        from astroquery.ipac.irsa import IrsaClass
        from pyvo.dal.exceptions import DALOverflowWarning

        def overflow(**kwargs):
            warnings.warn("truncated", DALOverflowWarning, stacklevel=2)
            return sia_products([(1, "bcd", "a_bcd.fits")])

        with patch.object(IrsaClass, "query_sia", side_effect=overflow):
            client = ArchiveClient()
            self.addCleanup(client.close)
            with self.assertRaisesRegex(ArchiveError, "truncated"):
                client.query_sia(self.position, 1 * u.arcmin)

    @unittest.skipUnless(HAS_ASTROQUERY, "Install .[download] for astroquery adapter")
    def test_sia_row_limit_is_not_accepted_as_complete(self):
        from astroquery.ipac.irsa import IrsaClass

        table = Mock()
        table.__len__ = Mock(return_value=100000)
        with patch.object(IrsaClass, "query_sia", return_value=table):
            client = ArchiveClient()
            self.addCleanup(client.close)
            with self.assertRaisesRegex(ArchiveError, "row limit"):
                client.query_sia(self.position, 1 * u.arcmin)


class InterfaceTests(Fixture):
    def cli(self, *extra):
        return [
            "--ra",
            "97.6366",
            "--dec",
            "29.6723",
            "--aor",
            "1234501",
            "--output",
            str(self.root),
            *extra,
        ]

    def test_cli_default_is_discovery_only(self):
        with (
            patch("SPCA.spitzer_download.ArchiveClient", return_value=self.client),
            patch("SPCA.spitzer_download.download_plan") as download,
            patch("SPCA.spitzer_download.plan_download") as planner,
        ):
            status = main(self.cli())
        self.assertEqual(status, 0)
        planner.assert_not_called()
        download.assert_not_called()
        self.assertTrue((self.root / "discovered_observations.ecsv").exists())

    def test_cli_preview_never_downloads(self):
        with (
            patch("SPCA.spitzer_download.ArchiveClient", return_value=self.client),
            patch("SPCA.spitzer_download.download_plan") as download,
        ):
            status = main(self.cli("--channels", "1", "--bcd-only", "--preview"))
        self.assertEqual(status, 0)
        download.assert_not_called()
        self.assertTrue((self.root / "download_plan.ecsv").exists())

    def test_cli_compare_with_reference_writes_report(self):
        reference = self.root / "sha.zip"
        with ZipFile(reference, "w") as archive:
            archive.writestr("r1234501/ch1/bcd/first_bcd.fits", b"")
        with (
            patch("SPCA.spitzer_download.ArchiveClient", return_value=self.client),
            patch("SPCA.spitzer_download.download_plan") as download,
        ):
            status = main(self.cli("--channels", "1", "--reference", str(reference)))
        self.assertEqual(status, 0)
        download.assert_not_called()
        report = Table.read(
            self.root / "inventory_comparison.ecsv", format="ascii.ecsv"
        )
        self.assertEqual(report["agreement"][0], "match_without_qa_count")

    def test_cli_query_error_is_nonzero(self):
        self.client.query_sia.side_effect = ArchiveError("unavailable")
        with patch("SPCA.spitzer_download.ArchiveClient", return_value=self.client):
            self.assertEqual(main(self.cli()), 1)

    def test_cli_rejects_removed_service_option(self):
        with self.assertRaises(SystemExit) as caught:
            main(self.cli("--service", "dataservice"))
        self.assertEqual(caught.exception.code, 2)

    @unittest.skipUnless(HAS_REQUESTS, "Install .[download] for HTTP transfers")
    def test_explicit_cli_download_runs_directory_setup(self):
        self.client.session.get.return_value = response(image_bytes())
        with patch("SPCA.spitzer_download.ArchiveClient", return_value=self.client):
            status = main(self.cli("--channels", "1", "--bcd-only", "--download"))
        self.assertEqual(status, 0)
        self.assertTrue((self.root / "analysis/aorSnippet.txt").is_file())

    def test_manifest_cannot_silently_ignore_channel_selection(self):
        path = save_plan(self.plan())
        with self.assertRaises(SystemExit) as caught:
            main(["--manifest", str(path), "--channels", "4"])
        self.assertEqual(caught.exception.code, 2)

    def test_invalid_manifest_is_actionable_nonzero_error(self):
        path = self.root / "invalid.ecsv"
        Table({"unrelated": [1]}).write(path, format="ascii.ecsv")
        self.assertEqual(main(["--manifest", str(path)]), 1)

    def run_notebook(self, namespace, after_settings=None):
        notebook = json.loads((ROOT / "Download_Spitzer_Data.ipynb").read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            exec(  # noqa: S102
                compile(source, "Download_Spitzer_Data.ipynb", "exec"), namespace
            )
            if after_settings and "DOWNLOAD = False" in source:
                namespace.update(after_settings)

    def test_notebook_runs_with_mocks_and_download_disabled(self):
        namespace = {"display": lambda value: None}
        with (
            patch(
                "SPCA.spitzer_download.discover_observations",
                return_value=self.observations(),
            ),
            patch("SPCA.spitzer_download.download_plan") as download,
            patch("SPCA.spitzer_download.compare_inventory") as compare,
        ):
            self.run_notebook(namespace)
        download.assert_not_called()
        compare.assert_not_called()
        self.assertIsNone(namespace["plan"])
        self.assertIsNone(namespace["comparison"])
        self.assertFalse(namespace["DOWNLOAD"])
        self.assertIsNotNone(namespace["POSITION"])

    def test_notebook_selected_aor_previews_and_compares_shared_plan(self):
        namespace = {"display": lambda value: None}
        self.client.query_sia.return_value = sia_products(
            [
                (1, "bcd", "a_bcd.fits"),
                (1, "bcd", "a_bunc.fits"),
                (1, "bcd", "a_bimsk.fits"),
                (1, "cal", "a_sdark.fits"),
                (1, "raw", "a_dce.fits"),
            ]
        )
        reference = self.root / "extracted/r1234501/ch1/bcd"
        reference.mkdir(parents=True)
        for name in ("a_bcd.fits", "a_bunc.fits", "a_bimsk.fits"):
            (reference / name).write_bytes(b"")
        with (
            patch("SPCA.spitzer_download.ArchiveClient", return_value=self.client),
            patch("SPCA.spitzer_download.download_plan") as download,
        ):
            self.run_notebook(
                namespace,
                {
                    "OUTPUT": self.root,
                    "SELECTED_AORS": ["1234501"],
                    "SELECTED_CHANNELS": [1],
                    "SHA_REFERENCE": self.root / "extracted",
                },
            )
        download.assert_not_called()
        plan = namespace["plan"]
        self.assertEqual(
            list(plan["product"]), ["science", "mask", "uncertainty", "ancillary"]
        )
        self.assertIn("data/ch1/r1234501/ch1/cal/a_sdark.fits", plan["local_path"])
        self.assertEqual(plan.meta["excluded_products"], {"level0_raw": 1})
        self.assertIn("1234501", plan.meta["time_coverage"])
        self.assertTrue((self.root / "download_plan.ecsv").exists())
        self.assertEqual(
            namespace["comparison"]["agreement"][0], "match_without_qa_count"
        )


if __name__ == "__main__":
    unittest.main()
