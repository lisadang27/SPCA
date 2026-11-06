"""Synthetic IRAC BCD regressions; no archive/science data are required."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, Sip

from SPCA import Photometry_Aperture as aperture
from SPCA import frameDiagnosticsBackend as diagnostics
from SPCA.Photometry_Common import (
    full_frame_crop,
    get_time,
    prepare_image,
    prepare_images,
)

TARGET = SkyCoord(97.6, 29.7, unit="deg")
POSITIONS = ((20.3, 27.7), (125.5, 127.8), (130.8, 181.1))


def image_header(position=POSITIONS[0], time=54768.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [TARGET.ra.deg, TARGET.dec.deg]
    w.wcs.crpix = np.array(position) + 1  # FITS reference pixels are one-based.
    angle = 0.31
    w.wcs.cd = np.array(
        [
            [-np.cos(angle), np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    ) * (1.22 / 3600)
    h = w.to_header()
    h.update(GAIN=2.0, EXPTIME=3.0, FLUXCONV=1.5, FRAMTIME=4.0, BMJD_OBS=time)
    return h


def full_image(position=POSITIONS[0], drift=(0, 0)):
    y, x = np.mgrid[:256, :256]
    cx, cy = np.array(position) + drift
    return 10.0 + (-1.0) ** (x + y) + 100 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 2)


class FullFrameTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = tmp.name + "/"
        (self.root / "planet/analysis/ch2").mkdir(parents=True)

    def write(self, aor, index=0, position=POSITIONS[0], data=None, time=54768.0):
        folder = self.root / f"planet/data/ch2/r{aor}/ch2/bcd"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{index:04d}_bcd.fits"
        fits.PrimaryHDU(
            full_image(position) if data is None else data,
            image_header(position, time),
        ).writeto(path)
        return str(path)

    def load(self, **kwargs):
        return prepare_images(self.base, "planet", "ch2", "r", ncpu=1, **kwargs)

    def test_verified_positions_wcs_origin_axes_units_and_conversion(self):
        for n, position in enumerate(POSITIONS):
            path = self.write(n, position=position)
            header = fits.getheader(path)
            crop = full_frame_crop(header, TARGET)
            x, y = position
            self.assertEqual(
                crop, (slice(int(y) - 15, int(y) + 17), slice(int(x) - 15, int(x) + 17))
            )
            result = prepare_image(
                self.base, "r", [path], [1], target_coord=(97.6, 29.7)
            )
            self.assertEqual(result.shape, (1, 1025))
            np.testing.assert_allclose(
                result[0, :-1].reshape(32, 32), full_image(position)[crop] * 4
            )
            self.assertAlmostEqual(result[0, -1], 54768.0 + 2 / 86400, places=11)
            cx, cy, _, _ = aperture.centroid_FWM(result[:, :-1].reshape(1, 32, 32) - 40)
            # The legacy centroid uses a finite 5x5 window, so it has a small
            # truncation bias for fractional positions; WCS itself is checked
            # independently above and in the distorted-WCS round trip.
            self.assertAlmostEqual(cx[0], x - crop[1].start, delta=0.15)
            self.assertAlmostEqual(cy[0], y - crop[0].start, delta=0.15)

    def test_distorted_wcs_accuracy_away_from_reference_pixel(self):
        w = WCS(image_header(POSITIONS[1]))
        w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        a, b = np.zeros((3, 3)), np.zeros((3, 3))
        a[2, 0], b[0, 2] = 2e-5, -3e-5
        w.sip = Sip(a, b, None, None, w.wcs.crpix)
        header = fits.PrimaryHDU(np.zeros((256, 256)), w.to_header(relax=True)).header
        for x, y in POSITIONS:
            coord = w.pixel_to_world(x, y)
            crop = full_frame_crop(header, coord)
            recovered_x, recovered_y = w.world_to_pixel(coord)
            np.testing.assert_allclose([recovered_x, recovered_y], [x, y], atol=1e-5)
            self.assertTrue(15 <= x - crop[1].start < 16)
            self.assertTrue(15 <= y - crop[0].start < 16)

    def test_all_image_boundaries_are_validated_without_shifting(self):
        for position in ((15.25, 15.25), (239.75, 239.75)):
            path = self.write(str(position), position=position)
            crop = full_frame_crop(fits.getheader(path), TARGET)
            self.assertEqual(fits.getdata(path)[crop].shape, (32, 32))
        for n, position in enumerate(
            ((14.9, 128), (241.1, 128), (128, 14.9), (128, 241.1), (-2, 80), (260, 80))
        ):
            path = self.write(f"bad{n}", position=position)
            with self.assertRaisesRegex(ValueError, "outside image"):
                full_frame_crop(fits.getheader(path), TARGET)

    def test_missing_or_invalid_target_and_missing_wcs_fail_clearly(self):
        path = self.write(1)
        with self.assertRaisesRegex(ValueError, "require target_coord"):
            self.load()
        with self.assertRaisesRegex(ValueError, "one target"):
            full_frame_crop(
                fits.getheader(path), SkyCoord([97.6, 98], [29.7, 30], unit="deg")
            )
        with self.assertRaisesRegex(ValueError, "finite pixel"):
            full_frame_crop(fits.getheader(path), SkyCoord(np.nan, 29.7, unit="deg"))
        header = fits.PrimaryHDU(np.zeros((256, 256))).header
        with self.assertRaisesRegex(ValueError, "celestial WCS"):
            full_frame_crop(header, TARGET)

    def test_fixed_crop_per_aor_retains_drift_and_uses_each_aors_wcs(self):
        paths, expected, times = [], [], []
        for n, position in enumerate(POSITIONS):
            for i in range(2):
                # Change the later exposure's WCS by several pixels. The
                # crop must still use the first exposure's WCS for this AOR.
                data = full_image(position, drift=(0.2 * i, -0.15 * i))
                changed = tuple(np.array(position) + 4 * i)
                time = 54768.0 + (10 - 2 * n - i) / 86400
                path = self.write(n, i, changed, data, time)
                crop = full_frame_crop(fits.getheader(paths[-1] if i else path), TARGET)
                paths.append(path)
                expected.append(data[crop] * 4)
                times.append(time + 2 / 86400)
        # Standalone loading is also fixed per AOR.
        for i in range(6):
            result = prepare_image(
                self.base, "r", paths, [2, 2, 2], i=i, target_coord=TARGET
            )
            np.testing.assert_allclose(result[0, :-1].reshape(32, 32), expected[i])
        with patch(
            "SPCA.Photometry_Common.full_frame_crop", wraps=full_frame_crop
        ) as choose:
            images, bg, bg_err, time = self.load(target_coord=TARGET)
        self.assertEqual(choose.call_count, 3)
        order = np.argsort(times)
        np.testing.assert_allclose(time, np.sort(times), rtol=0, atol=1e-11)
        np.testing.assert_allclose(
            images + bg[:, None, None], np.array(expected)[order]
        )
        self.assertTrue(np.isfinite(bg_err).all())
        breaks = np.fromstring(
            (self.root / "planet/analysis/ch2/aorBreaks.txt").read_text(), sep=" "
        )
        np.testing.assert_allclose(breaks, sorted(times[::2])[1:], rtol=0, atol=1e-8)

    def test_bad_pixels_and_full_frame_star_masks_use_crop_offsets(self):
        position = POSITIONS[1]
        data = full_image(position)
        data[128, 126], data[120, 118] = np.nan, np.inf
        path = self.write(1, position=position, data=data)
        header = fits.getheader(path)
        crop = full_frame_crop(header, TARGET)
        w = WCS(header)
        # Non-symmetric coordinates catch accidental x/y transposition.
        coords = [
            w.pixel_to_world(115, 120),
            w.pixel_to_world(crop[1].start, crop[0].start),
            w.pixel_to_world(250, 250),
        ]
        stars = [(c.ra, c.dec) for c in coords]
        result = prepare_image(
            self.base, "r", [path], [1], target_coord=TARGET, maskStars=stars
        )
        mask = result[0, :-1].reshape(32, 32).mask
        expected = ~np.isfinite(data[crop])
        y, x = 120 - crop[0].start, 115 - crop[1].start
        expected[y - 1 : y + 2, x - 1 : x + 2] = True
        expected[:2, :2] = True
        np.testing.assert_array_equal(mask, expected)
        flux = aperture.A_photometry(
            [4], img_data=result[:, :-1].reshape(1, 32, 32), cx=15.5, cy=15.8
        )
        self.assertTrue(np.isnan(flux).all())

    def test_full_frame_correction_stack_is_cropped_identically(self):
        path = self.write(1)
        correction = np.arange(256 * 256, dtype=float).reshape(256, 256)
        crop = full_frame_crop(fits.getheader(path), TARGET)
        stack = self.root / "stack.fits"
        for value in (
            correction,
            correction[None],
            correction[crop],
            correction[crop][None],
        ):
            fits.PrimaryHDU(value).writeto(stack, overwrite=True)
            result = prepare_image(
                self.base,
                "r",
                [path],
                [1],
                stacks=[str(stack)],
                addStack=True,
                target_coord=TARGET,
            )
            np.testing.assert_allclose(
                result[0, :-1].reshape(32, 32),
                (full_image()[crop] + correction[crop]) * 4,
            )
        fits.PrimaryHDU(np.ones((64, 32, 32))).writeto(stack, overwrite=True)
        with self.assertRaisesRegex(ValueError, "correction stack"):
            prepare_image(
                self.base,
                "r",
                [path],
                [1],
                stacks=[str(stack)],
                addStack=True,
                target_coord=TARGET,
            )

    def test_full_frame_diagnostics_skip_even_64_files_and_exclusions_reject(self):
        path = self.write(1)
        # Simulate a multiple-of-64 file count without duplicating fixtures.
        with (
            patch.object(diagnostics, "get_fnames", return_value=([path] * 64, [64])),
            patch.object(diagnostics, "get_lightcurve") as extract,
        ):
            self.assertEqual(
                diagnostics.run_diagnostics("planet", "ch2", "r", self.base, False), []
            )
        extract.assert_not_called()
        with self.assertRaisesRegex(ValueError, "cube positions"):
            self.load(target_coord=TARGET, ignoreFrames=[0])
        with self.assertRaisesRegex(ValueError, "cube positions"):
            prepare_image(
                self.base, "r", [path], [1], target_coord=TARGET, ignoreFrames=[1]
            )

    def test_subarray_pixels_timestamps_masks_and_diagnostics_remain_unchanged(self):
        y, x = np.mgrid[:32, :32]
        data = np.repeat((10 + (-1.0) ** (x + y))[None], 64, axis=0)
        data[:, 15, 15] += 100
        data[8, 0, 0] = np.nan
        path = self.write(1, data=data)
        # Subarrays must continue to work without any celestial WCS.
        header = image_header()
        calibration = fits.Header(
            {
                key: header[key]
                for key in ("GAIN", "EXPTIME", "FLUXCONV", "FRAMTIME", "BMJD_OBS")
            }
        )
        fits.PrimaryHDU(data, calibration).writeto(path, overwrite=True)
        header = fits.getheader(path)
        expected_time = np.linspace(54768 + 2 / 86400, 54768 + 63 * 4 / 86400, 64)
        np.testing.assert_array_equal(get_time(header, []), expected_time)
        before = prepare_image(self.base, "r", [path], [1], ignoreFrames=[1])
        after = prepare_image(
            self.base, "r", [path], [1], ignoreFrames=[1], target_coord=TARGET
        )
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(before.mask, after.mask)
        self.assertTrue(after.mask[1, :-1].all())
        self.assertTrue(after.mask[8, 0])
        np.testing.assert_array_equal(after[:, -1], expected_time)
        images, bg, bg_err, time = self.load(ignoreFrames=[1], target_coord=TARGET)
        self.assertEqual(images.shape, (64, 32, 32))
        np.testing.assert_allclose(bg[0], 40)
        np.testing.assert_allclose(bg_err[0], 4)
        np.testing.assert_array_equal(time, expected_time)
        # A valid cube still enters the original diagnostics extraction path.
        with patch.object(
            diagnostics, "get_lightcurve", side_effect=RuntimeError("entered")
        ) as extract:
            with self.assertRaisesRegex(RuntimeError, "entered"):
                diagnostics.run_diagnostics("planet", "ch2", "r", self.base, False)
        extract.assert_called_once()

    def test_end_to_end_baseline_radii_centroids_and_output_products(self):
        for i in range(11):
            self.write(
                1,
                i,
                position=POSITIONS[2],
                data=full_image(POSITIONS[2], drift=(i * 0.01, i * -0.01)),
                time=54768 + i * 4 / 86400,
            )
        options = dict(
            target_coord=TARGET,
            r=[2, 2.5],
            edges=["Exact", "Hard"],
            moveCentroids=[True, False],
            ncpu=1,
            highpassWidth=9,
            showPlots=False,
            savePlots=False,
        )
        full, binned = aperture.get_lightcurve(
            self.base,
            "r",
            "ch2",
            "planet",
            save=False,
            onlyBest=False,
            bin_size=3,
            **options,
        )
        self.assertEqual(len(full), 8)
        self.assertEqual(len(binned), 8)
        for data, bins in zip(full, binned):
            self.assertEqual(data.shape, (11, 8))
            self.assertEqual(bins.shape, (4, 16))
            self.assertTrue(np.isfinite(data).all())
            self.assertTrue(np.isfinite(bins).all())
            np.testing.assert_allclose(
                data[:, 1],
                54768 + np.arange(11) * 4 / 86400 + 2 / 86400,
                rtol=0,
                atol=1e-11,
            )
        aperture.get_lightcurve(
            self.base,
            "r",
            "ch2",
            "planet",
            save=True,
            onlyBest=False,
            bin_size=3,
            **options,
        )
        outputs = list(
            (self.root / "planet/analysis/ch2/addedBlank/noIgnore").glob(
                "*/*datacube*.dat"
            )
        )
        self.assertEqual(len(outputs), 16)
        for path in outputs:
            saved = np.loadtxt(path)
            self.assertEqual(saved.shape, (4, 16) if "binned" in path.name else (11, 8))
        # Optimization still selects one of the same candidates, and baseline
        # extraction also works without binning or frame exclusions.
        best = aperture.get_lightcurve(
            self.base,
            "r",
            "ch2",
            "planet",
            save=False,
            onlyBest=True,
            bin_data=False,
            ignoreFrames=[],
            **options,
        )
        self.assertEqual(len(best), 1)
        self.assertTrue(any(np.allclose(best[0], candidate) for candidate in full))


if __name__ == "__main__":
    unittest.main()
