"""Exercise real Photutils calls and rejected-frame propagation through SPCA."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes
import numpy as np
from astropy.io import fits
from photutils.aperture import (
    CircularAperture,
    EllipticalAperture,
    RectangularAperture,
    aperture_photometry,
)

from SPCA import Photometry_Aperture as aperture
from SPCA.Photometry_Common import (
    bin_array,
    clip_data,
    prepare_image,
    prepare_images,
    sigma_clipping,
)


def star_frames(n=24):
    y, x = np.mgrid[:32, :32]
    star = 100 * np.exp(-((x - 15) ** 2 + (y - 15) ** 2) / 2)
    return np.ma.array(np.repeat(star[None], n, axis=0), mask=False)


class ApertureMaskTests(unittest.TestCase):
    def test_valid_flux_order_and_uncertainty_inputs(self):
        frames = star_frames(1)
        frames[0, 15, 15] = -2  # Negative science pixels retain their flux.
        methods = ["center", "exact", "subpixel"]
        configurations = (
            ("Circular", {"r": [1.5, 2.5]}, CircularAperture, [{"r": 1.5}, {"r": 2.5}]),
            (
                "Elliptical",
                {"a": [2, 3], "b": [1, 2], "theta": [0.2, 0.3]},
                EllipticalAperture,
                [{"a": 2, "b": 1, "theta": 0.2}, {"a": 3, "b": 2, "theta": 0.3}],
            ),
            (
                "Rectangular",
                {"w_r": [2, 3], "h_r": [1, 2], "theta": [0.2, 0.3]},
                RectangularAperture,
                [{"w": 2, "h": 1, "theta": 0.2}, {"w": 3, "h": 2, "theta": 0.3}],
            ),
        )
        data = frames.data[0]
        expected_error = np.sqrt(3**2 + np.maximum(data, 0))
        for shape, kwargs, cls, sizes in configurations:
            with self.subTest(shape=shape):
                expected = []
                actual_tables = []

                def record_photometry(
                    data, apertures, actual_tables=actual_tables, **options
                ):
                    # Observe the real call; the failing Photutils code runs.
                    self.assertIs(type(data), np.ndarray)
                    self.assertIs(type(options["error"]), np.ndarray)
                    self.assertIs(type(options["mask"]), np.ndarray)
                    self.assertEqual(options["mask"].dtype, bool)
                    np.testing.assert_allclose(options["error"], expected_error)
                    table = aperture_photometry(data, apertures, **options)
                    actual_tables.append(table)
                    return table

                for position in ((15, 15), (15.25, 14.75)):
                    for method in methods:
                        aps = [cls(position, **size) for size in sizes]
                        table = aperture_photometry(
                            data, aps, error=expected_error, method=method
                        )
                        expected.extend(table[f"aperture_sum_{j}"][0] for j in range(2))
                with patch.object(aperture, "aperture_photometry", record_photometry):
                    result = aperture.A_photometry(
                        [3],
                        img_data=frames,
                        shape=shape,
                        methods=methods,
                        moveCentroids=[True, False],
                        cx_med=15.25,
                        cy_med=14.75,
                        **kwargs,
                    )
                np.testing.assert_allclose(result, expected)
                # For center weights, variances add with unit gain in electrons.
                weights = (
                    cls((15, 15), **sizes[0])
                    .to_mask(method="center")
                    .to_image(data.shape)
                )
                np.testing.assert_allclose(
                    actual_tables[0]["aperture_sum_err_0"][0],
                    np.sqrt(np.sum(expected_error**2 * weights)),
                )

    def test_masked_and_nonfinite_pixels_are_not_partial_flux(self):
        for bad in (np.ma.masked, np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                frames = star_frames(4)
                frames[1, 0, 0] = bad  # Outside all apertures.
                frames[2, 15, 15] = bad
                frames[3] = bad
                result = np.array(
                    [
                        aperture.A_photometry(
                            np.ones(4), img_data=frames, i=i, r=[1.5, 2.5]
                        )
                        for i in range(4)
                    ]
                )
                self.assertEqual(result.shape, (4, 4))
                np.testing.assert_allclose(result[0], result[1])
                self.assertTrue(np.isnan(result[2:]).all())

    def test_fractional_edges_and_each_aperture_use_their_actual_weights(self):
        frames = star_frames(1)
        frames[0, 15, 17] = np.ma.masked
        result = aperture.A_photometry(
            [1], img_data=frames, r=[1, 2], methods=["center", "exact", "subpixel"]
        ).reshape(3, 2)
        self.assertTrue(np.isfinite(result[:, 0]).all())
        self.assertTrue(np.isfinite(result[0, 1]))  # Center is on outer boundary.
        self.assertTrue(np.isnan(result[1:, 1]).all())

    def test_invalid_background_or_centroid_keeps_result_shape(self):
        frames = star_frames(2)
        for bg in ([np.nan, 1], [np.inf, 1], [-1, 1], np.ma.array([1, 1], mask=[1, 0])):
            result = aperture.A_photometry(bg, img_data=frames, r=[1, 2])
            self.assertEqual(result.shape, (4,))
            self.assertTrue(np.isnan(result).all())
        for cx in ([np.nan, 15], np.ma.array([15, 15], mask=[1, 0])):
            result = aperture.A_photometry(
                [1, 1], img_data=frames, cx=cx, moveCentroids=[True, False]
            )
            self.assertTrue(np.isnan(result[:2]).all())
            self.assertTrue(np.isfinite(result[2:]).all())

    def test_global_stack_and_explicit_stack_agree(self):
        frames = star_frames(2)
        frames[1] = np.ma.masked
        with patch.object(aperture, "image_stack", frames):
            for i in range(2):
                np.testing.assert_allclose(
                    aperture.A_photometry([1, 1], i=i),
                    aperture.A_photometry([1, 1], i=i, img_data=frames.copy()),
                    equal_nan=True,
                )
            self.assertIs(aperture.image_stack, frames)

    def test_empty_or_truncated_aperture_is_invalid(self):
        frames = star_frames(1)
        frames.mask[0, 10:21, 10:21] = True  # Frame itself still has valid pixels.
        self.assertTrue(np.isnan(aperture.A_photometry([1], img_data=frames)).all())
        for cx in (-1, 100):
            self.assertTrue(
                np.isnan(
                    aperture.A_photometry([1], cx=cx, img_data=star_frames(1))
                ).all()
            )


class RejectedFrameTests(unittest.TestCase):
    def test_sigma_clipping_rejects_target_frame_but_only_bad_background_pixel(self):
        frames = np.ones((64, 32, 32))
        frames[8, 15, 15] = 1000
        frames[9, 0, 0] = 1000
        clipped = sigma_clipping(frames, sigma=4)
        self.assertTrue(clipped.mask[8].all())
        self.assertEqual(clipped.mask[9].sum(), 1)
        self.assertTrue(
            np.isnan(aperture.A_photometry(np.ones(64), img_data=clipped, i=8)).all()
        )
        self.assertTrue(
            np.isfinite(aperture.A_photometry(np.ones(64), img_data=clipped, i=9)).all()
        )

    def test_clipping_preserves_missing_and_newly_rejected_flux(self):
        values = np.ma.array(np.ones(128), mask=False)
        values[5] = 1e9
        values[12] = np.nan
        values[20] = np.inf
        values.data[30] = 999
        values.mask[30] = True
        clipped = clip_data(values, 5, sigma1=3, sigma2=3, replace_masked=False)
        self.assertEqual(set(np.flatnonzero(clipped.mask)), {5, 12, 20, 30})
        np.testing.assert_array_equal(clipped.compressed(), 1)
        self.assertTrue(
            clip_data(np.ma.masked_all(8), 3, replace_masked=False).mask.all()
        )

    def test_bins_ignore_hidden_data_and_keep_empty_bins_invalid(self):
        values = np.ma.array(
            [2, 999, 888, 777, np.nan, 4, 6], mask=[0, 1, 1, 1, 0, 0, 0]
        )
        median, std = bin_array(values, 2)
        np.testing.assert_allclose(median, [2, np.nan, 4, 6], equal_nan=True)
        np.testing.assert_allclose(std, [0, np.nan, 0, 0], equal_nan=True)

    def test_bin_all_data_retains_masked_metadata_and_new_flux_rejections(self):
        flux = np.ones(128)
        flux[5] = 1e9
        metadata = np.ma.array(np.ones(128), mask=False)
        metadata.data[12] = 999
        metadata.mask[12] = True
        result = aperture.bin_all_data(3, flux, *([metadata] * 14), bin_size=1)
        self.assertTrue(np.isnan(result[5, :2]).all())
        self.assertTrue(np.isnan(result[12, 2:]).all())
        self.assertEqual(result[12, 0], 1)

    def test_fits_preparation_preserves_masks_and_subtracts_background_in_parent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bcd = root / "planet/data/ch2/r123/ch2/bcd"
            bcd.mkdir(parents=True)
            (root / "planet/analysis/ch2").mkdir(parents=True)
            header = fits.Header(
                {
                    "GAIN": 1.0,
                    "EXPTIME": 1.0,
                    "FLUXCONV": 1.0,
                    "FRAMTIME": 1.0,
                    "BMJD_OBS": 54768.0,
                }
            )
            y, x = np.mgrid[:32, :32]
            data = np.repeat((9.0 + 2 * ((x + y) % 2))[None], 64, axis=0)
            data[:, 15, 15] += 100
            data[8, 15, 15] += 10000
            data[9, 0, 0] = np.nan
            # No usable background, but valid target pixels: must also reject.
            data[10] = np.nan
            data[10, 11:19, 11:19] = 10
            file = bcd / "a_bcd.fits"
            fits.PrimaryHDU(data, header).writeto(file)
            prepared = prepare_image(tmp, "r", [str(file)], [1], ignoreFrames=[1])
            self.assertTrue(prepared.mask[1, :-1].all())
            self.assertTrue(prepared.mask[8, :-1].all())
            self.assertTrue(prepared.mask[9, 0])
            images, bg, bg_err, time = prepare_images(
                tmp + "/", "planet", "ch2", "r", ignoreFrames=[1], ncpu=1
            )
            self.assertEqual(images.shape, (64, 32, 32))
            self.assertTrue(images.mask[[1, 8, 10]].all())
            self.assertTrue(np.isnan(bg[[1, 8, 10]]).all())
            self.assertTrue(np.isnan(bg_err[[1, 8, 10]]).all())
            np.testing.assert_allclose([bg[0], bg_err[0]], [10, 1])
            np.testing.assert_allclose(images[0, 0, :2], [-1, 1])
            self.assertEqual(images[0, 15, 15], 99)
            self.assertEqual(time.shape, (64,))
            # Check the entire parent frame, not just its background estimate.
            # Zero or two subtractions would give different pixels and fluxes.
            np.testing.assert_allclose(images[0], data[0] - 10)
            inside = (x - 15) ** 2 + (y - 15) ** 2 < 2.4**2
            expected_flux = np.sum((data[0] - 10)[inside])
            self.assertEqual(expected_flux, 103)
            np.testing.assert_allclose(
                aperture.A_photometry(
                    bg_err, img_data=images, r=[2.4], methods=["center"]
                ),
                [expected_flux],
            )

            before = images.copy()
            full = self.run_lightcurve(
                tmp,
                images,
                bg=bg,
                bg_err=bg_err,
                time=time,
                bin_data=False,
                r=[2.4],
                edges=["Hard"],
                moveCentroids=[False],
            )[0]
            np.testing.assert_allclose(full[0, [0, 2, 3, 6]], [103, 15, 15, 10])
            np.testing.assert_array_equal(images.data, before.data)
            np.testing.assert_array_equal(images.mask, before.mask)

    def test_native_star_coordinates_and_flux_with_default_oversamp(self):
        frames = star_frames()
        n = len(frames)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bcd = root / "planet/data/ch2/r123/ch2/bcd"
            bcd.mkdir(parents=True)
            (root / "planet/analysis/ch2").mkdir(parents=True)
            header = fits.Header(
                {
                    "GAIN": 1.0,
                    "EXPTIME": 1.0,
                    "FLUXCONV": 1.0,
                    "FRAMTIME": 1.0,
                    "BMJD_OBS": 54768.0,
                }
            )
            fits.PrimaryHDU(frames.data + 10, header).writeto(bcd / "a_bcd.fits")
            for supplied in (False, True):
                for scale_options in ({}, {"scale": 2}, {"scale": 3}):
                    with self.subTest(supplied=supplied, **scale_options):
                        options = {
                            "save": False,
                            "savePlots": False,
                            "bin_data": False,
                            "onlyBest": False,
                            "highpassWidth": 6,
                            "ncpu": 1,
                        }
                        if supplied:
                            options.update(
                                image_stack_input=frames,
                                bg=np.full(n, 10.0),
                                bg_err=np.ones(n),
                                time=np.arange(n, dtype=float),
                            )
                        # Omit oversamp, r, and edges to exercise normal defaults;
                        # the first case also omits scale (whose default is 2).
                        full = aperture.get_lightcurve(
                            tmp, "r123", "ch2", "planet", **options, **scale_options
                        )[0]
                        np.testing.assert_allclose(full[:, 2:4], 15, atol=1e-12)
                        # Reference sum of this sampled Gaussian with an exact
                        # r=2.4 aperture centered at its known (15, 15) position.
                        np.testing.assert_allclose(full[:, 0], 581.31272831, rtol=1e-10)

    def test_preloaded_oversampled_star_keeps_explicit_scale(self):
        y, x = np.mgrid[:64, :64]
        star = 25 * np.exp(-((x - 30.5) ** 2 + (y - 30.5) ** 2) / 8)
        frames = np.repeat(star[None], 24, axis=0)
        expected_flux = np.sum(
            star
            * CircularAperture((30.5, 30.5), 4.8)
            .to_mask(method="exact")
            .to_image(star.shape)
        )
        with TemporaryDirectory() as tmp:
            full = self.run_lightcurve(
                tmp, frames, oversamp=True, scale=2, bin_data=False, r=[2.4]
            )[0]
        np.testing.assert_allclose(full[:, 2:4], 15.25, atol=1e-12)
        np.testing.assert_allclose(full[:, 0], expected_flux, rtol=1e-12)
        np.testing.assert_array_equal(frames, np.repeat(star[None], 24, axis=0))

    def run_lightcurve(self, tmp, frames, **kwargs):
        n = len(frames)
        options = {
            "save": False,
            "savePlots": False,
            "onlyBest": False,
            "image_stack_input": frames,
            "bg": np.ones(n),
            "bg_err": np.ones(n),
            "time": np.arange(n, dtype=float),
            "bin_size": 2,
            "highpassWidth": 6,
            "ncpu": 1,
            "r": [1.5, 2.5],
        }
        options.update(kwargs)
        return aperture.get_lightcurve(tmp, "r123", "ch2", "planet", **options)

    def test_real_pipeline_returned_saved_and_plotted_data_retain_rejections(self):
        frames = star_frames()
        frames[2:4] = np.ma.masked  # An empty bin.
        frames[6, 15, 15] = np.nan
        frames[9, 0, 0] = np.inf  # Outside aperture; measurement remains valid.
        with TemporaryDirectory() as tmp:
            full, binned = self.run_lightcurve(tmp, frames)
            for data, bins in zip(full, binned):
                self.assertIs(type(data), np.ndarray)
                self.assertTrue(np.isnan(data[[2, 3, 6], 0]).all())
                self.assertTrue(np.isfinite(data[9, 0]))
                self.assertTrue(np.isnan(bins[1]).all())
                self.assertEqual(bins[3, 2], 7)  # Surviving frame's time.
                self.assertTrue(np.isfinite(bins[3, 0]))

            plotted = []
            original_plot = matplotlib.axes.Axes.plot

            def record_plot(ax, x, y, *args, **kwargs):
                if args and args[0] == "k+":
                    plotted.append((np.asarray(x).copy(), np.asarray(y).copy()))
                return original_plot(ax, x, y, *args, **kwargs)

            (Path(tmp) / "planet/analysis/ch2").mkdir(parents=True)
            with patch.object(matplotlib.axes.Axes, "plot", record_plot):
                self.run_lightcurve(tmp, frames, save=True, savePlots=True)
            for i, (data, bins) in enumerate(zip(full, binned)):
                folder = next(Path(tmp).rglob(f"4umExactCircular{['1_5', '2_5'][i]}*"))
                np.testing.assert_allclose(
                    np.loadtxt(next(folder.glob("*full*.dat"))), data, equal_nan=True
                )
                np.testing.assert_allclose(
                    np.loadtxt(next(folder.glob("*binned*.dat"))), bins, equal_nan=True
                )
                np.testing.assert_allclose(
                    plotted[5 * i][1], bins[:, 0], equal_nan=True
                )
                self.assertTrue((folder / "Lightcurve.pdf").is_file())

    def test_entirely_invalid_dataset_and_rms_selection(self):
        with TemporaryDirectory() as tmp:
            frames = star_frames(8)
            frames[:] = np.ma.masked
            full, binned = self.run_lightcurve(tmp, frames, onlyBest=True)
            self.assertEqual(len(full), 2)
            for data, bins in zip(full, binned):
                self.assertTrue(np.isnan(data[:, 0]).all())
                self.assertTrue(np.isnan(bins[:, :2]).all())
            flux = np.column_stack((np.full(8, np.nan), np.ones(8)))
            rms = aperture.compare_RMS(
                ["invalid", "valid"],
                flux,
                np.array([1, 2]),
                np.arange(8),
                3,
                tmp,
                "planet",
                "ch2",
                [],
                False,
                save=False,
                savePlots=False,
            )
            self.assertEqual(rms[0], np.inf)
            self.assertEqual(rms[1], 0)

    def test_unbinned_outputs_and_low_flux_rejections(self):
        frames = star_frames()
        frames[2:4] = np.ma.masked
        frames[6] *= 0.01
        with TemporaryDirectory() as tmp:
            full = self.run_lightcurve(tmp, frames, bin_data=False)
            for data in full:
                self.assertTrue(np.isnan(data[[2, 3, 6], 0]).all())
                self.assertTrue(np.isfinite(data[0, 0]))

    def test_real_worker_exceptions_propagate_for_first_and_later_jobs(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.run_lightcurve(tmp, star_frames(4), r=[-1])
            with self.assertRaises(IndexError):
                self.run_lightcurve(tmp, star_frames(4), bg_err=np.ones(3))


if __name__ == "__main__":
    unittest.main()
