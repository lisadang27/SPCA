"""Explicit BCD/CBCD selection and minimal correction-stack requirements."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
from astropy.io import fits

from SPCA import (
    Photometry_Aperture,
    Photometry_Companion,
    Photometry_PLD,
    Photometry_PSF,
)
from SPCA.directory_structure import make_directory_structure
from SPCA.Photometry_Common import (
    get_fnames,
    get_stacks,
    oversample_cache_name,
    prepare_image,
)
from SPCA.Photometry_Companion import get_fnames as companion_fnames

HEADER = {
    "GAIN": 1.0,
    "EXPTIME": 1.0,
    "FLUXCONV": 1.0,
    "FRAMTIME": 1.0,
    "BMJD_OBS": 54768.0,
}


def _write_image(path, value):
    fits.PrimaryHDU(np.full((1, 32, 32), value), fits.Header(HEADER)).writeto(path)


def _upsample(data, scale):
    """Flux-conserving nearest-neighbour stand-in for ``oversampling``."""
    return np.repeat(np.repeat(data, scale, axis=1), scale, axis=2) / scale**2


class PhotometryProductTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data/ch1"
        self.bcd = self.data / "r123/ch1/bcd"
        self.bcd.mkdir(parents=True)

    def test_alternative_versions_never_mix(self):
        for name in ("a_bcd.fits", "a_cbcd.fits", "a_bunc.fits"):
            (self.bcd / name).touch()
        files, lengths = get_fnames(str(self.data), "r")
        self.assertEqual([Path(f).name for f in files], ["a_bcd.fits"])
        self.assertEqual(lengths, [1])
        files, lengths = get_fnames(str(self.data), "r", science_product="cbcd")
        self.assertEqual([Path(f).name for f in files], ["a_cbcd.fits"])
        self.assertEqual(lengths, [1])
        with self.assertRaises(ValueError):
            get_fnames(str(self.data), "r", science_product="both")
        for product in ("bcd", "cbcd"):
            files, count = companion_fnames(
                str(self.data), "r", "ch1", science_product=product
            )
            self.assertEqual([Path(f).name for f in files], [f"a_{product}.fits"])
            self.assertEqual(count, 1)

    def test_cbcd_only_directory_is_recognized(self):
        (self.bcd / "a_cbcd.fits").touch()
        self.assertEqual(make_directory_structure(self.root), "r123")

    def test_cbcd_cannot_reuse_bcd_oversampling_cache(self):
        filename = "SPITZER_I1_123_0002_6_C7915448_cbcd.fits"
        image = self.bcd / filename
        header = fits.Header(
            {
                "GAIN": 1.0,
                "EXPTIME": 1.0,
                "FLUXCONV": 1.0,
                "FRAMTIME": 1.0,
                "BMJD_OBS": 54768.0,
            }
        )
        fits.PrimaryHDU(np.full((1, 32, 32), 2.0), header).writeto(image)
        cache = self.root / "Oversampled"
        cache.mkdir()
        stale = [
            cache / "0002.pkl",
            cache / "0002_cbcd.pkl",
            cache / "SPITZER_I1_123_0002_6_C7915448_bcd.pkl",
            cache / "SPITZER_I1_123_0002_6_C7915448_cbcd.pkl",
        ]
        for old in stale:
            old.write_bytes(b"other cache")
        with (
            patch(
                "SPCA.Photometry_Common.np.load",
                side_effect=AssertionError("other cache used"),
            ),
            patch(
                "SPCA.Photometry_Common.oversampling", side_effect=_upsample
            ) as sample,
        ):
            result = prepare_image(
                str(self.root) + "/",
                "r",
                [str(image)],
                [1],
                oversamp=True,
                i=0,
            )
        sample.assert_called_once()
        np.testing.assert_allclose(result[:, :-1], 0.5)
        self.assertEqual(
            len(list(cache.glob("SPITZER_I1_123_0002_6_C7915448_cbcd_x2_*.pkl"))), 1
        )
        for old in stale:
            self.assertEqual(old.read_bytes(), b"other cache")

    def test_oversampling_cache_is_unique_per_aor_exposure(self):
        images = []
        # Both AORs contain exposure 0002; their pixel values differ.
        for aor, value in (("28566016", 1.0), ("28578560", 5.0)):
            directory = self.data / f"r{aor}/ch1/bcd"
            directory.mkdir(parents=True)
            image = directory / f"SPITZER_I1_{aor}_0002_6_C7915448_bcd.fits"
            _write_image(image, value)
            images.append(str(image))
        savepath = str(self.root) + "/"
        cache = self.root / "Oversampled"
        cache.mkdir()
        (cache / "0002.pkl").write_bytes(b"ambiguous old cache")

        def run():
            return [
                prepare_image(savepath, "r", images, [1, 1], oversamp=True, i=i)
                for i in range(2)
            ]

        upsample = patch("SPCA.Photometry_Common.oversampling", side_effect=_upsample)
        with upsample as sample:
            first = run()
        self.assertEqual(sample.call_count, 2)
        names = sorted(p.name for p in cache.iterdir())
        self.assertEqual(len(names), 3)
        self.assertEqual(names[0], "0002.pkl")
        self.assertTrue(
            names[1].startswith("SPITZER_I1_28566016_0002_6_C7915448_bcd_x2_")
        )
        self.assertTrue(
            names[2].startswith("SPITZER_I1_28578560_0002_6_C7915448_bcd_x2_")
        )
        # Reuse loads each AOR's own cache rather than recomputing.
        with upsample as sample:
            second = run()
        sample.assert_not_called()
        for before, after, value in zip(first, second, (1.0, 5.0)):
            np.testing.assert_allclose(after, before)
            np.testing.assert_allclose(after[:, :-1], value / 4)

    def test_oversampling_cache_tracks_scale_stack_and_image_contents(self):
        image = self.bcd / "SPITZER_I1_123_0002_6_C7915448_bcd.fits"
        _write_image(image, 2.0)
        stack = self.root / "correction_stack.fits"
        fits.PrimaryHDU(np.ones((1, 32, 32))).writeto(stack)
        savepath = str(self.root) + "/"
        (self.root / "Oversampled").mkdir()

        def run(**settings):
            with patch(
                "SPCA.Photometry_Common.oversampling", side_effect=_upsample
            ) as sample:
                result = prepare_image(
                    savepath,
                    "r",
                    [str(image)],
                    [1],
                    [str(stack)],
                    oversamp=True,
                    **settings,
                )
            return result, sample.call_count

        # Identical image and settings reuse the cache.
        first, calls = run(scale=2)
        self.assertEqual(calls, 1)
        again, calls = run(scale=2)
        self.assertEqual(calls, 0)
        np.testing.assert_allclose(again, first)

        # A different scale regenerates rather than reusing the x2 image.
        result, calls = run(scale=3)
        self.assertEqual(calls, 1)
        self.assertEqual(result.shape, (1, 96 * 96 + 1))

        # Enabling the correction stack cannot reuse the uncorrected cache.
        result, calls = run(scale=2, addStack=True)
        self.assertEqual(calls, 1)
        np.testing.assert_allclose(result[:, :-1], 3.0 / 4)
        _, calls = run(scale=2, addStack=True)
        self.assertEqual(calls, 0)

        # Replacing the stack contents invalidates the corrected cache.
        stack.unlink()
        fits.PrimaryHDU(np.full((1, 32, 32), 5.0)).writeto(stack)
        result, calls = run(scale=2, addStack=True)
        self.assertEqual(calls, 1)
        np.testing.assert_allclose(result[:, :-1], 7.0 / 4)

        # So does replacing the science image itself (e.g. a re-download).
        image.unlink()
        _write_image(image, 4.0)
        result, calls = run(scale=2)
        self.assertEqual(calls, 1)
        np.testing.assert_allclose(result[:, :-1], 1.0)

        # The blank-stack x2 cache is still valid and still reused.
        image.unlink()
        _write_image(image, 2.0)
        result, calls = run(scale=2)
        self.assertEqual(calls, 0)
        np.testing.assert_allclose(result, first)

    def test_cache_key_changes_when_only_the_mask_changes(self):
        # maskStars and ignoreFrames act through the mask, which is hashed.
        name = "SPITZER_I1_123_0002_6_C7915448_bcd.fits"
        image = np.ma.masked_array(np.ones((1, 32, 32)))
        masked = image.copy()
        masked[0, 15, 15] = np.ma.masked
        self.assertEqual(
            oversample_cache_name(name, image, 2),
            oversample_cache_name(name, image.copy(), 2),
        )
        self.assertNotEqual(
            oversample_cache_name(name, image, 2),
            oversample_cache_name(name, masked, 2),
        )

    def test_cache_directory_is_created_when_saving(self):
        image = self.bcd / "SPITZER_I1_123_0002_6_C7915448_bcd.fits"
        _write_image(image, 2.0)
        cache = self.root / "analysis/ch1/addedBlank/Oversampled"
        self.assertFalse(cache.parent.exists())
        savepath = str(cache.parent) + "/"

        def run():
            with patch(
                "SPCA.Photometry_Common.oversampling", side_effect=_upsample
            ) as sample:
                result = prepare_image(savepath, "r", [str(image)], [1], oversamp=True)
            return result, sample.call_count

        first, calls = run()
        self.assertEqual(calls, 1)
        (saved,) = cache.iterdir()
        self.assertTrue(saved.name.startswith("SPITZER_I1_123_0002_6_C7915448_bcd_x2_"))
        again, calls = run()
        self.assertEqual(calls, 0)
        np.testing.assert_allclose(again, first)

    def test_corrupt_or_interrupted_oversampling_cache_is_regenerated(self):
        image = self.bcd / "SPITZER_I1_123_0002_6_C7915448_bcd.fits"
        _write_image(image, 2.0)
        savepath = str(self.root) + "/"
        cache = self.root / "Oversampled"

        def run():
            with patch(
                "SPCA.Photometry_Common.oversampling", side_effect=_upsample
            ) as sample:
                result = prepare_image(savepath, "r", [str(image)], [1], oversamp=True)
            return result, sample.call_count

        # An interrupted write leaves neither a partial cache nor a temporary file.
        with (
            patch(
                "SPCA.Photometry_Common.os.replace", side_effect=OSError("disk full")
            ),
            self.assertRaises(OSError),
        ):
            run()
        self.assertEqual(list(cache.iterdir()), [])

        expected, calls = run()
        self.assertEqual(calls, 1)
        (saved,) = cache.iterdir()
        for damaged in (b"", b"\x80\x04truncated", None):
            if damaged is None:
                # A readable pickle with the wrong shape is rejected too.
                np.ma.masked_array(np.zeros((1, 32, 32))).dump(saved)
            else:
                saved.write_bytes(damaged)
            result, calls = run()
            self.assertEqual(calls, 1)
            np.testing.assert_allclose(result, expected)
        # The regenerated cache is valid again.
        _, calls = run()
        self.assertEqual(calls, 0)

    def test_sky_dark_filename_selects_external_stack_and_prepares_image(self):
        dark_name = "SPITZER_I1_68573184_0002_6_C7915448_sdark.fits"
        cal = self.data / "r123/ch1/cal"
        cal.mkdir()
        (cal / dark_name).touch()  # lookup uses its name, not the calibration data
        stacks = self.root / "Calibration"
        stacks.mkdir()
        stack = stacks / ("correction_" + dark_name)
        fits.PrimaryHDU(np.ones((1, 32, 32))).writeto(stack)
        header = fits.Header(
            {
                "GAIN": 1.0,
                "EXPTIME": 1.0,
                "FLUXCONV": 1.0,
                "FRAMTIME": 1.0,
                "BMJD_OBS": 54768.0,
            }
        )
        image = self.bcd / "a_cbcd.fits"
        fits.PrimaryHDU(np.full((1, 32, 32), 2.0), header).writeto(image)
        selected = get_stacks(str(stacks), str(self.data), "r", science_product="cbcd")
        self.assertEqual(list(selected), [str(stack)])
        result = prepare_image(
            str(self.root),
            "r",
            [str(image)],
            [1],
            stacks=selected,
            addStack=True,
            i=0,
        )
        np.testing.assert_allclose(result[:, :-1], 3.0)
        blank = prepare_image(str(self.root), "r", [str(image)], [1], i=0)
        np.testing.assert_allclose(blank[:, :-1], 2.0)

    def test_each_aor_uses_its_own_correction_stack(self):
        header = fits.Header(
            {
                "GAIN": 1.0,
                "EXPTIME": 1.0,
                "FLUXCONV": 1.0,
                "FRAMTIME": 1.0,
                "BMJD_OBS": 54768.0,
            }
        )
        stacks_dir = self.root / "Calibration"
        stacks_dir.mkdir()
        lens = (3, 2)
        for (aor, dark_aor, offset), count in zip(
            (("r100", "68573184", 10.0), ("r200", "68573440", 100.0)), lens
        ):
            dark = f"SPITZER_I1_{dark_aor}_0002_6_C7915448_sdark.fits"
            cal = self.data / aor / "ch1/cal"
            cal.mkdir(parents=True)
            (cal / dark).touch()
            fits.PrimaryHDU(np.full((1, 32, 32), offset)).writeto(
                stacks_dir / ("correction_" + dark)
            )
            bcd = self.data / aor / "ch1/bcd"
            bcd.mkdir(parents=True)
            for n in range(count):
                fits.PrimaryHDU(np.zeros((1, 32, 32)), header).writeto(
                    bcd / f"{aor}_{n}_bcd.fits"
                )
        fnames, found_lens = get_fnames(str(self.data), "r")
        self.assertEqual(found_lens, list(lens))
        stacks = get_stacks(str(stacks_dir), str(self.data), "r")
        added = [
            prepare_image(
                str(self.root), "r", fnames, found_lens, stacks, addStack=True, i=i
            )[0, 0]
            for i in range(len(fnames))
        ]
        # The first image of the second AOR (i == 3) must use that AOR's stack.
        self.assertEqual(added, [10.0, 10.0, 10.0, 100.0, 100.0])


class Stop(Exception):
    """Stop a photometry run once its image list has been observed."""


class PhotometryPathTests(unittest.TestCase):
    """Each photometry entry point must load exactly the selected version."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.basepath = temporary.name + "/"
        self.data = Path(self.basepath, "planet/data/ch1")
        bcd = self.data / "r123/ch1/bcd"
        bcd.mkdir(parents=True)
        Path(self.basepath, "planet/analysis/ch1").mkdir(parents=True)
        header = fits.Header(
            {
                "FRAMTIME": 1.0,
                "BMJD_OBS": 54768.0,
                "GAIN": 1.0,
                "EXPTIME": 1.0,
                "FLUXCONV": 1.0,
            }
        )
        for name in ("a_bcd.fits", "a_cbcd.fits", "a_bunc.fits"):
            fits.PrimaryHDU(np.ones((2, 32, 32)), header).writeto(bcd / name)

    def loaded_by_pool(self, run, product):
        """Return the image names handed to ``prepare_image`` workers."""
        seen = []

        class Pool:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def map(self, func, inds):
                seen.extend(Path(name).name for name in func.args[2])
                raise Stop

        with (
            patch("SPCA.Photometry_Common._get_pool", return_value=Pool()),
            self.assertRaises(Stop),
        ):
            run(self.basepath, "r", "ch1", "planet", science_product=product)
        return seen

    def test_aperture_psf_and_pld_load_only_selected_version(self):
        for module in (Photometry_Aperture, Photometry_PSF, Photometry_PLD):
            for product in ("bcd", "cbcd"):
                with self.subTest(module=module.__name__, product=product):
                    self.assertEqual(
                        self.loaded_by_pool(module.get_lightcurve, product),
                        [f"a_{product}.fits"],
                    )

    def test_companion_opens_only_selected_version(self):
        for product in ("bcd", "cbcd"):
            with self.subTest(product=product):
                opener = Mock(side_effect=Stop)
                with (
                    patch("SPCA.Photometry_Companion.fits", Mock(open=opener)),
                    self.assertRaises(Stop),
                ):
                    Photometry_Companion.get_lightcurve(
                        str(self.data),
                        self.basepath,
                        "r",
                        "ch1",
                        True,
                        science_product=product,
                    )
                self.assertEqual(
                    Path(opener.call_args.args[0]).name, f"a_{product}.fits"
                )
                opener.assert_called_once()


if __name__ == "__main__":
    unittest.main()
