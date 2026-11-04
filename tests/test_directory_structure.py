"""Regression tests for channel routing when unpacking Spitzer downloads."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from SPCA.directory_structure import make_directory_structure


class TestDirectoryStructure(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

    def make_archive(self, members, name="download.zip"):
        path = self.root / name
        with ZipFile(path, "w") as archive:
            for filename, contents in members:
                archive.writestr(filename, contents)
        return path

    def assert_analysis_directories(self, channel):
        for directory in (
            self.root / "analysis" / channel,
            self.root / "analysis" / "frameDiagnostics" / channel,
            self.root / "analysis" / "photometryComparison" / channel,
        ):
            self.assertTrue(directory.is_dir(), directory)

    def test_mixed_channels_are_routed_by_each_member(self):
        archive = self.make_archive(
            [
                ("download_readme.txt", b"Download metadata"),
                ("r1234501/", b""),
                ("r1234501/ch1/bcd/first_bcd.fits", b"Channel 1, AOR 1"),
                ("r1234501/ch2/bcd/first_bcd.fits", b"Channel 2, AOR 1"),
                ("r1234502/ch2/bcd/second_bcd.fits", b"Channel 2, AOR 2"),
                ("r1234502/ch1/cal/dark.fits", b"Channel 1 calibration"),
            ]
        )
        original_zip = archive.read_bytes()

        self.assertEqual(make_directory_structure(self.root), "r1234")

        expected = {
            "data/download_readme.txt": b"Download metadata",
            "data/ch1/r1234501/ch1/bcd/first_bcd.fits": b"Channel 1, AOR 1",
            "data/ch2/r1234501/ch2/bcd/first_bcd.fits": b"Channel 2, AOR 1",
            "data/ch2/r1234502/ch2/bcd/second_bcd.fits": b"Channel 2, AOR 2",
            "data/ch1/r1234502/ch1/cal/dark.fits": b"Channel 1 calibration",
        }
        actual = {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in (self.root / "data").rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual, expected)
        for channel in ("ch1", "ch2"):
            self.assert_analysis_directories(channel)
        self.assertEqual((self.root / "analysis/aorSnippet.txt").read_text(), "r1234")
        self.assertFalse(archive.exists())
        self.assertEqual((self.root / "raw/download.zip").read_bytes(), original_zip)

    def test_single_channel_archive_retains_the_pipeline_layout(self):
        for number in range(1, 5):
            channel = f"ch{number}"
            with self.subTest(channel=channel):
                self.make_archive(
                    [
                        (f"r1234501/{channel}/", b""),
                        (f"r1234501/{channel}/bcd/image_bcd.fits", channel.encode()),
                    ],
                    name=f"{channel}.zip",
                )
                make_directory_structure(str(self.root))
                image = (
                    self.root
                    / "data"
                    / channel
                    / "r1234501"
                    / channel
                    / "bcd/image_bcd.fits"
                )
                self.assertEqual(image.read_bytes(), channel.encode())
                self.assert_analysis_directories(channel)

    def test_multiple_archives_merge_into_existing_channel_directories(self):
        existing = self.root / "data/ch1/r1234501/ch1/bcd/existing_bcd.fits"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"Existing observation")
        first = self.make_archive(
            [
                ("r1234501/ch1/bcd/first_bcd.fits", b"First observation"),
            ]
        )
        second = self.make_archive(
            [
                ("r1299901/ch2/bcd/second_bcd.fits", b"Second observation"),
            ],
            name="second.ZIP",
        )
        ignored = self.root / "not-an-archive.zip.txt"
        ignored.write_text("Not a ZIP")

        self.assertEqual(make_directory_structure(self.root), "r12")

        self.assertEqual(existing.read_bytes(), b"Existing observation")
        self.assertEqual(
            (self.root / "data/ch1/r1234501/ch1/bcd/first_bcd.fits").read_bytes(),
            b"First observation",
        )
        self.assertEqual(
            (self.root / "data/ch2/r1299901/ch2/bcd/second_bcd.fits").read_bytes(),
            b"Second observation",
        )
        self.assertTrue((self.root / "raw" / first.name).is_file())
        self.assertTrue((self.root / "raw" / second.name).is_file())
        self.assertTrue(ignored.is_file())

    def test_no_archives_does_not_create_output_directories(self):
        with self.assertRaisesRegex(ValueError, "No ZIP archives"):
            make_directory_structure(self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_archive_without_channel_paths_is_left_in_place(self):
        archive = self.make_archive([("readme.txt", b"No observations")])
        with self.assertRaisesRegex(ValueError, "No AOR/chN directories"):
            make_directory_structure(self.root)
        self.assertEqual(list(self.root.iterdir()), [archive])

    def test_existing_raw_archive_is_not_overwritten(self):
        archive = self.make_archive([("r1234501/ch1/bcd/image.fits", b"New data")])
        (self.root / "raw").mkdir()
        archived = self.root / "raw" / archive.name
        archived.write_bytes(b"Original archive")

        with self.assertRaises(FileExistsError):
            make_directory_structure(self.root)

        self.assertEqual(archived.read_bytes(), b"Original archive")
        self.assertTrue(archive.is_file())
        self.assertFalse((self.root / "data").exists())

    def test_invalid_member_paths_are_rejected_before_extraction(self):
        for filename in ("../ch1/image.fits", "/r123/ch1/image.fits"):
            with self.subTest(filename=filename):
                archive = self.make_archive(
                    [
                        ("r1234501/ch1/bcd/image.fits", b"Observation"),
                        (filename, b"Invalid member"),
                    ]
                )
                with self.assertRaisesRegex(ValueError, "Invalid path"):
                    make_directory_structure(self.root)
                self.assertTrue(archive.is_file())
                self.assertFalse((self.root / "data").exists())

    def test_bad_zip_does_not_move_other_archives(self):
        archive = self.make_archive([("r1234501/ch1/bcd/image.fits", b"Observation")])
        (self.root / "second.zip").write_bytes(b"Invalid ZIP")
        with self.assertRaises(BadZipFile):
            make_directory_structure(self.root)
        self.assertTrue(archive.is_file())
        self.assertFalse((self.root / "data").exists())


if __name__ == "__main__":
    unittest.main()
