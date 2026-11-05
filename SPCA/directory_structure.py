"""Prepare Spitzer archive downloads for the photometry pipeline."""

import os
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def make_directory_structure(rootpath):
    """Prepare ZIP downloads or already-extracted BCD trees for SPCA.

    Members named ``AOR/chN/...`` are extracted to ``data/chN/AOR/chN/...``.
    Files outside a channel directory, such as download readmes, are preserved
    under ``data/``. Original ZIP files are moved to ``raw/`` after extraction.
    Existing output directories are reused, but archived ZIPs are not replaced.
    Programmatic downloads already in ``data/chN/AOR/chN/bcd/`` are reused
    without making ZIP copies. Repeated setup of extracted data is supported.

    Args:
        rootpath (str or pathlib.Path): Campaign root containing ZIPs or data/.

    Returns:
        str: Common AOR prefix, also written to ``analysis/aorSnippet.txt``.

    Raises:
        ValueError: No ZIPs or extracted BCDs, no channels, or invalid ZIP paths.
        FileExistsError: A ZIP with the same name already exists in ``raw/``.
    """
    rootpath = Path(rootpath)
    archives = sorted(
        path
        for path in rootpath.iterdir()
        if path.is_file() and path.suffix.lower() == ".zip"
    )
    channels = set()
    aors = set()
    for channel in ("ch1", "ch2", "ch3", "ch4"):
        for directory in sorted((rootpath / "data" / channel).glob("r*")):
            if not re.fullmatch(r"r[0-9]+", directory.name):
                continue
            bcd = directory / channel / "bcd"
            if any(path.is_file() for path in bcd.glob("*_bcd.fits")):
                channels.add(channel)
                aors.add(directory.name)
    if not archives and not aors:
        raise ValueError(f"No ZIP archives or extracted BCD data found in {rootpath}")

    archive_members = []
    for archive in archives:
        if (rootpath / "raw" / archive.name).exists():
            raise FileExistsError(f"Archive already exists in raw/: {archive.name}")

        members = []
        archive_channels = set()
        with ZipFile(archive) as zipfile:
            for member in zipfile.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(
                        f"Invalid path in {archive.name}: {member.filename}"
                    )

                parts = path.parts
                channel = None
                if len(parts) >= 2 and parts[1] in ("ch1", "ch2", "ch3", "ch4"):
                    channel = parts[1]
                    archive_channels.add(channel)
                    aors.add(parts[0])
                # Top-level AOR directory entries need no separate extraction.
                if channel is not None or not member.is_dir():
                    members.append((member, channel))

        if not archive_channels:
            raise ValueError(f"No AOR/chN directories found in {archive.name}")
        channels.update(archive_channels)
        archive_members.append((archive, members))

    aor_snippet = os.path.commonprefix(sorted(aors))[:5]
    if not aor_snippet:
        raise ValueError("AOR directory names have no common prefix")

    for channel in sorted(channels):
        for directory in (
            rootpath / "data" / channel,
            rootpath / "analysis" / channel,
            rootpath / "analysis" / "frameDiagnostics" / channel,
            rootpath / "analysis" / "photometryComparison" / channel,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    for archive, members in archive_members:
        with ZipFile(archive) as zipfile:
            for member, channel in members:
                destination = rootpath / "data"
                if channel is not None:
                    destination /= channel
                zipfile.extract(member, destination)

    (rootpath / "analysis" / "aorSnippet.txt").write_text(aor_snippet)
    (rootpath / "raw").mkdir(exist_ok=True)
    for archive in archives:
        archive.rename(rootpath / "raw" / archive.name)

    return aor_snippet
