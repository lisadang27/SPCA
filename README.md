# SPCA

SPCA is an open-source, modular, and automated pipeline for Spitzer Phase Curve Analyses.

The full API can be found at [https://spca.readthedocs.io](https://spca.readthedocs.io).

## Installation Instructions

SPCA requires Python 3.9 or newer. To install it, run the following in a terminal:

```
        git clone https://github.com/lisadang27/SPCA.git
        cd SPCA
        python -m pip install .
        # IF you want to install george to run GP analyses, also then run
        python -m pip install ".[GP]"
```

For development, use `python -m pip install -e ".[dev]"` (or
`python -m pip install -e ".[dev,GP]"` for GP support) so edits to the package
are available without reinstalling. Then run `pre-commit install` to enable the
formatting and file checks on each commit. They format Python files and notebook
code, check JSON, TOML, and YAML files, and fix common whitespace issues. If a
hook changes a file, review and stage the changes before committing again. To
check specific files before committing, run `pre-commit run --files path/to/file`.

Please note that SPCA is in a state of testing and development. The developers of this repo have very limited time to spend on maintenance and development.

## Update Log

### Version 0.3
As of version 0.3, we have made some important changes. These include:
* Added the ability to perform full-frame photometry
* Added the ability to fit unbinned photometry
* Added the ability to decorrelate using PSF-fitted centroids using the "\_PSFX" mode suffix
* Added sigma-clipping along full time axis (not just within data cube). This comes at the cost of needing to load all frames into RAM at the same time which can be quite RAM intensive for some phase curves. Will consider allowing user to switch to using dask in the future to reduce RAM cost when performing photometry at the cost of slower run speeds.
* Changed BLISS knot density selection algorithm to be more like POET's algorithm
* Added a progress bar for when running photometry
* No updates on emcee freezing, but it now rarely ever occurs for me (Taylor). Perhaps the better initialization routine is helping here.

### Version 0.2
As of version 0.2, we have made some important changes and some bug fixes. These include:
* Renamed decorrelation and fitting file to Decorrelation (.py and .ipynb)
* Full integration of PLD and PLDAper models
* Reduced the amount of static code that the user sees and placed this in separate files instead.
* Changed how photometry is saved. This will not be noticeable unless you want to use PLDAper models which cannot be run with the old photometry.
* Also changed the units in which the photometry is saved so that it is easy to compute the photon noise limit - a calculation that was previously done incorrectly.
* The emcee fitting routine sometimes freezing still seems to be an issue as of v0.2. This seems to be caused by the combination of a large dataset and either a large model or a poorly initialized model. Previous attempts at forcing a timeout have failed without directly editing emcee or multiprocessing code. If this occurs to you and you need to analyze your data soon, I'd recommend removing the few lines that use multiprocessing - contact us if you have a hard time doing this. We are looking into different samplers (such as PyMC3) to resolve this issue.

## Package Usage

0. Follow the installation instructions at the top of this page to ensure you have all of the necessary dependencies.

1. Obtain Spitzer Level 1 BCD data using the [programmatic workflow](#programmatic-spitzerirac-downloads)
   or the [Spitzer Heritage Archive](https://sha.ipac.caltech.edu/applications/Spitzer/SHA/).
   For browser downloads, place the ZIP files in your planet/campaign directory.
   Keep independent observing campaigns in separate directories.

Most of the following commands have an .ipynb ending and .py ending option available, where the .py version is optimized for analyzing many data sets and the .ipynb file is optimized for viewing the analysis of a single data set. Each file has a portion at the top where you can set parameters which will determine the techniques.

2. Next, use the Make\_Directory\_Structure file to extract ZIPs or recognize
already-extracted programmatic downloads and set up the required directories.

ZIP files containing multiple IRAC channels are split into the matching
`data/chN/` directories, retaining the `AOR/chN/` paths used by the photometry
pipeline. All channels are preserved; inspect the observations to decide which
channel to analyze. The original ZIP files are moved to `raw/` after extraction.
Already-extracted `data/chN/r########/chN/bcd/` trees are reused without ZIP
copies. Running setup again on these trees is supported.

3. Then use the Everything\_Photometry file to perform a suite of different photometries on your data. The code will automatically select the best aperture photometry method which gives the lowest scatter after smoothing the raw flux by a boxcar filter of a width provided by the user.

4. Then use the QuickLook file to ensure that you have looked at the raw data (to ensure it looks reasonable) and to determine whether you want to remove the first AOR (in case it is a short AOR to be discarded). By looking at the raw data, you can gain some insight into how successful different decorrelation models might be.

5. Then decorrelate the data using the Decorrelation file. Most parameters here are explained with a nearby comment. One key parameter though is the "mode" which sets the decorrelation method used. Modes that contain "Poly#" use a 2-dimentionsal polynomial of order #, modes that contain "BLISS" use BiLinearly-Interpolated Subpixel Sensitivity mapping, modes that contain "GP" use a Gaussian Process using x and y centroid positions as covariates, and modes that contain "PLD#\_$x$" use Pixel Level Decorrelation of order # (1 or 2) and use a pixel stamp size of $ by $ (3x3 or 5x5). PLD performed using aperture photometry (the recommended PLD technique) can be accessed using "PLDAper#\_$x$". Each mode is then followed by an underscore and either "v1" or "v2" indicating the use of either a 1st or 2nd order sinusoidal model for the phase variations. If "PSFX" is present in the mode string for non-PLD models, the data are decorrelated using the PSF-fitted centroids rather than the flux-weighted mean centroids. If "ellipse" is present in the mode string, phase variations due to the elliptical shape of the planet are modelled. Any other text can be added to the mode keyword for your own convenience (e.g. "Poly2\_v1\_run2" or "PLDAper2\_5x5\_v1\_testingFit").

6. Finally, some tables containing a selection of the fitted parameters from each model run can be made using the MakeTables file. These tables will also highlight the best decorrelation method for each analysis, determined using delta-BIC (selecting your model is actually quite a challenging and complicated step, and user discretion is absolutely recommended).

## Programmatic Spitzer/IRAC downloads

Install optional archive dependencies (existing-file users do not need them):

```sh
python -m pip install -e ".[download]"
python Download_Spitzer_Data.py --help
```

Use [Download_Spitzer_Data.ipynb](./Download_Spitzer_Data.ipynb) for interactive
discovery, metadata inspection, AOR selection, preview, and explicit downloading.
Its default settings do **not** download FITS files. The thin
[Download_Spitzer_Data.py](./Download_Spitzer_Data.py) CLI and notebook share
[SPCA/spitzer_download.py](./SPCA/spitzer_download.py).

**Completeness limitation:** IRSA's SIA service has no per-AOR file inventory,
so a download is never reported as complete by itself. Files are found when their
footprints overlap the search circle during the AOR. Channels 1 and 3 view one
field and channels 2 and 4 an adjacent one, so frames from the pair that never
covers the target are not retrieved; for cryogenic full-array AORs this
typically yields only the two on-target channels. Large SIA queries can take
several minutes (a one-year window took about 25 minutes for WASP-12b). IRSA
TAP has no `spitzer_sha` ObsCore rows, and the historical SHA DataService is no
longer available (HTTP 404), so it is not used. Use `--compare`/`--reference`
(below) to check a plan against an independent listing. An unavailable service
raises an error, never a successful empty inventory. The workflow does not
invent download URLs or use SHA's browser-launching API.

### Discover, preview, and explicitly download

Coordinate searches bypass SIMBAD/name resolution entirely. Decimal-degree
coordinates are also accepted. Quote sexagesimal values; negative declinations
can be supplied with `--dec=-12:34:56`. A target name is only resolved when
coordinates are absent, so resolver outages do not affect coordinate searches.

```sh
# Metadata only: candidates in IRAC channels 1--4, no FITS downloads.
python Download_Spitzer_Data.py \
    --target WASP-12b --ra 06:30:32.797 --dec +29:40:20.26 \
    --channels 1 2 3 4 --output /path/to/WASP-12b/campaign

# Optional cryogenic-window metadata search (end is inclusive UTC).
python Download_Spitzer_Data.py \
    --ra 06:30:32.797 --dec +29:40:20.26 --channels 3 4 \
    --start 2008-01-01 --end 2008-12-31T23:59:59.999

# Substitute verified AORKEYs from discovery; this saves a plan but no FITS.
python Download_Spitzer_Data.py \
    --ra 06:30:32.797 --dec +29:40:20.26 --aor AORKEY1 AORKEY2 --channels 3 4 \
    --output /path/to/WASP-12b/cryogenic_campaign --preview

# Explicitly download or restart the saved plan.
python Download_Spitzer_Data.py \
    --manifest /path/to/WASP-12b/cryogenic_campaign/download_plan.ecsv --download
```

`--aor` and `--pid` accept lists. For example, `--pid 50517 50759 70060 90186`
can be combined with a positional search; these are candidate programs, **not
verified assertions that every AOR targets WASP-12b**. Criteria intersect,
while values within a list are alternatives. A later `--aor ... --preview` run
repeats the positional search (coordinates are still required with SIA) and
keeps only the listed AORs.

For WASP-12b, the 2008 channel 3/4 search above returned (from live archive
metadata) AOR 28566016 (PID 50759, 2008-10-29, ch4, 1560 BCDs) and AOR 28578560
(PID 50517, 2008-11-03, ch3, 1697 BCDs), plus short mapping AORs 28250112 and
28312320 that list no BCDs (reported with a warning). AOR 28566016 also holds
1560 channel-2 BCDs (per its archive quality report), which a `--channels 3 4`
search intentionally does not plan; `--compare` lists such channels. Discovery
and planning each re-query whole AORs, so previewing a multi-hour staring AOR
takes minutes.

#### How files are found

`Irsa.query_sia(collection="spitzer_sha")` returns **one record per archive
file**, matching every file whose image footprint overlaps the search circle
(`--radius`, default 0.5 arcmin); dithered or mapped frames whose centers lie
arcminutes away are therefore still found. The AORKEY is the record's
hexadecimal `obs_id`, the PID is the trailing number of `proposal_id`, and the
observing mode comes from `instrument_keywords` and the frame size. Each
selected AOR is then re-queried over its own time span (`--expansion-radius`,
default 0.5 arcmin), so a date filter selects whole AOR/channel observations
rather than clipping frames from them. A date-only bound means midnight UTC.

A date filter may catch only the beginning of a long AOR (e.g. a 32-hour phase
curve), so the AOR's full start/end is established before planning, in order
of preference, and recorded per AOR as `time_coverage`:

- `aor_metadata`: the time span of the AOR-level archive records (the AOR
  `DOC`/README files) returned by SIA;
- `qa_report`: the observation start/end in the archive's
  `QualityAnalysis_<AOR>.README` (a small text download);
- `gap_search` (warning): neither is available, so the search is extended until
  0.5 days beyond the first/last frame contain no further frames of the AOR;
- `unbounded` (warning): frames continue beyond 10 days, so the search stops;
- `full_mission`: the AOR was searched without a time constraint.

SIA has no PID or AOR parameter, so those filters are applied to positional
results. IRSA silently truncates SIA results at the row limit, so any query
reaching it is split in time and repeated until no part is truncated.

Products are classified by their archive path
(`.../r<AOR>/chN/<bcd|cal|raw|pbcd>/<file>`), not by IVOA `calib_level` (all
SHA files report level 2). By default, `bcd/` science `*_bcd.fits`, uncertainty
(`*_bunc`, `*_cbunc`), masks (`*_bimsk`, `*_brmsk`), other `bcd/` FITS products
(e.g. `*_cbcd`), and the `cal/` dark/mask files that SPCA reads are retained.
`cal/` files are stored under the science AOR even when the archive serves them
from a calibration AOR. `--bcd-only` retrieves only `*_bcd.fits`. Level 0
`raw/`, Level 2 `pbcd/`, logs, and AOR-level documents are never downloaded;
the preview reports every unplanned file, counted by reason.

#### Checking a plan against an independent listing

`--compare` checks the planned `bcd/` filenames of each AOR/channel against the
IRSA archive's own directory listing of `r<AOR>/chN/bcd/` and the
processed-frame count in the AOR's quality report (metadata only; no FITS data).
`--reference` instead compares against SHA ZIP files or extracted directories
you obtained independently, e.g. from the SHA web interface:

```sh
python Download_Spitzer_Data.py \
    --ra 06:30:32.797 --dec +29:40:20.26 --aor 28578560 --channels 3 \
    --start 2008-11-03 --end 2008-11-04 --output /path/to/campaign --compare
python Download_Spitzer_Data.py ... --reference /path/to/SHA_download.zip
```

```python
from SPCA.spitzer_download import compare_inventory

comparison = compare_inventory(plan)  # or compare_inventory(plan, "sha.zip")
print(comparison.meta["details"], comparison.meta["unplanned_channels_with_data"])
```

The result (also saved as `inventory_comparison.ecsv`) reports missing files
(in the reference but not planned), extra files (planned but not in the
reference), intentionally excluded files (logs, unsupported, or not-requested
ancillary products), channels holding data that were not planned, and an
`agreement` of `match`, `match_without_qa_count`, `mismatch`, or
`not_in_reference`. Only `bcd/` files are compared, because `cal/` files are
served from separate calibration AORs. A saved `download_inventory.ecsv` can
also be compared; it records whether ancillary files were requested for each
AOR/channel, so a BCD-only download does not report them as missing. The
archive directory listing is an HTML
index intended for browsing, not a versioned API, so an unparseable listing is
an error rather than an empty match.

### Verification, storage, and the next processing step

Downloads stream with one overall tqdm progress bar, configurable
`--timeout`/`--retries`, temporary files, and atomic completion. Interrupted
files restart from zero. HTTP lengths are checked and FITS structure is
validated without rewriting data. SIA sizes
(`access_estsize`) are rounded estimates used only for preview totals. Local
SHA-256 hashes, computed while streaming, verify previously completed files
before skipping them; IRSA supplies **no archive checksums**. The manifest and
inventory are rewritten atomically every 100 files or 30 seconds
(`download_plan(..., checkpoint_files=, checkpoint_seconds=)`) and on
completion, failure, or interruption, so AORs with thousands of files do not
rewrite them per file. Between checkpoints, each completed file's path, source
URL, size, and SHA-256 are appended to `download_journal.jsonl` before the file
is moved into place; after a hard kill or power failure, the next run adopts
journaled files whose size and hash still match, so they are neither downloaded
again nor treated as untracked. Both 2D full-frame images and 3D subarray
cubes are preserved byte-for-byte. Files are written directly into the SPCA
layout, so no duplicate ZIP/FITS storage is kept. Existing files the downloader
did not record (in `download_inventory.ecsv`, or in the journal with matching
size and hash) are never overwritten by default:
they are marked `conflict` in the manifest and the run stops with an error
after the other files finish. Move them aside or pass `--overwrite-untracked`
(`download_plan(..., overwrite_untracked=True)`). Corrupted files recorded in
the inventory are still repaired automatically.

The campaign directory contains:

- `discovered_observations.ecsv`: candidate AOR/channel metadata (CLI discovery).
- `download_plan.ecsv`: original links, estimated sizes, local paths, and
  per-AOR time coverage.
- `download_manifest.ecsv`: current transfer status and failure details.
- `download_inventory.ecsv`: actual files, original source URLs, sizes, and hashes.
- `download_journal.jsonl`: transient per-file provenance since the last
  inventory checkpoint (removed after a clean run).
- `download_completeness.ecsv`: local per-AOR/channel product counts and
  missing uncertainty/mask associations; archive completeness stays unverified.
- `inventory_comparison.ecsv`: optional `--compare`/`--reference` results.
- `data/chN/r########/chN/bcd/`: science/associated files in SPCA's existing layout.

Preview summaries show estimated bytes, files of unknown size, BCD counts, and
available PID/date/mode metadata.
`inventory_summary` counts science, uncertainty, mask, and ancillary files per
**AOR/channel**. Successful transfer means only that the returned products were
downloaded: by itself this leaves dataset
**completeness unverified**, even if science/uncertainty/mask counts match.
The report also identifies local BCDs lacking same-stem uncertainty/mask files;
this is an association check; use `compare_inventory` for a listing comparison.
Ancillary associations are not inferred by inventing filename-based download
URLs. Check the inventory and archive documentation before scientific analysis.

After a successful CLI download, directory setup runs automatically; notebook
users run its final setup cell. Alternatively, call `make_directory_structure`
from [SPCA/directory_structure.py](./SPCA/directory_structure.py) on the campaign
root. Configure `Everything_Photometry` with that root and the chosen
channel/AORs. The AOR snippet is not an epoch filter: do not automatically
analyze different campaigns together.

Offline regression tests require no IRSA access:

```sh
python -m unittest discover -s tests
```

Official references: [astroquery IRSA](https://docs.astropy.org/projects/astroquery/en/latest/ipac/irsa/irsa.html),
[IRSA APIs](https://irsa.ipac.caltech.edu/voapi.html),
[image APIs](https://irsa.ipac.caltech.edu/docs/program_interface/api_images.html),
[SHA knowledgebase](https://irsa.ipac.caltech.edu/docs/knowledgebase/spitzer_sha.html),
and [IRSA tutorials](https://caltech-ipac.github.io/irsa-tutorials/).

## Contributions

[Lisa Dang](https://github.com/lisadang27) contributed the initial code and idea for SPCA, and wrote much of the photometry and fitting routines, as well as the polynomial and PLD decorrelation methods.

Both authors worked extensively to debug the code and simplify the user experience.

[Taylor James Bell](https://github.com/taylorbell57) further generalized and streamlined SPCA, allowing it to be run quickly and easily for any given planet with minimal effort. Taylor also contributed most of the documentation, and the GP decorrelation method.

## Acknowledgements

We thank Joel Schwartz for his aid in writing out BLISS decorrelation method. We also thank Dylan Keating for alpha testing SPCA. We thank Antoine Darveau-Bernier for writing code to parse through the Exoplanet Archive data and select the best constrained value for each parameter in the database (code can be found [here](https://github.com/AntoineDarveau/masterfile)).

## License & Attribution

Copyright © 2018-2020 Lisa Dang & Taylor James Bell.

SPCA is free software made available under the GPL3 License. For details
see the [LICENSE](https://github.com/lisadang27/SPCA/blob/master/LICENSE).

If you make use of SPCA in your work, please cite the Dang et al. (2018) paper that was the first to use this pipeline
[arXiv](https://arxiv.org/abs/1801.06548),
[ADS](https://ui.adsabs.harvard.edu/abs/2018NatAs...2..220D),
[BibTeX](https://ui.adsabs.harvard.edu/abs/2018NatAs...2..220D/exportcitation>).
