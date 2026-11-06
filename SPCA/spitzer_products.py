"""IRAC science-image versions and download product profiles.

Each IRAC exposure can have two processing versions of its science image,
``*_bcd.fits`` and ``*_cbcd.fits`` (artifact-corrected). They are alternatives,
never independent exposures, so exactly one version is selected for analysis.

Profiles (all keep exactly one selected science version):

``spca`` (default)
    Science images plus ``cal/*_sdark.fits`` files, whose names select SPCA's
    optional correction stacks (``addStack=True``).
``science-qa``
    ``spca`` plus each selected image's uncertainty and mask files.
``archive``
    Every Level 1 ``bcd/`` and ``cal/`` FITS product, including both science
    versions; only the selected version is labelled ``science``.
"""

from __future__ import annotations

SCIENCE_PRODUCTS = ("bcd", "cbcd")
PRODUCT_PROFILES = ("spca", "science-qa", "archive")
DEFAULT_PROFILE = "spca"
DEFAULT_SCIENCE_PRODUCT = "bcd"
UNCERTAINTY_SUFFIXES = {"bcd": "_bunc.fits", "cbcd": "_cbunc.fits"}
MASK_SUFFIXES = ("_bimsk.fits", "_imsk.fits", "_brmsk.fits")
SKY_DARK_SUFFIX = "_sdark.fits"


def validate_science_product(science_product: str) -> str:
    """Validate the explicitly selected science-image processing version."""
    if science_product not in SCIENCE_PRODUCTS:
        raise ValueError(f"science_product must be one of {SCIENCE_PRODUCTS}")
    return science_product


def science_variant(filename: str) -> str | None:
    """Return an image's processing version, or None for non-science products."""
    return next(
        (
            product
            for product in SCIENCE_PRODUCTS
            if filename.endswith(f"_{product}.fits")
        ),
        None,
    )


def science_stem(filename: str) -> str:
    """Return the exposure stem shared by a file's BCD/CBCD/QA products."""
    return filename.rsplit("_", 1)[0]


def product_selection(
    profile: str = DEFAULT_PROFILE, science_product: str = DEFAULT_SCIENCE_PRODUCT
) -> dict:
    """Build a validated, serializable product selection."""
    if profile not in PRODUCT_PROFILES:
        raise ValueError(f"profile must be one of {PRODUCT_PROFILES}")
    validate_science_product(science_product)
    return {"profile": profile, "science_product": science_product}


def classify(filename: str, science_product: str, subdir: str = "bcd") -> str:
    """Label a retained Level 1 FITS file relative to the selected science version.

    Returns ``science`` only for the selected version, ``uncertainty`` only for
    its matching uncertainty type, ``mask`` for masks, and ``ancillary`` for
    everything else (including the unselected science version and uncertainty).
    """
    variant = science_variant(filename)
    if variant and subdir == "bcd":
        return "science" if variant == science_product else "ancillary"
    if filename.endswith(UNCERTAINTY_SUFFIXES[science_product]):
        return "uncertainty"
    if filename.endswith(MASK_SUFFIXES):
        return "mask"
    return "ancillary"


def selected_product(
    subdir: str, filename: str, selection: dict, science_stems: set[str]
) -> bool:
    """Whether a Level 1 ``bcd/`` or ``cal/`` FITS file belongs to a selection.

    ``science_stems`` are the exposure stems of the selected science images, so
    Science-QA only keeps uncertainty/mask files of planned exposures.
    """
    profile = selection["profile"]
    if profile == "archive":
        return subdir in ("bcd", "cal")
    if subdir == "cal":
        return filename.endswith(SKY_DARK_SUFFIX)
    if subdir != "bcd":
        return False
    product = classify(filename, selection["science_product"])
    if product == "science":
        return True
    return (
        profile == "science-qa"
        and product in ("uncertainty", "mask")
        and science_stem(filename) in science_stems
    )
