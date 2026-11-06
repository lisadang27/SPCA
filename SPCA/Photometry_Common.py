import glob
import hashlib
import os
import pickle
import warnings
from functools import partial
from multiprocessing import get_context

import numpy as np
import scipy.interpolate
from astropy.convolution import Box1DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from .spitzer_products import validate_science_product

warnings.filterwarnings("ignore")

image_stack = np.zeros((0, 32, 32))


def _get_pool(processes):
    """Create a multiprocessing pool that preserves shared module state.

    The aperture and PSF routines use module-level image data to avoid
    pickling large arrays. macOS uses ``spawn`` by default, which does not
    inherit that state; use ``fork`` where available and fall back to the
    platform default elsewhere.
    """
    try:
        context = get_context("fork")
    except ValueError:
        context = get_context()
    return context.Pool(processes)


def create_folder(fullname, auto=False, overwrite=False):
    """Create a folder unless it exists.

    Args:
        fullname (string): Full path to the folder to be created.
        auto (bool, optional): If the folder already exists, should the folder just be skipped (True)
            or should the user be asked whether they want to overwrite the folder or change the folder name (False, Default).
        overwrite (bool, optional): Whether you want to overwrite the folder if it already exists

    Returns:
        string: The final name used for the folder.

    """

    solved = "no"
    while solved == "no":
        if not os.path.exists(fullname):
            # Folder doesn't exist yet and can be safely written to
            os.makedirs(fullname)
            solved = "yes"
        elif len(os.listdir(fullname)) == 0:
            # Folder exists but is empty and can be safely written to
            solved = "yes"
        else:
            if overwrite:
                solved = "yes"
            elif auto:
                fullname = None
                solved = "yes"
            else:
                folder = fullname.split("/")[-1]
                print(
                    "Warning:",
                    folder,
                    "already exists! Are you sure you want to overwrite this folder? (y/n)",
                )
                answer = input()
                if answer == "y":
                    solved = "yes"
                else:
                    print("What would you like the new folder name to be?")
                    folder = input()
                    fullname = "/".join(fullname.split("/")[0:-1]) + "/" + folder
    return fullname


def get_fnames(directory, AOR_snip, science_product="bcd"):
    """Find paths to all the fits files.

    Args:
        directory (string): Path to the directory containing all the Spitzer data.
        AOR_snip (string): Common first characters of data directory eg. 'r579'.
        science_product (string): Select exactly 'bcd' (default) or 'cbcd',
            never both processing versions of an exposure.

    Returns:
        tuple: fname, lens (list, list).
            List of paths to all bcd.fits files, number of files for each AOR (needed for adding correction stacks).

    """

    validate_science_product(science_product)
    while directory[-1] == "/":
        directory = directory[:-1]
    ch = directory.split("/")[-1]

    lst = os.listdir(directory)
    AOR_list = sorted(folder for folder in lst if AOR_snip == folder[: len(AOR_snip)])
    fnames = []
    lens = []
    for i in range(len(AOR_list)):
        path = directory + "/" + AOR_list[i] + "/" + ch + "/bcd"
        files = np.sort(glob.glob(os.path.join(path, f"*_{science_product}.fits")))
        if len(files) != 0:
            fnames.extend(files)
            lens.append(len(files))
    return fnames, lens


def get_stacks(calDir, dataDir, AOR_snip, science_product="bcd"):
    """Find paths to all the background subtraction correction stacks FITS files.

    Args:
        calDir (string): Path to the directory containing the correction stacks.
        dataDir (string): Path to the directory containing the Spitzer data to be corrected.
        AOR_snip (string): Common first characters of data directory eg. 'r579'.
        science_product (string): Match AORs containing the selected image type.

    Returns:
        list: List of paths to the relevant correction stacks

    """

    while dataDir[-1] == "/":
        dataDir = dataDir[:-1]
    ch = dataDir.split("/")[-1]

    stacks = np.array(os.listdir(calDir))
    locs = np.array([stacks[i].find("SPITZER_I") for i in range(len(stacks))])
    good = np.where(
        locs != -1
    )[
        0
    ]  # filter out all files that don't fit the correct naming convention for correction stacks
    offset = 11  # legth of the string "SPITZER_I#_"
    keys = np.array(
        [stacks[i][locs[i] + offset :].split("_")[0] for i in good]
    )  # pull out just the key that says what sdark this stack is for

    lst = os.listdir(dataDir)
    fnames, _ = get_fnames(dataDir, AOR_snip, science_product)
    selected_aors = {
        os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
        for f in fnames
    }
    AOR_list = sorted(folder for folder in lst if folder in selected_aors)
    calFiles = []
    for i in range(len(AOR_list)):
        path = dataDir + "/" + AOR_list[i] + "/" + ch + "/cal/"
        if not os.path.isdir(path):
            print(
                "Error: Folder '"
                + path
                + "' does not exist, so automatic correction stack selection cannot be performed"
            )
            return []
        darks = sorted(glob.glob(path + "*_sdark.fits"))
        if not darks:
            raise ValueError(
                f"No sky-dark filename available for correction-stack selection: {path}"
            )
        fname = darks[0]
        loc = fname.find("SPITZER_I") + offset
        key = fname[loc:].split("_")[0]
        calFiles.append(
            os.path.join(calDir, stacks[list(good)][np.where(keys == key)[0][0]])
        )
    return np.array(calFiles)


def get_time(header, ignoreFrames):
    """Gets the time stamp for each image.

    Args:
        header (astropy.io.fits.header.Header): Header of fits file.
        ignoreFrames (ndarray): Array of frames to ignore (consistently bad frames).

    Returns:
        ndarray: Updated time stamp array.

    """

    if header["NAXIS"] == 2:
        h = 1
    else:
        h = header["NAXIS3"]
    sec2day = 1.0 / (3600.0 * 24.0)
    step = header["FRAMTIME"] * sec2day
    t = np.linspace(
        header["BMJD_OBS"] + step / 2,
        header["BMJD_OBS"] + (h - 1) * step,
        h,
        endpoint=True,
    )
    t = np.ma.masked_invalid(t)
    if ignoreFrames != []:
        t[ignoreFrames].mask = True
    return t


def oversampling(image_data, scale=2):
    """First, substitutes all invalid/sigma-clipped pixels by interpolating the value, then linearly oversamples the image.

    Warning:
        This implementation is currently nonfunctional (it references an
        undefined scale variable, and its interpolation grid transposes the
        image) and needs separate scientific validation before use, so
        ``oversamp=True`` should not be used. When it is corrected, increment
        ``OVERSAMPLE_CACHE_VERSION`` so caches from the earlier algorithm are
        never reused.

    Args:
        image_data (ndarray): Data cube of images (2D arrays of pixel values).
        scale (int, optional):  Sampling factor, e.g. if scale = 2, there will be twice as many data points in the x
            and y axis. Default is 2. (Do not recommend larger than 2)

    Returns:
        ndarray: Data cube of oversampled images (2D arrays of pixel values).

    """

    l, h, w = image_data.shape
    gridy, gridx = np.mgrid[0 : h : 1 / a, 0 : w : 1 / a]
    image_over = np.zeros((l, h * a, w * a))
    for i in range(l):
        image_masked = np.ma.masked_invalid(image_data[i, :, :])
        if np.all(image_masked.mask):
            image_over[i, :, :].mask = True
            continue
        points = np.where(image_masked.mask == False)
        image_compre = np.ma.compressed(image_masked)
        image_over[i, :, :] = scipy.interpolate.griddata(
            points, image_compre, (gridx, gridy), method="linear"
        )

    # Mask any bad values
    image_masked = np.ma.masked_invalid(image_over)

    # conserve flux
    return image_masked / (a**2)


def sigma_clipping(image_stack, bounds=(13, 18, 13, 18), sigma=5, maxiters=3):
    """Sigma clips bad pixels and mask entire frame if the sigma clipped pixel is too close to the target.

    Args:
        image_stack (3D array): Data cube of images (2D arrays of pixel values).
        bounds (tuple, optional): Bounds of box around the target. Default is (13, 18, 13, 18).
        sigma (float, optional): How many sigma should something differ by to be clipped. Default is 5 which shouldn't trim any real data for Ndata=64*1000.
        maxiters (int, optional): How many iterations of sigma clipping should be done.

    Returns:
        3D array: sigma_clipped_data - Data cube of sigma clipped images (2D arrays of pixel values).

    """

    image_stack = np.ma.masked_invalid(image_stack, copy=False)
    lbx, ubx, lby, uby = bounds

    # Using a nested for-loop to significantly reduce RAM
    for i in range(image_stack.shape[1]):
        for j in range(image_stack.shape[2]):
            try:
                image_stack[:, i, j] = sigma_clip(
                    image_stack[:, i, j].flatten(),
                    sigma=sigma,
                    maxiters=maxiters,
                    cenfunc=np.ma.median,
                    stdfunc=np.ma.std,
                )
            except TypeError:
                image_stack[:, i, j] = sigma_clip(
                    image_stack[:, i, j].flatten(),
                    sigma=sigma,
                    iters=maxiters,
                    cenfunc=np.ma.median,
                    stdfunc=np.ma.std,
                )

    # If any pixels near the target star are bad, mask the entire frame
    image_stack[np.any(image_stack.mask[:, lbx:ubx, lby:uby], axis=(1, 2))] = (
        np.ma.masked
    )

    return image_stack


def clip_data(arr, highpassWidth, sigma1=5, sigma2=5, maxiters=3, replace_masked=True):
    """Clip outliers, optionally retaining rejections instead of imputing them.

    Aperture extraction uses ``replace_masked=False`` so missing measurements
    and clipped outliers never become synthetic science measurements. The
    default retains the legacy replacement behavior for other callers.
    """
    arr = np.ma.masked_invalid(np.ma.array(arr, dtype=float, copy=True))
    if arr.count() == 0:
        return arr
    try:
        arr = sigma_clip(
            arr,
            sigma=sigma1,
            maxiters=maxiters,
            cenfunc=np.ma.median,
            stdfunc=np.ma.std,
        )
    except TypeError:
        arr = sigma_clip(
            arr, sigma=sigma1, iters=maxiters, cenfunc=np.ma.median, stdfunc=np.ma.std
        )
    if replace_masked:
        arr = replace_clipped(arr)
    arr = rolling_clip(arr, highpassWidth, sigma=sigma2, maxiters=maxiters)
    if replace_masked:
        arr = replace_clipped(arr)

    return arr


def replace_clipped(arr):
    for i in np.where(np.ma.getmaskarray(arr))[0]:
        inds = np.array([i - 2, i - 1, i + 1, i + 2])
        inds = inds[inds < len(arr)]

        if not np.all(arr[inds].mask):
            arr[i] = np.ma.median(arr[inds])
        else:
            arr[i] = np.ma.median(arr)

    arr.mask = np.ma.nomask

    return arr


def rolling_clip(arr, highpassWidth, sigma=5, maxiters=3):
    smooth = highpassflist(arr, highpassWidth)
    smoothed = arr - smooth
    try:
        smoothed = sigma_clip(
            smoothed,
            sigma=sigma,
            maxiters=maxiters,
            cenfunc=np.ma.median,
            stdfunc=np.ma.std,
        )
    except TypeError:
        smoothed = sigma_clip(
            smoothed,
            sigma=sigma,
            iters=maxiters,
            cenfunc=np.ma.median,
            stdfunc=np.ma.std,
        )
    arr[np.ma.getmaskarray(smoothed)] = np.ma.masked
    return arr


def bgsubtract(bounds=(11, 19, 11, 19), i=0):
    """Measure background level and scatter for subtraction in the parent.

    Args:
        bounds (tuple, optional): Bounds of box around the target. Default is (11, 19, 11, 19).

    Returns:
        ndarray: Background median and per-pixel standard deviation. Both are
            NaN if no background pixels are available.

    """

    lbx, ubx, lby, uby = bounds

    # Get the initial mask
    image = np.ma.masked_invalid(image_stack[i])

    # Mask out the target star
    image[lbx:ubx, lby:uby] = np.ma.masked

    # Compute background and error
    bg = np.ma.median(image)
    bg_err = np.ma.std(image)

    # Workers measure only; the parent subtracts from its own image stack.
    return np.ma.stack((bg, bg_err)).filled(np.nan)


def bin_array(data, size):
    """Median bin an array.

    Args:
        data (1D array): Array of data to be binned.
        size (int): Number of data points in each bin.

    Returns:
        tuple: binned_data (1D array) Array of binned data.
            binned_data_std (1D array) Array of standard deviation for each entry in binned_data.

    """

    if size < 1 or int(size) != size:
        raise ValueError("Bin size must be a positive integer")
    size = int(size)
    data = np.ma.masked_invalid(np.ma.array(data, dtype=float))
    nbin = (len(data) + size - 1) // size
    binned_data = np.full(nbin, np.nan)
    binned_data_std = np.full(nbin, np.nan)
    # scipy.stats.binned_statistic converts masked arrays to ndarrays, exposing
    # the rejected values underneath the mask. Reduce each masked bin directly.
    for i in range(nbin):
        values = data[i * size : (i + 1) * size]
        if values.count():
            binned_data[i] = np.ma.median(values)
            binned_data_std[i] = np.ma.std(values)

    return binned_data, binned_data_std


def highpassflist(signal, highpassWidth):
    g = Box1DKernel(highpassWidth)
    signal = np.ma.array(signal, dtype=float).filled(np.nan)
    return convolve(signal, g, boundary="extend")


# Increment whenever oversampling() or the cached data format changes, so caches
# produced by an earlier algorithm are never reused. In particular, the current
# oversampling() is nonfunctional; fixing it requires incrementing this value.
OVERSAMPLE_CACHE_VERSION = 1


def oversample_cache_name(fname, image, scale):
    """Return the oversampling cache filename for one prepared science image.

    The name combines the image's full archive filename stem (AOR, exposure,
    pipeline version, and BCD/CBCD version), the oversampling ``scale``, and a
    SHA-256 digest of the prepared image that is about to be oversampled, e.g.
    ``SPITZER_I1_28578560_0002_6_C7915448_bcd_x2_<digest>.pkl``. The image is
    hashed after correction stacks, ignored frames, sigma clipping, unit
    conversion, and star masking, so any change to those inputs or settings
    selects a different cache instead of reusing a stale one. Older caches
    (``0002.pkl`` or names without a digest) are never matched.
    """
    digest = hashlib.sha256()
    digest.update(f"v{OVERSAMPLE_CACHE_VERSION};scale={scale};".encode())
    digest.update(f"{image.dtype.str};{image.shape};".encode())
    digest.update(np.ascontiguousarray(np.ma.getmaskarray(image)).tobytes())
    digest.update(np.ascontiguousarray(np.ma.filled(image, 0)).tobytes())
    stem = os.path.splitext(os.path.basename(fname))[0]
    return f"{stem}_x{scale}_{digest.hexdigest()[:16]}.pkl"


def _load_oversample_cache(savename, expected_shape):
    """Return a cached oversampled image, or None if it is missing or unusable."""
    if not os.path.isfile(savename):
        return None
    try:
        # Caches are pickled masked arrays written by SPCA itself.
        cached = np.load(savename, allow_pickle=True)
    except (OSError, EOFError, ValueError, pickle.UnpicklingError) as error:
        print(f"Warning: ignoring unreadable oversampling cache {savename}: {error}")
        return None
    if not isinstance(cached, np.ma.MaskedArray) or cached.shape != expected_shape:
        print(f"Warning: ignoring malformed oversampling cache {savename}")
        return None
    return cached


def _save_oversample_cache(image, savename):
    """Atomically write a cache so interrupted writes never leave a partial file."""
    os.makedirs(os.path.dirname(savename) or ".", exist_ok=True)
    temporary = f"{savename}.{os.getpid()}.tmp"
    try:
        image.dump(temporary)
        os.replace(temporary, savename)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def prepare_image(
    savepath,
    AOR_snip,
    fnames,
    lens,
    stacks=[],
    ignoreFrames=[],
    oversamp=False,
    scale=2,
    reuse_oversamp=True,
    saveoversamp=True,
    addStack=False,
    stackPath="",
    maskStars=[],
    i=0,
):

    # open fits file
    with fits.open(fnames[i]) as hdu_list:
        header = hdu_list[0].header
        time = get_time(header, ignoreFrames)

        if len(hdu_list[0].data.shape) == 2:
            # Reshape fullframe data so that it can be used with our routines
            # Getting just the stamp around the sweet spot
            image = np.ma.masked_invalid(hdu_list[0].data[np.newaxis, 217:249, 9:41])
        else:
            image = hdu_list[0].data
            # ignore any consistently bad frames in datacubes
            image[ignoreFrames] = np.nan
            image = np.ma.masked_invalid(image)

    # add background correcting stack if requested
    if addStack:
        # Index of the AOR containing image i (lens counts images per AOR).
        j = int(np.searchsorted(np.cumsum(lens), i, side="right"))
        with fits.open(stacks[j]) as stackHDU:
            image += np.ma.masked_invalid(stackHDU[0].data)

    if image.shape[0] != 1:
        # Sigma clipping within datacubes
        image = sigma_clipping(image, sigma=4.0)

    # convert MJy/str to electron count
    convfact = header["GAIN"] * header["EXPTIME"] / header["FLUXCONV"]
    image = convfact * image

    # Mask any other stars in the frame to avoid them influencing the background subtraction
    if maskStars != []:
        header["CTYPE3"] = (
            "Time-SIP"  # Just need to add a type so astropy doesn't complain
        )
        w = WCS(header, naxis=[1, 2])
        mask = np.ma.getmaskarray(image)
        for st in maskStars:
            coord = SkyCoord(st[0], st[1])
            x, y = np.rint(skycoord_to_pixel(coord, w)).astype(int)
            x = x + np.arange(-1, 2)
            y = y + np.arange(-1, 2)
            x, y = np.meshgrid(x, y)
            mask[:, x, y] = True
        image = np.ma.masked_array(image, mask=mask)

    # oversampling
    if oversamp:
        savename = (
            savepath + "Oversampled/" + oversample_cache_name(fnames[i], image, scale)
        )
        n, h, w = image.shape
        cached = None
        if reuse_oversamp:
            cached = _load_oversample_cache(savename, (n, h * scale, w * scale))
        if cached is not None:
            image = cached
        else:
            if reuse_oversamp:
                print(
                    "Warning: No valid oversampled image cache for these settings! Making a new one now..."
                )
            image = np.ma.masked_invalid(oversampling(image, scale))
            if saveoversamp:
                _save_oversample_cache(image, savename)

    data = image.reshape(-1, np.prod(image.shape[1:]))
    data = np.ma.concatenate((data, time[:, np.newaxis]), axis=1)

    return data


def prepare_images(
    basepath,
    planet,
    channel,
    AOR_snip,
    ignoreFrames=[],
    oversamp=False,
    scale=2,
    reuse_oversamp=True,
    saveoversamp=True,
    addStack=False,
    maskStars=[],
    ncpu=4,
    science_product="bcd",
):
    """Prepare exactly one science-image version for photometry.

    ``science_product`` selects 'bcd' (default) or 'cbcd'. Uncertainty and mask
    archive products are not loaded; optional correction-stack lookup uses
    sky-dark filenames for the same selected AORs.
    """

    # folder containing properly named correction stacks (will be automatically selected)
    stackPath = basepath + "Calibration/"
    datapath = basepath + planet + "/data/" + channel
    savepath = basepath + planet + "/analysis/" + channel + "/"
    if addStack:
        savepath += "addedStack/"
    else:
        savepath += "addedBlank/"

    if maskStars is None:
        maskStars = []
    if ignoreFrames is None:
        ignoreFrames = []

    print("\tGetting frames", end="", flush=True)
    if len(ignoreFrames) != 0:
        print(", masking bad frames", end="", flush=True)
    print("... ", end="")
    # get list of filenames and number of files
    fnames, lens = get_fnames(datapath, AOR_snip, science_product)
    if not fnames:
        raise ValueError(f"No *_{science_product}.fits images found in {datapath}")

    # get path where the aor breaks will be saved
    breakpath = basepath + planet + "/analysis/" + channel + "/aorBreaks.txt"
    # get & write aor breaks
    index = 0
    breaktimes = []
    for length in lens:
        with fits.open(fnames[index]) as rawImage:
            header = rawImage[0].header
            breaktimes.append(get_time(header, []).flatten()[0])
        index += length
    breaktimes = np.sort(breaktimes)[1:]
    with open(breakpath, "w") as f:
        f.write(str(breaktimes)[1:-1])

    # if need to add correction stack
    if addStack:
        stacks = get_stacks(stackPath, datapath, AOR_snip, science_product)
    else:
        stacks = []

    # Load all of the images using multiprocessing to speed things up
    with _get_pool(ncpu) as pool:
        func = partial(
            prepare_image,
            savepath,
            AOR_snip,
            fnames,
            lens,
            stacks,
            ignoreFrames,
            oversamp,
            scale,
            reuse_oversamp,
            saveoversamp,
            addStack,
            stackPath,
            maskStars,
        )
        inds = range(len(fnames))
        results = np.ma.concatenate(pool.map(func, inds), axis=0)

    # Access global variable
    global image_stack

    time = results[:, -1].flatten()
    image_stack = results[:, :-1].reshape(-1, 32, 32)
    # Free up a bit of RAM
    results = None

    # Sort data into correct order
    order = np.argsort(time)
    time = time[order]
    image_stack = image_stack[order]
    # Free up a bit of RAM
    order = None

    print("Sigma clipping... ", end="", flush=True)
    # sigma clip bad pixels along full time axis
    image_stack = sigma_clipping(image_stack, sigma=5)

    print("Subtracting background... ", end="", flush=True)
    # background subtraction is done on global variable
    with _get_pool(ncpu) as pool:
        func = partial(bgsubtract, (11, 19, 11, 19))
        inds = range(image_stack.shape[0])
        bg, bg_err = np.array(pool.map(func, inds)).T

    # Forked workers do not modify the parent's memory. Subtract here, masking
    # frames for which no background could be measured.
    image_stack -= np.ma.masked_invalid(bg)[:, np.newaxis, np.newaxis]

    print("Frames loaded!", flush=True)

    return image_stack, bg, bg_err, time
