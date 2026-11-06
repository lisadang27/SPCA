import os
import sys
import warnings
from collections.abc import Iterable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from photutils.aperture import (
    CircularAperture,
    EllipticalAperture,
    RectangularAperture,
    aperture_photometry,
)
from photutils.utils import calc_total_error
from tqdm import tqdm

from .Photometry_Common import (
    _get_pool,
    bin_array,
    clip_data,
    create_folder,
    highpassflist,
    prepare_images,
)

warnings.filterwarnings("ignore")

# Make into a global variable so that A_Photometry can be run with multiprocessing without
# needing to pickle image_stack - this is critical for datasets with many GB of data!
image_stack = np.zeros((0, 32, 32))

# We need to resort to some hackery to make a tqdm progress bar work with multiprocessing
results = []
func = lambda arg: arg
pbar = None


def wrapMyFunc(arg):
    global func
    return arg, func(arg)


def update(outputs):
    # note: input comes from async `wrapMyFunc`
    global results
    results[outputs[0]] = outputs[1]  # put answer into correct index of result list
    global pbar
    pbar.update()
    return


def noisepixparam(image_stack, bounds=(13, 18, 13, 18)):
    """Compute the noise pixel parameter.

    Args:
        image_stack (ndarray): FITS images stack.
        npp (list, optional): Previously computed noise pixel parameters for other frames that will be appended to.

    Returns:
        list: The noise pixel parameter for each image in the stack.

    """

    lbx, ubx, lby, uby = bounds

    # To find noise pixel parameter for each frame. For eqn, refer Knutson et al. 2012
    numer = np.ma.sum(image_stack[:, lbx:ubx, lby:uby], axis=(1, 2)) ** 2
    denom = np.ma.sum(image_stack[:, lbx:ubx, lby:uby] ** 2, axis=(1, 2))

    return numer / denom


def centroid_FWM(
    image_stack,
    highpassWidth=5 * 64,
    scale=1,
    bounds=(13, 18, 13, 18),
    defaultCentroid=["median", "median"],
    defaultPSFW=["median", "median"],
):
    """Gets the centroid of the target by flux weighted mean and the PSF width of the target.

    Args:
        image_stack (ndarray): Data cube of images (2D arrays of pixel values).
        scale (int, optional): If the image is oversampled, scaling factor for centroid and bounds,
            i.e, give centroid in terms of the pixel value of the initial image.
        bounds (tuple, optional): Bounds of box around the target to exclude background . Default is (14, 18, 14, 18).
        defaultCentroid (list, optional): Default location for sigma clipped centroids.
            Default is median centroid position.
        defaultPSFW (list, optional): Default width for sigma clipped PSF widths.
            Default is median of widths.

    Returns:
        tuple: xo, yo, wx, wy (list, list, list, list). The updated lists of x-centroid, y-centroid,
            PSF width (x-axis), and PSF width (y-axis).
    """

    lbx, ubx, lby, uby = np.array(bounds) * scale
    starbox = image_stack[:, lbx:ubx, lby:uby]
    h, w, l = starbox.shape

    # get centroid
    Y, X = np.mgrid[:w, :l]
    cx = np.ma.sum(X * starbox, axis=(1, 2)) / np.ma.sum(starbox, axis=(1, 2)) + lbx
    cy = np.ma.sum(Y * starbox, axis=(1, 2)) / np.ma.sum(starbox, axis=(1, 2)) + lby

    # get PSF widths
    X, Y = (
        np.repeat(X[np.newaxis, :, :], h, axis=0),
        np.repeat(Y[np.newaxis, :, :], h, axis=0),
    )
    cx, cy = np.reshape(cx, (h, 1, 1)), np.reshape(cy, (h, 1, 1))
    X2, Y2 = (X + lbx - cx) ** 2, (Y + lby - cy) ** 2
    with np.errstate(invalid="ignore"):
        widx = np.sqrt(
            np.ma.sum(X2 * starbox, axis=(1, 2)) / (np.ma.sum(starbox, axis=(1, 2)))
        )
        widy = np.sqrt(
            np.ma.sum(Y2 * starbox, axis=(1, 2)) / (np.ma.sum(starbox, axis=(1, 2)))
        )

    xo = cx.flatten() / scale
    yo = cy.flatten() / scale
    xw = widx / scale
    yw = widy / scale

    return xo, yo, xw, yw


def A_photometry(
    bg_err,
    cx=15,
    cx_med=15,
    cy=15,
    cy_med=15,
    r=[2.5],
    a=[5],
    b=[5],
    w_r=[5],
    h_r=[5],
    theta=[0],
    scale=1,
    shape="Circular",
    methods=["center", "exact"],
    moveCentroids=[True],
    i=0,
    img_data=None,
):
    """Performs aperture photometry, first by creating the aperture then summing the flux within the aperture.

    Note that this will implicitly use the global variable image_stack (3D array) to allow for parallel computing
        with large (many GB) datasets.

    Args:
        bg_err (1D array): Array of uncertainties on pixel value.
        cx (int/array, optional): x-coordinate(s) of the center of the aperture. Default is 15.
        cy (int/array, optional): y-coordinate(s) of the center of the aperture. Default is 15.
        r (int/array, optional): If shape is 'Circular', the radii to try for aperture photometry
            in units of pixels. Default is just 2.5 pixels.
        a (int/array, optional): If shape is 'Elliptical', the semi-major axes to try for elliptical aperture
            in units of pixels. Default is 5.
        b (int/array, optional): If shape is 'Elliptical', the semi-minor axes to try for elliptical aperture
            in units of pixels. Default is 5.
        w_r (int/array, optional): If shape is 'Rectangular', the full widths to try for rectangular aperture (x-axis).
            Default is 5.
        h_r (int/array, optional): If shape is 'Rectangular', the full heights to try for rectangular aperture (y-axis).
            Default is 5.
        theta (int/array, optional): If shape is 'Elliptical' or 'Rectangular', the rotation angles in radians
            of the semimajor axis from the positive x axis. The rotation angle increases counterclockwise. Default is 0.
        scale (int, optional): If the image is oversampled, scaling factor for centroid and bounds,
            i.e, give centroid in terms of the pixel value of the initial image.
        shape (string, optional): shape is the shape of the aperture. Possible aperture shapes are 'Circular',
            'Elliptical', 'Rectangular'. Default is 'Circular'.
        methods (iterable, optional): The methods used to determine the overlap of the aperture on the pixel grid. Possible
            methods are 'exact', 'subpixel', 'center'. Default is ['center', 'exact'].
        i (int, optional): The current frame number being examined.
        img_data (3D array): The image stack being analyzed if not using the global variable to allow for parallel computing.

    Returns:
        ndarray: Fluxes in moveCentroids/method/size order (a 1D array).
            Missing centroids, invalid background errors, and apertures with
            masked or nonfinite contributing pixels return NaN. No correction
            for missing aperture area is attempted. Formal errors are computed
            by Photutils, but this API historically returns only fluxes.

    """

    stack = image_stack if img_data is None else img_data

    if not isinstance(r, Iterable):
        r = [r]
    if not isinstance(cx, Iterable):
        cx = [cx]
    if not isinstance(cy, Iterable):
        cy = [cy]
    if not isinstance(a, Iterable):
        a = [a]
    if not isinstance(b, Iterable):
        b = [b]
    if not isinstance(w_r, Iterable):
        w_r = [w_r]
    if not isinstance(h_r, Iterable):
        h_r = [h_r]
    if not isinstance(theta, Iterable):
        theta = [theta]
    if isinstance(methods, str) or not isinstance(methods, Iterable):
        methods = [methods]

    sizes = {"Circular": r, "Elliptical": a, "Rectangular": w_r}
    if shape not in sizes:
        raise ValueError(f"Unsupported aperture shape: {shape}")
    nsize = len(sizes[shape])
    nresult = len(moveCentroids) * len(methods) * nsize

    # Photutils expects ordinary ndarrays plus an explicit boolean mask.
    # Keep invalid science values excluded; never impute pixels for photometry.
    frame = np.asarray(np.ma.getdata(stack[i]), dtype=float)
    mask = np.ma.getmaskarray(stack[i]) | ~np.isfinite(frame)
    background = np.ma.asarray(bg_err)[i]
    if (
        mask.all()
        or np.ma.is_masked(background)
        or not np.isfinite(background)
        or background < 0
    ):
        return np.full(nresult, np.nan)
    background_error = np.full(frame.shape, background, dtype=float)
    with np.errstate(invalid="ignore", over="ignore"):
        data_error = calc_total_error(frame, background_error, effective_gain=1)
    mask |= ~np.isfinite(data_error)

    results = []

    for moveCentroid in moveCentroids:
        # Set up aperture(s)
        apertures = []
        if not moveCentroid:
            xpos, ypos = cx_med, cy_med
        else:
            xpos = cx[0 if len(cx) == 1 else i]
            ypos = cy[0 if len(cy) == 1 else i]
        position = (
            np.ma.concatenate((np.ma.atleast_1d(xpos), np.ma.atleast_1d(ypos)))
            .astype(float)
            .filled(np.nan)
            * scale
        )
        if not np.all(np.isfinite(position)):
            results.extend([np.nan] * (len(methods) * nsize))
            continue
        if shape == "Circular":
            for j in range(len(r)):
                apertures.append(CircularAperture(position, r=r[j] * scale))
        elif shape == "Elliptical":
            for j in range(len(a)):
                apertures.append(
                    EllipticalAperture(
                        position, a=a[j] * scale, b=b[j] * scale, theta=theta[j]
                    )
                )
        elif shape == "Rectangular":
            for j in range(len(w_r)):
                apertures.append(
                    RectangularAperture(
                        position, w=w_r[j] * scale, h=h_r[j] * scale, theta=theta[j]
                    )
                )

        for method in methods:
            phot_table = aperture_photometry(
                frame, apertures, error=data_error, mask=mask, method=method
            )
            for j, aperture in enumerate(apertures):
                # A masked sum would underestimate the target's total flux.
                # Use exactly the same pixel weights as Photutils, including
                # fractional edge pixels, to reject incomplete apertures.
                weights = aperture.to_mask(method=method)
                large, small = weights.get_overlap_slices(frame.shape)
                incomplete = large is None
                if not incomplete:
                    contributing = weights.data > 0
                    incomplete = (
                        np.any(mask[large] & contributing[small])
                        or np.count_nonzero(contributing[small])
                        != np.count_nonzero(contributing)
                        or not np.any(contributing[small])
                    )
                value = phot_table[f"aperture_sum_{j}"][0]
                results.append(
                    np.nan
                    if incomplete or np.ma.is_masked(value) or not np.isfinite(value)
                    else float(value)
                )

    return np.array(results)


def compare_RMS(
    Run_list,
    fluxes,
    r,
    time,
    highpassWidth,
    basepath,
    planet,
    channel,
    ignoreFrames,
    addStack,
    save=True,
    onlyBest=False,
    showPlots=False,
    savePlots=True,
):

    RMS = np.full(len(Run_list), np.inf)

    for i, foldername in enumerate(Run_list):
        flux = np.ma.masked_invalid(fluxes[:, i])
        if flux.count() < 2:
            continue
        smooth = highpassflist(flux, highpassWidth)
        smoothed = (flux - smooth) + np.ma.median(flux)
        #         RMS[i] = np.ma.sqrt(np.ma.median((flux-smooth)**2.))/np.ma.median(smoothed)
        residual = np.ma.masked_invalid(flux - smooth)
        normalization = np.ma.median(smoothed)
        if residual.count() >= 2 and np.isfinite(normalization) and normalization > 0:
            RMS[i] = np.ma.std(residual) / normalization

    exact_moving = np.array(
        [
            "exact" in Run_list[i].lower() and "moving" in Run_list[i].lower()
            for i in range(len(Run_list))
        ],
        dtype=bool,
    )
    soft_moving = np.array(
        [
            "soft" in Run_list[i].lower() and "moving" in Run_list[i].lower()
            for i in range(len(Run_list))
        ],
        dtype=bool,
    )
    hard_moving = np.array(
        [
            "hard" in Run_list[i].lower() and "moving" in Run_list[i].lower()
            for i in range(len(Run_list))
        ],
        dtype=bool,
    )

    exact = np.array(
        [
            "exact" in Run_list[i].lower() and "moving" not in Run_list[i].lower()
            for i in range(len(Run_list))
        ],
        dtype=bool,
    )
    soft = np.array(
        [
            "soft" in Run_list[i].lower() and "moving" not in Run_list[i].lower()
            for i in range(len(Run_list))
        ],
        dtype=bool,
    )
    hard = np.array(
        [
            "hard" in Run_list[i].lower() and "moving" not in Run_list[i].lower()
            for i in range(len(Run_list))
        ],
        dtype=bool,
    )

    if showPlots or savePlots:
        plt.figure(figsize=(10, 4))

        if np.any(exact_moving):
            plt.plot(
                r[exact_moving],
                RMS[exact_moving] * 1e6,
                "o-",
                label="Circle: Exact Edge, Moving",
            )
        if np.any(soft_moving):
            plt.plot(
                r[soft_moving],
                RMS[soft_moving] * 1e6,
                "o-",
                label="Circle: Soft Edge, Moving",
            )
        if np.any(hard_moving):
            plt.plot(
                r[hard_moving],
                RMS[hard_moving] * 1e6,
                "o-",
                label="Circle: Hard Edge, Moving",
            )

        if np.any(exact):
            plt.plot(r[exact], RMS[exact] * 1e6, "o-", label="Circle: Exact Edge")
        if np.any(soft):
            plt.plot(r[soft], RMS[soft] * 1e6, "o-", label="Circle: Soft Edge")
        if np.any(hard):
            plt.plot(r[hard], RMS[hard] * 1e6, "o-", label="Circle: Hard Edge")

        plt.xlabel("Aperture Radius")
        plt.ylabel("RMS Scatter (ppm)")
        plt.legend(loc="best")

        if savePlots:
            figpath = (
                basepath + planet + "/analysis/photometryComparison/" + channel + "/"
            )
            if addStack:
                figpath += "addedStack/"
            else:
                figpath += "addedBlank/"
            if not os.path.exists(figpath):
                os.makedirs(figpath)

            if ignoreFrames != []:
                figpath += "ignore/"
            else:
                figpath += "noIgnore/"
            if not os.path.exists(figpath):
                os.makedirs(figpath)

            if channel == "ch2":
                fname = figpath + "4um"
            else:
                fname = figpath + "3um"
            fname += "_Photometry_Comparison.pdf"

            plt.savefig(fname)
        if showPlots:
            plt.show()
        plt.close()

    if save and np.any(np.isfinite(RMS)):
        if np.any(exact_moving):
            print(
                "\tExact Moving - Best RMS (ppm):",
                np.round(np.ma.min(RMS[exact_moving]) * 1e6, decimals=2),
            )
            print(
                "\tExact Moving - Best Aperture Radius:",
                r[exact_moving][
                    np.where(RMS[exact_moving] == np.ma.min(RMS[exact_moving]))[0][0]
                ],
            )
            print()
        if np.any(soft_moving):
            print(
                "\tSoft Moving - Best RMS (ppm):",
                np.round(np.ma.min(RMS[soft_moving]) * 1e6, decimals=2),
            )
            print(
                "\tSoft Moving - Best Aperture Radius:",
                r[soft_moving][
                    np.where(RMS[soft_moving] == np.ma.min(RMS[soft_moving]))[0][0]
                ],
            )
            print()
        if np.any(hard_moving):
            print(
                "\tHard Moving - Best RMS (ppm):",
                np.round(np.ma.min(RMS[hard_moving]) * 1e6, decimals=2),
            )
            print(
                "\tHard Moving - Best Aperture Radius:",
                r[hard_moving][
                    np.where(RMS[hard_moving] == np.ma.min(RMS[hard_moving]))[0][0]
                ],
            )
            print()
        if np.any(exact):
            print(
                "\tExact - Best RMS (ppm):",
                np.round(np.ma.min(RMS[exact]) * 1e6, decimals=2),
            )
            print(
                "\tExact - Best Aperture Radius:",
                r[exact][np.where(RMS[exact] == np.ma.min(RMS[exact]))[0][0]],
            )
            print()
        if np.any(soft):
            print(
                "\tSoft - Best RMS (ppm):",
                np.round(np.ma.min(RMS[soft]) * 1e6, decimals=2),
            )
            print(
                "\tSoft - Best Aperture Radius:",
                r[soft][np.where(RMS[soft] == np.ma.min(RMS[soft]))[0][0]],
            )
            print()
        if np.any(hard):
            print(
                "\tHard - Best RMS (ppm):",
                np.round(np.ma.min(RMS[hard]) * 1e6, decimals=2),
            )
            print(
                "\tHard - Best Aperture Radius:",
                r[hard][np.where(RMS[hard] == np.ma.min(RMS[hard]))[0][0]],
            )

        bestPhOption = Run_list[np.ma.argmin(RMS)]
        print("Best photometry of this batch:", bestPhOption)

        with open(
            basepath + planet + "/analysis/" + channel + "/bestPhOption.txt", "a"
        ) as file:
            file.write(bestPhOption + "\n")
            file.write("IgnoreFrames = " + str(ignoreFrames)[1:-1] + "\n")
            file.write(str(np.round(np.ma.min(RMS) * 1e6, 1)) + "\n\n")

    return RMS


def bin_all_data(
    highpassWidth,
    flux,
    binned_time,
    binned_time_std,
    binned_xo,
    binned_xo_std,
    binned_yo,
    binned_yo_std,
    binned_xw,
    binned_xw_std,
    binned_yw,
    binned_yw_std,
    binned_bg,
    binned_bg_std,
    binned_npp,
    binned_npp_std,
    bin_size,
):

    binned_flux, binned_flux_std = bin_array(flux, bin_size)

    # Do a rolling median based sigma clipping to remove bad data
    binned_flux = clip_data(
        binned_flux,
        highpassWidth,
        sigma1=10,
        sigma2=5,
        maxiters=3,
        replace_masked=False,
    )
    binned_flux_std[np.ma.getmaskarray(binned_flux)] = np.nan

    return np.ma.column_stack(
        (
            binned_flux,
            binned_flux_std,
            binned_time,
            binned_time_std,
            binned_xo,
            binned_xo_std,
            binned_yo,
            binned_yo_std,
            binned_xw,
            binned_xw_std,
            binned_yw,
            binned_yw_std,
            binned_bg,
            binned_bg_std,
            binned_npp,
            binned_npp_std,
        )
    ).filled(np.nan)


def get_lightcurve(
    basepath,
    AOR_snip,
    channel,
    planet,
    save=True,
    onlyBest=True,
    highpassWidth=5 * 64,
    bin_data=True,
    bin_size=64,
    showPlots=False,
    savePlots=True,
    oversamp=False,
    scale=2,
    saveoversamp=True,
    reuse_oversamp=True,
    r=[2.4],
    edges=["Exact"],
    addStack=False,
    ignoreFrames=None,
    maskStars=None,
    moveCentroids=[True],
    ncpu=4,
    image_stack_input=None,
    bg=None,
    bg_err=None,
    time=None,
    science_product="bcd",
):
    """Given a directory, looks for data (bcd.fits files), opens them and performs photometry.

    Args:
        science_product (str): Select 'bcd' (default) or 'cbcd' when loading images.
        datapath (string): Directory where the spitzer data is stored.
        savepath (string): Directory the outputs will be saved.
        AORsnip (string):  Common first characters of data directory eg. 'r579'
        channel (string): Channel used for the observation eg. 'ch1' for channel 1
        shape (string, optional): The aperture shape to try. Possible aperture shapes are 'Circular',
            'Elliptical', 'Rectangular'. Default is 'Circular'.
        edges (iterable, optional): The aperture edges to try. Options are 'hard',
            'soft', and 'exact' which correspond to the 'center', 'subpixel',
            and 'exact' methods in astropy. Default is just ['hard'].
        save (bool, optional): True if you want to save the outputs. Default is True.
        save_full (string, optional): Filename of the full unbinned output data. Default is '/ch2_datacube_full_AORs579.dat'.
        bin_data (bool, optional): True you want to get binned data. Default is True.
        bin_size (int, optional): If bin_data is True, the size of the bins. Default is 64.
        save_bin (string, optional): Filename of the full binned output data. Default is '/ch2_datacube_binned_AORs579.dat'.
        plot (bool, optional): True if you want to plot the time resolved lightcurve. Default is True.
        plot_name (string, optional): If plot and save is True, the filename of the plot to be saved as. Default is True.
        oversamp (bool, optional): True if you want to oversample the image by a factor of 2. Default is False.
            Currently nonfunctional pending separate scientific validation; leave False.
        scale (int, optional): Sampling factor used only when oversamp=True.
            With oversamp=False, native pixel coordinates are always used,
            even if scale is supplied. Default is 2 for oversampled data.
        image_stack_input (3D array, optional): Already prepared, background-subtracted
            frames. No resampling or background subtraction is applied to this
            stack. For a preloaded oversampled stack, set oversamp=True and
            supply its sampling factor as scale.
        save_oversamp (bool, optional): True if you want to save oversampled images. Default is True.
        reuse_oversamp (bool, optional): True if you want to reuse oversampled images that were previously saved.
            Default is False.
        planet (string, optional): The name of the planet. Default is CoRoT-2b.
        rs (iterable, optional): The radii to try for aperture photometry in units of pixels. Default is just 2.5 pixels.
        ignoreFrames (list, optional) A list of frames to be masked when performing aperature photometry (e.g. first
            frame to remove first-frame systematic).
        maskStars (list, optional): An array-like object where each element is an array-like object with the RA and DEC
            coordinates of a nearby star which should be masked out when computing background subtraction.
        moveCentroids (iterable, optional): True centers each aperture on its
            frame's flux-weighted centroid. Missing/clipped centroids produce
            NaN fluxes. False uses the median valid centroid. Default is [True].
        rerun_photometry (bool, optional): Whether to overwrite old photometry if it exists. Default is False.
        ncpu (int, optional): The number of aperture radii to try at the same time with multiprocessing. Default is 4.

    Raises:
        Error: If Photometry method is not supported/recognized by this pipeline.

    Notes:
        Rejected fluxes are saved/returned as NaN, without median replacement.
        Bins use only surviving flux samples (and their corresponding metadata);
        empty bins remain NaN. The saved ``Flux std`` is within-bin scatter,
        not a propagated formal uncertainty, and is zero for a single sample.

    """

    # Currently only circular apertures are supported!
    shape = "Circular"
    # Both loaded and supplied native frames need native centroid bounds,
    # positions, and radii, regardless of the oversampling-only scale default.
    if not oversamp:
        scale = 1

    if ignoreFrames is None:
        ignoreFrames = []

    if maskStars is None:
        maskStars = []

    if basepath[-1] != "/":
        basepath += "/"

    savepath = basepath + planet + "/analysis/" + channel + "/"
    if addStack:
        savepath += "addedStack/"
    else:
        savepath += "addedBlank/"
    if ignoreFrames != []:
        savepath += "ignore/"
    else:
        savepath += "noIgnore/"

    # prepare filenames for saved data
    save_full = channel + "_datacube_full_AORs" + AOR_snip[1:] + ".dat"
    save_bin = channel + "_datacube_binned_AORs" + AOR_snip[1:] + ".dat"

    if not isinstance(r, Iterable):
        r = [r]
    if not isinstance(edges, Iterable):
        edges = [edges]
    if not isinstance(moveCentroids, Iterable):
        moveCentroids = [moveCentroids]

    if shape != "Circular" and shape != "Elliptical" and shape != "Rectangular":
        print(
            'Warning: No such aperture shape "' + shape + '".',
            "Using Circular aperture instead.",
        )
        shape = "Circular"

    methods = []
    for edge_tmp in edges:
        edge_tmp = edge_tmp.lower()
        if edge_tmp == "hard" or edge_tmp == "center" or edge_tmp == "centre":
            methods.append("center")
        elif edge_tmp == "soft" or edge_tmp == "subpixel":
            methods.append("subpixel")
        elif edge_tmp == "exact":
            methods.append("exact")
        else:
            print(
                'Warning: No such method "' + edge_tmp + '".',
                "Using hard edged aperture instead.",
            )
            methods.append("center")

    # Access the global variable
    global image_stack
    if image_stack_input is None:
        # Prepare all of the images
        image_stack, bg, bg_err, time = prepare_images(
            basepath,
            planet,
            channel,
            AOR_snip,
            ignoreFrames,
            oversamp,
            scale,
            reuse_oversamp,
            saveoversamp,
            addStack,
            maskStars,
            ncpu,
            science_product=science_product,
        )
    else:
        image_stack = image_stack_input

    #     image_stack, bg, bg_err, time = prepare_images(basepath, planet, channel, AOR_snip, ignoreFrames,
    #                                                    oversamp, scale, reuse_oversamp, saveoversamp,
    #                                                    addStack, maskStars, ncpu)

    clip = partial(
        clip_data,
        highpassWidth=highpassWidth,
        sigma1=10,
        sigma2=5,
        maxiters=3,
        replace_masked=False,
    )
    bg = clip(bg)
    bg_err = clip(bg_err)

    # get centroids & PSF width
    print("\tGetting centroids... ", end="", flush=True)
    xo, yo, xw, yw = centroid_FWM(image_stack, highpassWidth=5 * 64, scale=scale)

    # Compute noise pixel parameter for each frame
    print("Getting noise pixel parameter... ", end="", flush=True)
    npp = noisepixparam(image_stack)

    npp = clip(npp)
    xo = clip(xo)
    xw = clip(xw)
    yo = clip(yo)
    yw = clip(yw)

    # perform aperture photometry
    print("Starting photometry!", flush=True)

    # Resorting to a bit of hackery to get tqdm to work with multiprocessing
    global pbar
    global results
    global func
    N = image_stack.shape[0]
    pbar = tqdm(total=N)
    results = [None] * N  # result list of correct size
    func = partial(
        A_photometry,
        bg_err,
        xo,
        np.ma.median(xo),
        yo,
        np.ma.median(yo),
        r,
        [],
        [],
        [],
        [],
        [],
        scale,
        shape,
        methods,
        moveCentroids,
    )

    pool = _get_pool(ncpu)
    try:
        first = pool.apply_async(wrapMyFunc, args=(0,), callback=update)
        first.get()

        jobs = [
            pool.apply_async(wrapMyFunc, args=(i,), callback=update)
            for i in range(1, N)
        ]
        pool.close()

        for job in jobs:
            job.get()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    finally:
        pbar.close()
    sys.stderr.flush()

    # Free up some RAM
    image_stack = None

    fluxes = np.ma.masked_invalid(np.array(results))
    # Free up some RAM
    results = None

    # removing outrageously bad flux outliers for each technique
    print("\tSigma clipping fluxes... ", end="", flush=True)

    # Sometimes we get points that are so close to zero that they break the code (even the sigma clipping)
    for i in range(fluxes.shape[1]):
        flux = fluxes[:, i]
        flux = np.ma.masked_where(flux < 0.1 * np.ma.median(flux), flux)
        fluxes[:, i] = clip(flux)

    # Make a folder name for each method
    techniques = []
    all_edges = []
    all_rs = []
    all_moveCentroids = []
    for moveCentroid in moveCentroids:
        for edge in edges:
            for r_tmp in r:
                if channel == "ch1":
                    folder = "3um"
                else:
                    folder = "4um"
                folder += edge + shape + "_".join(str(np.round(r_tmp, 2)).split("."))
                if moveCentroid:
                    folder += "_movingCentroid"
                techniques.append(savepath + folder)
                all_moveCentroids.append(moveCentroid)
                all_edges.append(edge)
                all_rs.append(r_tmp)

    BIN_datas = []
    if bin_data:
        RMS_fluxes = np.zeros((int(np.ceil(fluxes.shape[0] / bin_size)), 0))
        RMS_times = []
        highpassWidth_tmp = highpassWidth / bin_size

        print("Binning... ", end="", flush=True)
        for i in range(fluxes.shape[1]):
            flux = fluxes[:, i]
            # Bin metadata from the same surviving frames as this flux series.
            binned = [
                bin_array(np.ma.masked_where(np.ma.getmaskarray(flux), data), bin_size)
                for data in (time, xo, yo, xw, yw, bg, npp)
            ]
            (
                (binned_time, binned_time_std),
                (binned_xo, binned_xo_std),
                (binned_yo, binned_yo_std),
                (binned_xw, binned_xw_std),
                (binned_yw, binned_yw_std),
                (binned_bg, binned_bg_std),
                (binned_npp, binned_npp_std),
            ) = binned
            BIN_data = bin_all_data(
                highpassWidth_tmp,
                flux,
                binned_time,
                binned_time_std,
                binned_xo,
                binned_xo_std,
                binned_yo,
                binned_yo_std,
                binned_xw,
                binned_xw_std,
                binned_yw,
                binned_yw_std,
                binned_bg,
                binned_bg_std,
                binned_npp,
                binned_npp_std,
                bin_size,
            )

            RMS_fluxes = np.append(RMS_fluxes, BIN_data[:, 0, np.newaxis], axis=1)
            RMS_times = BIN_data[:, 2]
            BIN_datas.append(BIN_data)
    else:
        RMS_fluxes = fluxes
        RMS_times = time
        highpassWidth_tmp = highpassWidth

    # Choose the best photometry method, save diagnostic plot(s)
    print("Choosing best photometry...", flush=True)
    RMSs = compare_RMS(
        techniques,
        RMS_fluxes,
        np.array(all_rs),
        RMS_times,
        highpassWidth_tmp,
        basepath,
        planet,
        channel,
        ignoreFrames,
        addStack,
        save,
        onlyBest,
        showPlots,
        savePlots,
    )
    if not np.any(np.isfinite(RMSs)):
        print("No photometry method has enough valid measurements to rank.")

    if onlyBest and np.any(np.isfinite(RMSs)):
        # Keep these as arrays so they can be indexed lated
        # If the user only want's to save the best results, discard the rest
        fluxes = fluxes[:, np.argmin(RMSs) : np.argmin(RMSs) + 1]
        BIN_datas = BIN_datas[np.argmin(RMSs) : np.argmin(RMSs) + 1]
        all_moveCentroids = all_moveCentroids[np.argmin(RMSs) : np.argmin(RMSs) + 1]
        all_edges = all_edges[np.argmin(RMSs) : np.argmin(RMSs) + 1]
        all_rs = all_rs[np.argmin(RMSs) : np.argmin(RMSs) + 1]

    # Bin, save, and/or plot each of the methods depending on what was requested
    FULL_datas = []
    for i in range(fluxes.shape[1]):
        flux = fluxes[:, i]
        FULL_data = np.ma.column_stack((flux, time, xo, yo, xw, yw, bg, npp)).filled(
            np.nan
        )

        if save or savePlots:
            print("\tSaving... ", end="", flush=True)
            # create save folder
            if channel == "ch1":
                folder = "3um"
            else:
                folder = "4um"
            folder += (
                all_edges[i] + shape + "_".join(str(np.round(all_rs[i], 2)).split("."))
            )
            if all_moveCentroids[i]:
                folder += "_movingCentroid"
            folder += "/"

            savepath_tmp = savepath + folder

            savepath_tmp = create_folder(savepath_tmp, True, True)

        # Plot the photometry if requested
        if savePlots or showPlots:
            if bin_data:
                plotx = BIN_datas[i][:, 2]
                ploty0 = BIN_datas[i][:, 0]
                ploty1 = BIN_datas[i][:, 4]
                ploty2 = BIN_datas[i][:, 6]
                ploty3 = BIN_datas[i][:, 8]
                ploty4 = BIN_datas[i][:, 10]
            else:
                plotx = FULL_data[:, 1]
                ploty0 = FULL_data[:, 0]
                ploty1 = FULL_data[:, 2]
                ploty2 = FULL_data[:, 3]
                ploty3 = FULL_data[:, 4]
                ploty4 = FULL_data[:, 5]

            fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(15, 15))

            axes[0].set_title(planet, fontsize="x-large")
            axes[0].plot(plotx, ploty0, "k+")
            axes[0].set_ylabel("Stellar Flux (electrons)")

            axes[1].plot(plotx, ploty1, "k+")
            axes[1].set_ylabel("$x_0$")

            axes[2].plot(plotx, ploty2, "k+")
            axes[2].set_ylabel("$y_0$")

            axes[3].plot(plotx, ploty3, "k+")
            axes[3].set_ylabel("$x_w$")

            axes[4].plot(plotx, ploty4, "k+")
            axes[4].set_ylabel("$y_w$")

            fig.subplots_adjust(hspace=0)
            axes[4].set_xlabel("Time (BMJD))")
            axes[4].ticklabel_format(useOffset=False)

            if savePlots:
                # Save the plot if requested
                pathplot = savepath_tmp + "Lightcurve.pdf"
                fig.savefig(pathplot)
            if showPlots:
                plt.show()
            plt.close()

        # Save the data if requested
        if save:
            FULL_head = (
                "Flux, Time, x-centroid, y-centroid, x-PSF width, y-PSF width, bg flux"
            )
            FULL_head += ", Noise Pixel Parameter"
            pathFULL = savepath_tmp + save_full
            np.savetxt(pathFULL, FULL_data, header=FULL_head)
            if bin_data:
                BIN_head = "Flux, Flux std, Time, Time std, x-centroid, x-centroid std, y-centroid, y-centroid std"
                BIN_head += ", x-PSF width, x-PSF width std, y-PSF width, y-PSF width std, bg flux, bg flux std"
                BIN_head += ", Noise Pixel Parameter, Noise Pixel Parameter std"
                pathBIN = savepath_tmp + save_bin
                np.savetxt(pathBIN, BIN_datas[i], header=BIN_head)
        else:
            FULL_datas.append(FULL_data)

    print("Done.", flush=True)

    if save:
        # We are actually running the photometry
        return
    elif bin_data:
        # We are running frame diagnostics and should return our results
        return FULL_datas, BIN_datas
    else:
        # We are running frame diagnostics and should return our results
        return FULL_datas


import unittest


class TestAperturehotometryMethods(unittest.TestCase):
    # Test that centroiding gives the expected values and doesn't swap x and y
    def test_FWM_centroiding(self):
        fake_images = np.zeros((4, 32, 32))
        for i in range(fake_images.shape[0]):
            fake_images[i, 14 + i, 15] = 2
        xo, yo, _, _ = centroid_FWM(fake_images)
        self.assertTrue(np.all(xo == np.ones_like(xo) * 15.0))
        self.assertTrue(np.all(yo == np.arange(14, 18)))

    # Test that circular aperture photometry properly follows the input centroids and gives the expected values
    def test_circularAperture(self):
        image_stack = np.zeros((4, 32, 32))

        for i in range(image_stack.shape[0]):
            image_stack[i, 14 + i, 15] = 2
        xo = np.ones(image_stack.shape[0]) * 15
        yo = np.arange(14, 18)
        with _get_pool(1) as pool:
            jobs = [
                pool.apply_async(
                    A_photometry,
                    kwds={
                        "bg_err": np.zeros_like(xo),
                        "cx": xo,
                        "cy": yo,
                        "r": [1.0, 2.0, 3.0],
                        "methods": ["center"],
                        "i": i,
                        "img_data": image_stack,
                    },
                )
                for i in range(image_stack.shape[0])
            ]
            flux = np.array([job.get() for job in jobs])

        self.assertTrue(np.all(flux == np.ones_like(flux) * 2.0))


if __name__ == "__main__":
    unittest.main()
