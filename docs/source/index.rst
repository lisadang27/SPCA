.. SPCA documentation master file, created by
   sphinx-quickstart on Tue Dec 17 21:20:49 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SPCA's documentation!
================================

SPCA is an open-source, modular, and automated pipeline for Spitzer Phase Curve Analyses.

Installation Instructions
=========================

To install SPCA, run the following in a terminal:

.. code-block:: bash

        git clone git@github.com:lisadang27/SPCA.git
        cd SPCA
        pip install .

Please note however that SPCA is in a state of alpha testing and is still under development. Frequent changes are expected over the upcoming few months as we finalize some aspects like PSF fitting and nearby companion removal using PSF subtraction.

Update Log
==========

Version 0.2
-----------
As of version 0.2, we have made some important changes and some bug fixes. These include:
- Renamed decorrelation and fitting file to Decorrelation (.py and .ipynb)
- Full integration of PLD and PLDAper models
- Reduced the amount of static code that the user sees and placed this in separate files instead.
- Changed how photometry is saved. This will not be noticeable unless you want to use PLDAper models which cannot be run with the old photometry.
- Also changed the units in which the photometry is saved so that it is easy to compute the photon noise limit - a calculation that was previously done incorrectly.
- The emcee fitting routine sometimes freezing still seems to be an issue as of v0.2. This seems to be caused by the combination of a large dataset and either a large model or a poorly initialized model. Previous attempts at forcing a timeout have failed without directly editing emcee or multiprocessing code. If this occurs to you and you need to analyze your data soon, I'd recommend removing the few lines that use multiprocessing - contact us if you have a hard time doing this. We are looking into different samplers (such as PyMC3) to resolve this issue.

Package Usage
=============

1. To use SPCA, you must first download your data from the Spitzer Heritage Archive: https://sha.ipac.caltech.edu/applications/Spitzer/SHA/. Place the downloaded zip files in a directory with the same name as the planet (exluding spaces).

Most of the following commands have an .ipynb ending and .py ending option available, where the .py version is optimized for analyzing many data sets and the .ipynb file is optimized for viewing the analysis of a single data set. Each file has a portion at the top where you can set parameters which will determine the techniques.

2. Next, use the Make_Directory_Structure file to extract the data and setup the required directories.

3. Then use the Everything_Photometry file to perform a suite of different photometries on your data and have the code automatically select the best photometry (selecting the photometry that gives the lowest scatter after smoothing the raw flux by a boxcar filter of a width provided by the user).

4. Then use the QuickLook file to ensure that you have looked at the raw data and to determine whether you want to remove the first AOR (in case it is a short AOR before PCRS peak-up was used). By looking at the raw data, you can gain some insight into how successful different decorrelation models might be.

5. Then decorrelate the data using the Decorrelation file. Most parameters here are explained with a nearby comment. One key parameter though is the "mode" which sets the decorrelation method used. Modes that contain "Poly#" use a 2-dimentionsal polynomial of order #, modes that contain "BLISS" use BiLinearly-Interpolated Subpixel Sensitivity mapping, modes that contain "GP" use a Gaussian Process using x and y centroid positions as covariates, and modes that contain "PLD#_$x$" use Pixel Level Decorrelation of order # (1 or 2) and use a pixel stamp size of $ by $ (3x3 or 5x5). PLD performed using aperture photometry (the recommended PLD technique) can be accessed using "PLDAper#_$x$". Each mode is then followed by an underscore and either "v1" or "v2" indicating the use of either a 1st or 2nd order sinusoidal model for the phase variations. If "ellipse" is present in the mode string, phase variations due to the elliptical shape of the planet are modelled. Any other text can be added to the mode keyword for your own convenience (e.g. "Poly2_v1_run2" or "PLDAper2_5x5_v1_testingFit").

6. Finally, some tables containing a selection of the fitted parameters from each model run can be made using the MakeTables file. These tables will also highlight the best decorrelation method for each analysis, determined using delta-BIC (selecting your model is actually quite a challenging and complicated step, and user discretion is absolutely recommended).


.. toctree::
   :maxdepth: 2
   :caption: API Table of Contents:

   SPCA


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. * :ref:`search`

Contributions
=============

`Lisa Dang <https://github.com/lisadang27>`_ contributed the initial code and idea for SPCA, and wrote much of the photometry and fitting routines, as well as the polynomial and PLD decorrelation methods.

Both authors worked extensively to debug the code and simplify the user experience.

`Taylor James Bell <https://github.com/taylorbell57>`_ further generalized and streamlined SPCA, allowing it to be run quickly and easily for any given planet with minimal effort. Taylor also contributed much of the documentation, and the GP decorrelation method.

Acknowledgements
================

We thank Joel Schwartz for his aid in writing out BLISS decorrelation method. We also thank Dylan Keating for alpha testing SPCA. We thank Antoine Darveau-Bernier for writing code to parse through the Exoplanet Archive data and select the best constrained value for each parameter in the database (code can be found `here <https://github.com/AntoineDarveau/masterfile>`_).

License & Attribution
=====================

Copyright © 2018-2020 Lisa Dang & Taylor James Bell.

SPCA is free software made available under the GPL3 License. For details
see the `LICENSE <https://github.com/lisadang27/SPCA/blob/master/LICENSE>`_.

If you make use of SPCA in your work, please cite the Dang et al. (2018) paper that was the first to use this pipeline
(`arXiv <https://arxiv.org/abs/1801.06548>`_,
`ADS <https://ui.adsabs.harvard.edu/abs/2018NatAs...2..220D>`_,
`BibTeX <https://ui.adsabs.harvard.edu/abs/2018NatAs...2..220D/exportcitation>`_).
