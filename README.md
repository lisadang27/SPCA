# SPCA

SPCA is an open-source, modular, and automated pipeline for Spitzer Phase Curve Analyses.

The full API can be found at [https://spca.readthedocs.io](https://spca.readthedocs.io).

## Installation Instructions

To install SPCA, run the following in a terminal:

```
        git clone git@github.com:lisadang27/SPCA.git
        cd SPCA
        pip install .
```

Please note however that SPCA is in a state of alpha testing and is still under development. Frequent changes are expected over the upcoming few months as we finalize some aspects and encorporate PLD analyses as well as PSF fitting and nearby companion removal using PSF subtraction.


## Package Usage

1. To use SPCA, you must first download your data from the Spitzer Heritage Archive: https://sha.ipac.caltech.edu/applications/Spitzer/SHA/. Place the downloaded zip files in a directory with the same name as the planet (exluding spaces).

Most of the following commands have an .ipynb ending and .py ending option available, where the .py version is optimized for analyzing many data sets and the .ipynb file is optimized for viewing the analysis of a single data set. Each file has a portion at the top where you can set parameters which will determine the techniques.

2. Next, use the Make\_Directory\_Structure file to extract the data and setup the required directories.

3. Then use the Everything\_Photometry file to perform a suite of different photometries on your data and have the code automatically select the best photometry (selecting the photometry that gives the lowest scatter after smoothing the raw flux by a boxcar filter of a width provided by the user).

4. Then use the QuickLook file to ensure that you have looked at the raw data and to determine whether you want to remove the first AOR (in case it is a short AOR before PCRS peak-up was used). By looking at the raw data, you can gain some insight into how successful different decorrelation models might be.

5. Then decorrelate the data using the Poly-BLISS-GP file. Most parameters here are explained with a nearby comment. One key parameter though is the "mode" which sets the decorrelation method used. Modes that contain "Poly#" use a 2D poly of order #, modes that contain "BLISS" use BiLinearly-Interpolated Subpixel Sensitivity mapping, modes that contain "GP" use a Gaussian Process using x and y centroid positions as covariates, and modes that contain "PLD" use Pixel Level Decorrelation. Each mode is then followed by an underscore and either "v1" or "v2" indicating the use of either a 1st or 2nd order sinusoidal model for the phase variations. If "ellipse" is present, phase variations due to the elliptical shape of the planet are modelled. Any other text can be added to the mode keyword for your own convenience (e.g. "Poly2\_v1\_run2").

6. Finally, some tables containing a selection of the fitted parameters from each model run can be made using the MakeTables file. These tables will also highlight the best decorrelation method for each analysis, determined using delta-BIC.

## Contributions

[Lisa Dang](https://github.com/lisadang27) contributed the initial code and idea for SPCA, and wrote much of the photometry and fitting routines, as well as the polynomial and PLD decorrelation methods.

Both authors worked extensively to debug the code and simplify the user experience.

[Taylor James Bell](https://github.com/taylorbell57) further generalized and streamlined SPCA, allowing it to be run quickly and easily for any given planet with minimal effort. Taylor also contributed much of the documentation, and the GP decorrelation method.

## Acknowledgements

We thank Joel Schwartz for his aid in writing out BLISS decorrelation method. We also thank Dylan Keating for alpha testing SPCA.

## License & Attribution

Copyright © 2019 Lisa Dang & Taylor James Bell.

SPCA is free software made available under the GPL3 License. For details
see the [LICENSE](https://github.com/lisadang27/SPCA/blob/master/LICENSE).

If you make use of SPCA in your work, please cite the Dang et al. (2018) paper that was the first to use this pipeline
[arXiv](https://arxiv.org/abs/1801.06548),
[ADS](https://ui.adsabs.harvard.edu/abs/2018NatAs...2..220D),
[BibTeX](https://ui.adsabs.harvard.edu/abs/2018NatAs...2..220D/exportcitation>).
