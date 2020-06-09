import numpy as np

# SPCA libraries
from . import make_plots
from . import helpers
from . import astro_models

def lh_axes_binning(xo, yo, nBin, nData):
    """Create knots using the fitted centroids and an input number of knots.

    Args:
        xo (ndarray): The x-centroids for each data point.
        yo (ndarray): The y-centroids for each data point.
        nBin (int): The number of knots you want along each axis.
        nData (int): The number of data points.

    Returns:
        tuple: low_bnd_x (ndarray; the index of the knot to the left of each centroid),
            up_bnd_x (ndarray; the index of the knot to the right of each centroid),
            low_bnd_y (ndarray; the index of the knot below each centroid),
            up_bnd_y (ndarray; the index of the knot above each centroid),
            knots_x (ndarray; the detector coordinate for the x center of each knot),
            knots_y (ndarray; the detector coordinate for the y center of each knot),
            knotNdata (ndarray; the number of data assoicated with each knot),
            x_edg (ndarray; the x-coordinates for the edges of the knots),
            y_edg (ndarray; the y-coordinates for the edges of the knots),
            knots_x_mesh (ndarray; the 2D array of the x-position of each knot),
            knots_y_mesh (ndarray; the 2D array of the y-position of each knot).
    
    """
    
    knotNdata, x_edg, y_edg = np.histogram2d(xo, yo, bins=nBin)
    
    # get the coordinate of the centre of the knots
    knots_x = x_edg[1:] - 0.5*(x_edg[1:] - x_edg[0:-1])
    knots_y = y_edg[1:] - 0.5*(y_edg[1:] - y_edg[0:-1])
    
    # get knots boundaries
    low_bnd_x, up_bnd_x = lh_knot_ass(knots_x, xo, nBin, nData)
    low_bnd_y, up_bnd_y = lh_knot_ass(knots_y, yo, nBin, nData)
    
    knots_x_mesh, knots_y_mesh = np.meshgrid(knots_x, knots_y)
    
    return (low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y, knots_x, knots_y,
            knotNdata, x_edg, y_edg, knots_x_mesh, knots_y_mesh)


def lh_knot_ass(knots_pos, cents, nBin, nData):
    """Find the two knots adjacted to each knot to later be used with linear interpolation.

    Args:
        knots_pos (ndarray): The detector coordinate for the x or y center of each knot.
        cents (ndarray): The x or y centroids for each data point.
        nBin (int): The number of knots you want along each axis.
        nData (int): The number of data points.

    Returns:
        tuple: low_bnd (ndarray; the index of the knot to the left of or below each centroid),
            up_bnd (ndarray; the index of the knot to the right of or above each centroid).
    
    """
    
    # pre-finding points "outside" the knots
    bad_low = (cents < knots_pos[0])  
    bad_up = (cents > knots_pos[-1]) 
    
    # calculate the distance between x or y coordinate of each centroid
    # to all knots
    mid_cln = np.transpose(np.tile(knots_pos, (nData,1)))
    
    # lower knots associated
    diff_cln = cents - mid_cln
    diff_cln[diff_cln < 0] = (knots_pos[-1] - knots_pos[0])
    low_bnd = np.argmin(diff_cln**2.0,axis=0)
    
    # upper knots associated
    diff_cln = mid_cln - cents
    diff_cln[diff_cln < 0] = (knots_pos[-1] - knots_pos[0])
    up_bnd = np.argmin(diff_cln**2.0, axis=0)
    
    # tuning l_b upper bound and vice versa
    low_bnd[low_bnd == nBin-1] = nBin-2  
    up_bnd[up_bnd == 0] = 1
    up_bnd[up_bnd == low_bnd] += 1  # Avoiding same bin reference (PROBLEMS?)
    
    # manually extrapolating points "outside" the knots
    low_bnd[bad_low] = 0  
    up_bnd[bad_low] = 1
    low_bnd[bad_up] = nBin-2
    up_bnd[bad_up] = nBin-1
    
    return low_bnd, up_bnd


def which_NNI(knotNdata, low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y):
    """Figure out which data points can use BLISS and which need to use NNI.

    Args:
        knotNdata (ndarray): The number of data assoicated with each knot.
        low_bnd_x (ndarray): The index of the knot to the left of each centroid.
        up_bnd_x (ndarray): The index of the knot to the right of each centroid.
        low_bnd_y (ndarray): The index of the knot below each centroid.
        up_bnd_y (ndarray): The index of the knot above each centroid.
        
    Returns:
        tuple: nni_mask (ndarray; boolean array saying which data points will use NNI),
            bliss_mask (ndarray; boolean array saying which data points will use BLISS).
    
    """
    
    # get mask for points surrounded by a bad knot (no centroid there!)
    # The precompute function sets all points in knotNdata with zero data points within a knot to 0.1
    #     FIX - do we really need to have done that though?!
    bad_left  = np.logical_or(knotNdata[low_bnd_y,low_bnd_x] == 0.1,
                              knotNdata[low_bnd_y,up_bnd_x] == 0.1)
    bad_right = np.logical_or(knotNdata[up_bnd_y,low_bnd_x] == 0.1,
                              knotNdata[up_bnd_y,up_bnd_x] == 0.1)
    nni_mask  = np.logical_or(bad_left, bad_right)
    
    return nni_mask, np.logical_not(nni_mask)


def get_knot_bounds(knots_x, knots_y, low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y):
    """Get the x and y coordinates of the knots adjacted to each centroid.

    Args:
        knots_x (ndarray): The detector coordinate for the x center of each knot.
        knots_y (ndarray): The detector coordinate for the y center of each knot.
        low_bnd_x (ndarray): The index of the knot to the left of each centroid.
        up_bnd_x (ndarray): The index of the knot to the right of each centroid.
        low_bnd_y (ndarray): The index of the knot below each centroid.
        up_bnd_y (ndarray): The index of the knot above each centroid.
        
    Returns:
        tuple: low_bnd_xk (ndarray; the x-position of the knot to the left of each centroid),
            up_bnd_xk (ndarray; the x-position of the knot to the right of each centroid),
            low_bnd_yk (ndarray; the y-position of the knot below each centroid),
            up_bnd_yk (ndarray; the y-position of the knot above each centroid).
    
    """
    
    # turns index of knots into knots coordinates 
    low_bnd_xk = knots_x[low_bnd_x] 
    up_bnd_xk = knots_x[up_bnd_x]
    low_bnd_yk = knots_y[low_bnd_y]
    up_bnd_yk = knots_y[up_bnd_y]
    
    return low_bnd_xk, up_bnd_xk, low_bnd_yk, up_bnd_yk


def bound_knot(xo, yo, low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y,
               low_bnd_xk, up_bnd_xk, low_bnd_yk, up_bnd_yk, nData):
    """Get the x and y coordinates of the knots adjacted to each centroid.

    Args:
        xo (ndarray): The x-centroids for each data point.
        yo (ndarray): The y-centroids for each data point.
        low_bnd_x (ndarray): The index of the knot to the left of each centroid).
        up_bnd_x (ndarray): The index of the knot to the right of each centroid).
        low_bnd_y (ndarray): The index of the knot below each centroid).
        up_bnd_y (ndarray): The index of the knot above each centroid).
        low_bnd_xk (ndarray): The x-position of the knot to the left of each centroid).
        up_bnd_xk (ndarray): The x-position of the knot to the right of each centroid).
        low_bnd_yk (ndarray): The y-position of the knot below each centroid).
        up_bnd_yk (ndarray): The y-position of the knot above each centroid).
        nData (int): The number of data points.
        
    Returns:
        tuple: knot_nrst_x (ndarray; the x-index of closest knot),
            knot_nrst_y (ndarray; the y-index of closest knot).
    
    """
    
    left = (xo - low_bnd_xk <= up_bnd_xk - xo)
    right = np.logical_not(left)
    bottom = (yo - low_bnd_yk <= up_bnd_yk - yo)
    top = np.logical_not(bottom)
    
    knot_nrst_x, knot_nrst_y = np.zeros(nData), np.zeros(nData)
    knot_nrst_x[left]   = low_bnd_x[left]
    knot_nrst_x[right]  = up_bnd_x[right]
    knot_nrst_y[bottom] = low_bnd_y[bottom]
    knot_nrst_y[top]    = up_bnd_y[top]
    
    return knot_nrst_x.astype(int), knot_nrst_y.astype(int)


def bliss_dist(xo, yo, low_bnd_xk, up_bnd_xk, low_bnd_yk, up_bnd_yk):
    """Compute the distance from each centroid to its adjacent knots.

    Args:
        xo (ndarray): The x-centroids for each data point.
        yo (ndarray): The y-centroids for each data point.
        low_bnd_xk (ndarray): The x-position of the knot to the left of each centroid).
        up_bnd_xk (ndarray): The x-position of the knot to the right of each centroid).
        low_bnd_yk (ndarray): The y-position of the knot below each centroid).
        up_bnd_yk (ndarray): The y-position of the knot above each centroid).
        
    Returns:
        tuple: LL_dist (ndarray; the distance to the lower-left knot),
            LR_dist (ndarray; the distance to the lower-right knot),
            UL_dist (ndarray; the distance to the upper-left knot),
            UR_dist (ndarray; the distance to the upper-right knot).
    
    """
    
    LL_dist = (up_bnd_xk - xo)*(up_bnd_yk - yo)
    LR_dist = (xo - low_bnd_xk)*(up_bnd_yk - yo)
    UL_dist = (up_bnd_xk - xo)*(yo - low_bnd_yk)
    UR_dist = (xo - low_bnd_xk)*(yo - low_bnd_yk)
    
    return LL_dist, LR_dist, UL_dist, UR_dist

def map_flux_avgQuick(flux, astroModel, knot_nrst_lin, nBin, knotNdata):
    """Compute the average sensitivity of each knot.

    Args:
        flux (ndarray): The flux measurements for each data point.
        astroModel (ndarray): The astrophysical model for each data point.
        knot_nrst_lin (ndarray): Index of the knot assoicated with each data point.
        nBin (int): The number of knots you want along each axis.
        knotNdata (ndarray): The number of data assoicated with each knot.
        
    Returns:
        ndarray: sensMap (The average flux/astroModel (aka ~ sensitivity) value for each knot).
    
    """
    
    # Avg flux at each data knot
    sensMap = np.bincount(knot_nrst_lin, weights=flux/astroModel,
                          minlength=(nBin*nBin)).reshape((nBin,nBin))
    
    return sensMap/knotNdata


def precompute(flux, xdata, ydata, nBin=10, astroGuess=None, savepath=None, plot=False):
    """Precompute all of the knot associations, etc. that are needed to run BLISS in a fitting routine.

    Args:
        flux (ndarray): The flux measurements for each data point.
        xdata (ndarray): The x-centroid for each data point.
        ydata (ndarray): The y-centroid for each data point.
        nBin (int): The number of knots you want along each axis.
        astroGuess (ndarray): The astrophysical model for each data point.
        savepath (string): The full path to where you would like to save plots that can be used to debug BLISS.
        plot (boolean): Whether or not you want to make plots that can be used to debug BLISS (default is False).
        
    Returns:
        tuple: signal_input (All of the inputs needed to feed into the signal_bliss function)
    
    """
    
    nData = len(xdata)

    (low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y,
     knots_x, knots_y, knotNdata, x_edg, y_edg,
     knots_x_mesh, knots_y_mesh) = lh_axes_binning(xdata, ydata, nBin, nData)
    
    # Avoid division error and y,x for vizualising and consistency
    knotNdata[knotNdata == 0] = 0.1
    knotNdata = np.transpose(knotNdata) 

    # Determine which will be interpolated and which will BLISS
    # Get mask for both methods
    NNI, BLS = which_NNI(knotNdata, low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y)

    # Knots separation
    delta_xo, delta_yo = knots_x[1] - knots_x[0], knots_y[1] - knots_y[0]

    # Converting knot index to knot coordinate
    (low_bnd_xk, up_bnd_xk,
     low_bnd_yk, up_bnd_yk) = get_knot_bounds(knots_x, knots_y,
                                              low_bnd_x, up_bnd_x,
                                              low_bnd_y, up_bnd_y)

    # get closest knot association
    knot_nrst_x, knot_nrst_y = bound_knot(xdata, ydata, low_bnd_x, up_bnd_x,
                                          low_bnd_y, up_bnd_y,
                                          low_bnd_xk, up_bnd_xk,
                                          low_bnd_yk, up_bnd_yk, nData)

    # find distance to 4 associated knots
    LL_dist, LR_dist, UL_dist, UR_dist = bliss_dist(xdata, ydata,
                                                    low_bnd_xk, up_bnd_xk,
                                                    low_bnd_yk, up_bnd_yk)

    # Linear indexing for faster 'np.bincount' routine!
    knot_nrst_lin = knot_nrst_x + (nBin*knot_nrst_y)

    # For full MCMC: jumping in ALL knot parameters
    mask_good_knotNdata  = (np.transpose(knotNdata) != 0.1)
    # The important version for [y,x] consistency
    tmask_good_knotNdata = (knotNdata != 0.1)  
    tot_goodK     = len(knotNdata[tmask_good_knotNdata])

    # print('N/K = %.2f' % (nData/tot_goodK))
    
    if plot:
        make_plots.plot_knots(xdata, ydata, flux, astroGuess, knot_nrst_lin,
                              tmask_good_knotNdata, knots_x, knots_y, 
                              knots_x_mesh, knots_y_mesh, nBin, knotNdata, savepath)
    
    return (flux, nBin, nData, knotNdata,
            low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y, LL_dist, LR_dist, UL_dist, UR_dist,
            delta_xo, delta_yo, knot_nrst_x, knot_nrst_y, knot_nrst_lin, BLS, NNI,
            knots_x_mesh, knots_y_mesh, tmask_good_knotNdata)
