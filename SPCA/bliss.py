import numpy as np

# SPCA libraries
from . import make_plots

def knot_assiation(knots_pos, cents, nBin, nData):
    """Find the two knots adjacted to each knot to later be used with linear interpolation.

    Args:
        knots_pos (array): The detector coordinate for the x or y center of each knot.
        cents (array): The x or y centroids for each data point.
        nBinX (int): The number of knots you want along the axis.
        nData (int): The number of data points.

    Returns:
        tuple: low_bnd (array; the index of the knot to the left of or below each centroid),
               up_bnd (array; the index of the knot to the right of or above each centroid).
    
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

def compute_sensMap(flux, astroModel, knot_nrst_lin, nBinX, nBinY, knotNdata):
    """Compute the average sensitivity of each knot.

    Args:
        flux (ndarray): The flux measurements for each data point.
        astroModel (ndarray): The astrophysical model for each data point.
        knot_nrst_lin (ndarray): Index of the knot assoicated with each data point.
        nBinX (int): The number of knots you want along the x-axis.
        nBinY (int): The number of knots you want along the y-axis.
        knotNdata (ndarray): The number of data assoicated with each knot.
        
    Returns:
        ndarray: sensMap (The average flux/astroModel (aka ~ sensitivity) value for each knot).
    
    """
    
    # Avg flux at each data knot
    #    knotNdata was transposed, so need to have (nBinY, nBinX)
    sensMap = np.bincount(knot_nrst_lin, weights=flux/astroModel, minlength=(nBinY*nBinX)).reshape((nBinY, nBinX))/knotNdata
    
    return sensMap

def precompute(flux, xdata, ydata, nBinX=8, nBinY=8, astroModel=None, savepath=None, plot=False, showPlot=False):
    """Precompute all of the knot associations, etc. that are needed to run BLISS in a fitting routine.

    Args:
        flux (ndarray): The flux measurements for each data point.
        xdata (ndarray): The x-centroid for each data point.
        ydata (ndarray): The y-centroid for each data point.
        nBinX (int, optional): The number of knots you want along the x-axis.
        nBinY (int, optional): The number of knots you want along the y-axis.
        astroModel (ndarray, optional): The astrophysical model for each data point.
        savepath (string, optional): The full path to where you would like to save plots that can be used to debug BLISS.
        plot (boolean, optional): Whether or not you want to make plots that can be used to debug BLISS (default is False).
        
    Returns:
        tuple: signal_input (All of the inputs needed to feed into the signal_bliss function)
    
    """
    
    nData = len(xdata)

    # Create the knots
    knotNdata, x_edg, y_edg = np.histogram2d(xdata, ydata, bins=[nBinX, nBinY])
    # Avoid division error and y,x for vizualising and consistency
    knotNdata[knotNdata == 0] = 0.1
    knotNdata = np.transpose(knotNdata) 
    
    # get the coordinate of the centre of the knots
    knots_x = x_edg[1:] - 0.5*(x_edg[1:] - x_edg[0:-1])
    knots_y = y_edg[1:] - 0.5*(y_edg[1:] - y_edg[0:-1])
    
    # get knots boundaries
    low_bnd_x, up_bnd_x = knot_assiation(knots_x, xdata, nBinX, nData)
    low_bnd_y, up_bnd_y = knot_assiation(knots_y, ydata, nBinY, nData)

    # Make the 2D grid of knots
    knots_x_mesh, knots_y_mesh = np.meshgrid(knots_x, knots_y)

    # Determine which will be interpolated and which will BLISS
    #     The precompute function sets all points in knotNdata with zero data points within a knot to 0.1
    #         FIX - do we really need to have done that though?!
    bad_left  = np.logical_or(knotNdata[low_bnd_y,low_bnd_x] == 0.1,
                              knotNdata[low_bnd_y,up_bnd_x] == 0.1)
    bad_right = np.logical_or(knotNdata[up_bnd_y,low_bnd_x] == 0.1,
                              knotNdata[up_bnd_y,up_bnd_x] == 0.1)
    NNI  = np.logical_or(bad_left, bad_right)
    BLS = np.logical_not(NNI)
    
    # Knots separation
    delta_xo, delta_yo = knots_x[1] - knots_x[0], knots_y[1] - knots_y[0]

    # Converting knot index to knot coordinate
    low_bnd_xk = knots_x[low_bnd_x] 
    up_bnd_xk = knots_x[up_bnd_x]
    low_bnd_yk = knots_y[low_bnd_y]
    up_bnd_yk = knots_y[up_bnd_y]
    
    # Get the x and y coordinates of the knots adjacted to each centroid.
    left = (xdata - low_bnd_xk <= up_bnd_xk - xdata)
    right = np.logical_not(left)
    bottom = (ydata - low_bnd_yk <= up_bnd_yk - ydata)
    top = np.logical_not(bottom)
    knot_nrst_x, knot_nrst_y = np.zeros(nData), np.zeros(nData)
    knot_nrst_x[left]   = low_bnd_x[left]
    knot_nrst_x[right]  = up_bnd_x[right]
    knot_nrst_y[bottom] = low_bnd_y[bottom]
    knot_nrst_y[top]    = up_bnd_y[top]
    knot_nrst_x = knot_nrst_x.astype(int)
    knot_nrst_y = knot_nrst_y.astype(int)

    # find distance to 4 associated knots
    LL_dist = (up_bnd_xk - xdata)*(up_bnd_yk - ydata)
    LR_dist = (xdata - low_bnd_xk)*(up_bnd_yk - ydata)
    UL_dist = (up_bnd_xk - xdata)*(ydata - low_bnd_yk)
    UR_dist = (xdata - low_bnd_xk)*(ydata - low_bnd_yk)

    # Linear indexing for faster 'np.bincount' routine!
    knot_nrst_lin = knot_nrst_x + (nBinX*knot_nrst_y)

    # For full MCMC: jumping in ALL knot parameters
    mask_good_knotNdata  = (np.transpose(knotNdata) != 0.1)
    # The important version for [y,x] consistency
    tmask_good_knotNdata = (knotNdata != 0.1)  
    tot_goodK     = len(knotNdata[tmask_good_knotNdata])
    
    if plot:
        make_plots.plot_knots(xdata, ydata, tmask_good_knotNdata, knots_x, knots_y, 
                              knots_x_mesh, knots_y_mesh, knotNdata, savepath, showPlot)
    
    return (flux, xdata, ydata, nBinX, nBinY, knotNdata, low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y,
            LL_dist, LR_dist, UL_dist, UR_dist, delta_xo, delta_yo, knot_nrst_x, knot_nrst_y,
            knot_nrst_lin, BLS, NNI)
