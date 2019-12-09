import numpy as np

# SPCA libraries
from . import make_plots
from . import helpers
from . import astro_models

def lh_axes_binning(xo, yo, nBin, nData):
    '''
    Input:
        xo    = x-centroids
        yo    = y-centroids
        nBin  = number of bins along 1 axis
        nData = number of data points
    '''
    knotNdata, x_edg, y_edg = np.histogram2d(xo, yo, bins=nBin)
    
    # get the coordinate of the centre of the knots
    knots_x = x_edg[1:] - 0.5*(x_edg[1:] - x_edg[0:-1])
    knots_y = y_edg[1:] - 0.5*(y_edg[1:] - y_edg[0:-1])
    
    # get knots boundaries
    low_bnd_x, up_bnd_x = lh_knot_ass(knots_x, xo, nBin, nData)
    low_bnd_y, up_bnd_y = lh_knot_ass(knots_y, yo, nBin, nData)
    
    knots_x_mesh, knots_y_mesh = np.meshgrid(knots_x, knots_y)
    
    '''
    Output:
        low_bx,high_bx,low_by,high_by = bound association of each knots (index)
        x_Knots,y_Knots               = coordinated of the knots
        BK_T                          = bin counts
        xEdg,yEdg                     = coordinates of edges of the bin
        xKmesh,yKmesh                 = meshigrid for the knots
    '''
    return (low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y, knots_x, knots_y,
            knotNdata, x_edg, y_edg, knots_x_mesh, knots_y_mesh)


def lh_knot_ass(knots_pos, cents, nBin, nData):
    '''
    Input:
        knots_pos = x or y coordinate of knots
        cents     = x or y coordinate of centroids
        nBin      = number of bins along 1 axis
        nData     = number of data points
    '''
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
    
    '''
    Output:
        low_bnd = index of lower knots associated
        up_bnd  = index of upper knots associated
    '''
    return low_bnd, up_bnd


def which_NNI(knotNdata, low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y):  
    '''
    Input:
        knotNdata = bin counts
        low_bnd_x = index of lower x-knots associated
        up_bnd_x  = index of upper x-knots associated
        low_bnd_y = index of lower y-knots associated
        up_bnd_y  = index of upper y-knots associated
    '''
    # get mask for points surrounded by a bad knot (no centroid there!)
    bad_left  = np.logical_or(knotNdata[low_bnd_y,low_bnd_x] == 0.1,
                              knotNdata[low_bnd_y,up_bnd_x] == 0.1)
    bad_right = np.logical_or(knotNdata[up_bnd_y,low_bnd_x] == 0.1,
                              knotNdata[up_bnd_y,up_bnd_x] == 0.1)
    nni_mask  = np.logical_or(bad_left, bad_right)
    '''
    Output:
        nni_mask                 = mask the data that should be NNI
        np.logical_not(nni_mask) = mask the data that should be BLISS
    '''
    return nni_mask, np.logical_not(nni_mask)


def get_knot_bounds(knots_x, knots_y, low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y):
    '''
    Input:
        x_k    = coord of x-knots
        y_k    = coord of y-knots
        low_bnd_x = index of lower x-knots associated
        up_bnd_x  = index of upper x-knots associated
        low_bnd_y = index of lower y-knots associated
        up_bnd_y  = index of upper y-knots associated
    '''
    # turns index of knots into knots coordinates 
    low_bnd_xk = knots_x[low_bnd_x] 
    up_bnd_xk = knots_x[up_bnd_x]
    low_bnd_yk = knots_y[low_bnd_y]
    up_bnd_yk = knots_y[up_bnd_y]
    '''
    Output:
        low_bnd_xk = coord of lower x-knots associated
        up_bnd_xk  = coord of upper x-knots associated
        low_bnd_yk = coord of lower y-knots associated
        up_bnd_yk  = coord of upper y-knots associated
    '''
    return low_bnd_xk, up_bnd_xk, low_bnd_yk, up_bnd_yk


def bound_knot(xo, yo, low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y,
               low_bnd_xk, up_bnd_xk, low_bnd_yk, up_bnd_yk, nData):
    '''
    Input:
        xo         = coord of x-centroid
        yo         = coord of y-centroid
        low_bnd_x  = index of lower x-knots associated
        up_bnd_x   = index of upper x-knots associated
        low_bnd_y  = index of lower y-knots associated
        up_bnd_y   = index of upper y-knots associated
        low_bnd_xk = coord of lower x-knots associated
        up_bnd_xk  = coord of upper x-knots associated
        low_bnd_yk = coord of lower y-knots associated
        up_bnd_yk  = coord of upper y-knots associated
        nData      = number of data points
    '''
    left = (xo - low_bnd_xk <= up_bnd_xk - xo)
    right = np.logical_not(left)
    bottom = (yo - low_bnd_yk <= up_bnd_yk - yo)
    top = np.logical_not(bottom)
    
    knot_nrst_x, knot_nrst_y = np.zeros(nData), np.zeros(nData)
    knot_nrst_x[left]   = low_bnd_x[left]
    knot_nrst_x[right]  = up_bnd_x[right]
    knot_nrst_y[bottom] = low_bnd_y[bottom]
    knot_nrst_y[top]    = up_bnd_y[top]
    
    '''
    Output:
        knot_nrst_x = index of closest knot
        knot_nrst_y = index of closest knot
    '''
    return knot_nrst_x.astype(int), knot_nrst_y.astype(int)


def bliss_dist(xo, yo, low_bnd_xk, up_bnd_xk, low_bnd_yk, up_bnd_yk):
    '''
    Input:
        xo    = coord of x-centroid
        yo    = coord of y-centroid
        low_bnd_xk  = coord of lower and lower x-knots associated
        up_bnd_xk  = coord of upper and lower x-knots associated
        low_bnd_yk  = coord of lower and lower y-knots associated
        up_bnd_yk  = coord of upper and lower y-knots associated
    '''
    LL_dist = (up_bnd_xk - xo)*(up_bnd_yk - yo)
    LR_dist = (xo - low_bnd_xk)*(up_bnd_yk - yo)
    UL_dist = (up_bnd_xk - xo)*(yo - low_bnd_yk)
    UR_dist = (xo - low_bnd_xk)*(yo - low_bnd_yk)
    '''
    Output:
        LL_dist = distance to lower-left  knot
        LR_dist = distance to lower-right knot
        UL_dist = distance to upper-left  knot
        UR_dist = distance to upper-right knot
    '''
    return LL_dist, LR_dist, UL_dist, UR_dist

def map_flux_avgQuick(flux, astroModel, knot_nrst_lin, nBin, knotNdata):
    '''
    Input:
        flux          = measurements
        astroModel    = astro model
        knot_nrst_lin = index of bin associated with
        nBin          = number of bins along 1 axis
        sensMap       = bin counts to get mean
    '''
    # Avg flux at each data knot
    sensMap = np.bincount(knot_nrst_lin, weights=flux/astroModel,
                          minlength=(nBin*nBin)).reshape((nBin,nBin))
    # Using [y,x] for consistency
    '''
    Output:
        sensMap = average values at knots 
    '''
    return sensMap/knotNdata


def precompute(flux, time, xdata, ydata, psfxw, psfyw, mode, astroGuess, nBin=10, savepath=None, plot=True):
    '''Pre-computing D(xo,yo) associations'''
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
    
    return (flux, time, psfxw, psfyw, nBin, nData, knotNdata,
            low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y, LL_dist, LR_dist, UL_dist, UR_dist,
            delta_xo, delta_yo, knot_nrst_x, knot_nrst_y, knot_nrst_lin, BLS, NNI,
            knots_x_mesh, knots_y_mesh, tmask_good_knotNdata, mode)
