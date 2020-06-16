import numpy as np

try:
    import george
    from george.modeling import Model
except ImportError:
    print('Warning: george failed to import. Without this installed, you cannot run GP analyses.')
    print('For instructions on how to install george, visit https://george.readthedocs.io')
    print('Alternatively, if installing SPCA with "pip install .", you can also specify "pip install .[GP]"')


######################################################################################
#THIS IS THE MAIN SIGNAL FUNCTION WHICH BRANCHES OUT TO THE CORRECT SIGNAL FUNCTION
def signal(p0, p0_labels, astrofunc, astro_labels, astro_input, detecfuncs, detec_labels, detec_inputs):
    
    # Compute the astrophysical signal
    astro_dictionary = dict([[label,p0[i]]  for i, label in enumerate(p0_labels) if label in astro_labels])
    astr   = astrofunc(astro_input, **astro_dictionary)
    
    # Iterate over all the different detector models
    fulldetec = 1
    gp = None
    for detecfunc, detec_label, detec_input in zip(detecfuncs, detec_labels, detec_inputs):
        # Make a dictionary of the fitted parameters relevant to this detector model
        detec_dictionary = dict([[label,p0[i]]  for i, label in enumerate(p0_labels) if label in detec_label])
        
        # Compute the detector model
        detec = detecfunc(detec_input, astr, **detec_dictionary)
        
        if type(detec)==np.ndarray:
            # Include this model trend in the detector model
            fulldetec *= detec
        else:
            # We are returning a GP object to compute the log-likelihood
            gp = detec
    
    if gp is None:
        return astr*fulldetec
    else:
        return astr*fulldetec, gp

######################################################################################
#THESE ARE THE INDIVIDUAL DETECTOR MODEL FUNCTIONS WHICH MODEL THE DETECTOR SYSTEMATICS

def detec_model_poly(detec_inputs, astroModel, c1, c2, c3, c4, c5, c6, c7=0, c8=0, c9=0, c10=0, c11=0, 
                     c12=0, c13=0, c14=0, c15=0, c16=0, c17=0, c18=0, c19=0, c20=0, c21=0,
                     d1=1, d2=0, d3=0,
                     s0=0, s0break=0, s1=0, s1break=0, s2=0, s2break=0, s3=0, s3break=0, s4=0, s4break=0,
                     m1=0):
    """Model the detector systematics with a 2D polynomial model based on the centroid.

    Args:
        detec_inputs (tuple): (x, y, mode) with dtypes (ndarray, ndarray, string). Formatted this
            way to allow for easy minimization with scipy.optimize.minimize.
        c1--c21 (float): The polynomial model amplitudes.

    Returns:
        ndarray: The flux variations due to the detector systematics.

    """
    
    x, y, mode = detec_inputs
    
    if   'poly2' in mode.lower():
        pos = np.vstack((np.ones_like(x),
                        x   ,      y,
                        x**2, x   *y,      y**2))
        detec = np.array([c1, c2, c3, c4, c5, c6])
    elif 'poly3' in mode.lower():
        pos = np.vstack((np.ones_like(x),
                        x   ,      y,
                        x**2, x   *y,      y**2,
                        x**3, x**2*y,    x*y**2,      y**3))
        detec = np.array([c1,  c2,  c3,  c4,  c5,  c6,
                          c7,  c8,  c9,  c10,])
    elif 'poly4' in mode.lower():
        pos = np.vstack((np.ones_like(x),
                        x   ,      y,
                        x**2, x   *y,      y**2,
                        x**3, x**2*y,    x*y**2,      y**3,
                        x**4, x**3*y, x**2*y**2, x**1*y**3,   y**4))
        detec = np.array([c1,  c2,  c3,  c4,  c5,  c6,
                          c7,  c8,  c9,  c10,
                          c11, c12, c13, c14, c15,])
    elif 'poly5' in mode.lower():
        pos = np.vstack((np.ones_like(x),
                        x   ,      y,
                        x**2, x   *y,      y**2,
                        x**3, x**2*y,    x*y**2,      y**3,
                        x**4, x**3*y, x**2*y**2, x**1*y**3,   y**4,
                        x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5))
        detec = np.array([c1,  c2,  c3,  c4,  c5,  c6,
                          c7,  c8,  c9,  c10,
                          c11, c12, c13, c14, c15,
                          c16, c17, c18, c19, c20, c21])
    
    return np.dot(detec[np.newaxis,:], pos).reshape(-1)

def detec_model_PLD(input_data, astroModel, p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1,
                    p10_1=0, p11_1=0, p12_1=0, p13_1=0, p14_1=0, p15_1=0,
                    p16_1=0, p17_1=0, p18_1=0, p19_1=0, p20_1=0, p21_1=0,
                    p22_1=0, p23_1=0, p24_1=0, p25_1=0,
                    p1_2=0, p2_2=0, p3_2=0, p4_2=0, p5_2=0, p6_2=0, p7_2=0,
                    p8_2=0, p9_2=0, p10_2=0, p11_2=0, p12_2=0, p13_2=0, p14_2=0,
                    p15_2=0, p16_2=0, p17_2=0, p18_2=0, p19_2=0, p20_2=0, p21_2=0,
                    p22_2=0, p23_2=0, p24_2=0, p25_2=0):
    """Model the detector systematics with a PLD model.
    
    Args:
        input_data (tuple): (Pgroup, mode) with dtypes (ndarray, string).
        p0_0 (float): The constant offset term for PLD decorrelation.
        p1_1--p25_1 (float): The 1st order PLD coefficients for 3x3 or 5x5 PLD stamps.
        p1_2--p25_2 (float): The 2nd order PLD coefficients for 3x3 or 5x5 PLD stamps.
        
    Returns:
        ndarray: The flux variations due to the detector systematics.

    """
    
    pixels, mode = input_data # Pgroup are pixel "lightcurves" 
    
    detec = [p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1]
    if '5x5' in mode.lower():
        # Add additional pixels
        detec.extend([p10_1, p11_1, p12_1, p13_1, p14_1, p15_1, p16_1, p17_1,
                      p18_1, p19_1, p20_1, p21_1, p22_1, p23_1, p24_1, p25_1])
    if 'pld2' in mode.lower() or 'pldaper2' in mode.lower():
        # Add higher order terms for 3x3 pixels
        detec.extend([p1_2, p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, p8_2, p9_2])
    if ('pld2' in mode.lower() or 'pldaper2' in mode.lower()) and '5x5' in mode.lower():
        # Add higher order terms for 5x5 pixels
        detec.extend([p10_2, p11_2, p12_2, p13_2, p14_2, p15_2, p16_2, p17_2,
                      p18_2, p19_2, p20_2, p21_2, p22_2, p23_2, p24_2, p25_2])
    
    return np.dot(np.array(detec), pixels).reshape(-1)

def detec_model_bliss(signal_input, astroModel):
    """Model the detector systematics with a BLISS model based on the centroid.

    Args:
        signal_input (tuple): (flux, time, psfxw, psfyw, nBin, nData, knotNdata, low_bnd_x,
            up_bnd_x, low_bnd_y, up_bnd_y, LL_dist, LR_dist, UL_dist, UR_dist,
            delta_xo, delta_yo, knot_nrst_x, knot_nrst_y, knot_nrst_lin, BLS, NNI,
            knots_x_mesh, knots_y_mesh, tmask_good_knotNdata, mode) with dtypes (????). # FIX dtypes!
        astroModel (ndarray): The modelled astrophysical flux variations.

    Returns:
        ndarray: The flux variations due to the detector systematics.

    """
    
    '''
    Input:
        sensMap = sensitivity values at the knots
        nData   = number of measurements
        LL_dist = distance to lower-left  knot
        LR_dist = distance to lower-right knot
        UL_dist = distance to upper-left  knot
        UR_dist = distance to upper-right knot
        BLS     = BLISS points
        NNI     = NNI points
    '''
    
    (flux, time, psfxw, psfyw, nBin, nData, knotNdata, low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y,
     LL_dist, LR_dist, UL_dist, UR_dist, delta_xo, delta_yo, knot_nrst_x, knot_nrst_y,
     knot_nrst_lin, BLS, NNI, knots_x_mesh, knots_y_mesh, tmask_good_knotNdata, mode) = signal_input
    
    sensMap = np.bincount(knot_nrst_lin, weights=flux/astroModel, minlength=(nBin*nBin)).reshape((nBin,nBin))
    sensMap /= knotNdata
    
    detec = np.empty(nData)
    
    # weight knots values by distance to knots
    LL = sensMap[low_bnd_y, low_bnd_x]*LL_dist
    LR = sensMap[low_bnd_y, up_bnd_x]*LR_dist
    UL = sensMap[up_bnd_y, low_bnd_x]*UL_dist
    UR = sensMap[up_bnd_y, up_bnd_x]*UR_dist
    
    # BLISS points
    detec[BLS] = (LL[BLS] + LR[BLS] + UL[BLS] + UR[BLS])/(delta_xo*delta_yo)
    
    # Nearest Neighbor points
    detec[NNI] = sensMap[knot_nrst_y[NNI],knot_nrst_x[NNI]]  
    
    return detec

def detec_model_GP(input_data, astroModel, gpAmp, gpLx, gpLy, sigF):
    """Model the detector systematics with a Gaussian process based on the centroid.

    Args:
        input_data (tuple): (flux, xdata, ydata, time, returnGp, astroModel) with dtypes
            (ndarray, ndarray, ndarray, ndarray, bool, ndarray). Formatted this way to allow
            for scipy.optimize.minimize optimization.
        gpAmp (float): The natural logarithm of the GP covariance amplitude.
        gpLx (float): The natural logarithm of the GP covariance lengthscale in x.
        gpLy (float): The natural logarithm of the GP covariance lengthscale in y.
        sigF (float): The white noise in units of F_star.

    Returns:
        ndarray: The flux variations due to the detector systematics.

    """
    
    flux, xdata, ydata, predictGp = input_data
    
    gp = george.GP(np.exp(gpAmp)*george.kernels.ExpSquaredKernel(np.exp([gpLx, gpLy]), ndim=2, axes=[0, 1]))#, solver=george.HODLRSolver, tol=1e-8)
    
    gp.compute(np.array([xdata, ydata]).T, sigF)
    
    if predictGp:
        mu, _ = gp.predict(flux-astroModel, np.array([xdata, ydata]).T)
        mu = 1.+mu/astroModel
        return mu
    else:
        return gp    

def detec_model_PSFW(input_data, astroModel, d1=1, d2=0, d3=0):
    """Model the detector systematics with a simple linear model based on the PSF width.

    Args:
        detec_inputs (tuple): (x, y, mode) with dtypes (ndarray, ndarray, string). Formatted this
            way to allow for easy minimization with scipy.optimize.minimize.
        d1 (float): The constant offset term. #FIX - I don't think this should be here.
        d2 (float): The slope in sensitivity with the PSF width in the x direction.
        d3 (float): The slope in sensitivity with the PSF width in the y direction.

    Returns:
        ndarray: The flux variations due to the detector systematics.

    """
    px, py = input_data
    pw     = np.vstack((np.ones_like(px), px, py))
    syst   = np.array([d1, d2, d3])
    return np.dot(syst[np.newaxis,:], pw).reshape(-1)

def hside(time, astroModel, s0=0, s0break=0, s1=0, s1break=0, s2=0, s2break=0, s3=0, s3break=0, s4=0, s4break=0):
    """Model the detector systematics with a heaviside step function at up to five AOR breaks.

    Args:
        time (ndarray): The time.
        s# (float): The amplitude of the heaviside step function.
        s#break (float): The time of the aor break

    Returns:
        ndarray: The flux variations due to the detector systematics.

    """
    model = np.ones_like(time)
    amplitudes = [s0, s1, s2, s3, s4]
    breaks = [s0break, s1break, s2break, s3break, s4break]
    for i, amp in enumerate(amplitudes):
        if amp==0:
            continue
        model += amp*np.heaviside(time-breaks[i], 1)
    
    return model

def tslope(time, astroModel, m1=0):
    """Model the detector systematics with a simple slope in time.

    Args:
        time (ndarray): The time.
        m1 (float): The slope in sensitivity over time with respect to time[0].

    Returns:
        ndarray: The flux variations due to the detector systematics.

    """
    
    return 1+(time-time[0])*m1
