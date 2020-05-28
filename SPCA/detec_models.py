import numpy as np

try:
    import george
    from george.modeling import Model
except ImportError:
    print('Warning: george failed to import. Without this installed, you cannot run GP analyses.')
    print('For instructions on how to install george, visit https://george.readthedocs.io')
    print('Alternatively, if installing SPCA with "pip install .", you can also specify "pip install .[GP]"')

# SPCA libraries
from . import astro_models



######################################################################################
#THIS IS THE MAIN SIGNAL FUNCTION WHICH BRANCHES OUT TO THE CORRECT SIGNAL FUNCTION
def signal(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
           c1,  c2,  c3,  c4,  c5,  c6, c7,  c8,  c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, 
           d1, d2, d3, s1, s2, m1, 
           p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1, p10_1, p11_1, p12_1, p13_1, p14_1, p15_1,
           p16_1, p17_1, p18_1, p19_1, p20_1, p21_1, p22_1, p23_1, p24_1, p25_1,
           p1_2, p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, p8_2, p9_2, p10_2, p11_2, p12_2, p13_2, p14_2, p15_2,
           p16_2, p17_2, p18_2, p19_2, p20_2, p21_2, p22_2, p23_2, p24_2, p25_2,
           gpAmp, gpLx, gpLy, sigF,
           predictGp=True, returnGp=False):
    """Model the flux variations as a product of astrophysical varations multiplied by a non-uniform detector sensitivity.
    
    This is a super-function that sets up the framework of SPCA. It calls the relevant astrophysical functions and the relevant detector model functions, depending on the value of mode: the last variable in the signal_input parameter.

    Args:
        signal_input (tuple): Varying contents depending on the detector model.
            The last value of the tuple is invariably the mode string.
        t0 (float): Time of inferior conjunction.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        a (float): Semi-major axis (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        ecosw (float): Eccentricity multiplied by the cosine of the longitude of periastron (value between -1 and 1).
        esinw (float): Eccentricity multiplied by the sine of the longitude of periastron (value between -1 and 1).
        q1 (float): Limb darkening coefficient 1, parametrized to range between 0 and 1.
        q2 (float): Limb darkening coefficient 2, parametrized to range between 0 and 1.
        fp (float): Planet-to-star flux ratio.
        A (float): Amplitude of the first-order cosine term.
        B (float): Amplitude of the first-order sine term.
        C (float): Amplitude of the second-order cosine term. Default=0.
        D (float): Amplitude of the second-order sine term. Default=0.
        r2 (float): Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
        r2off (float): Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.
        c1--c21 (float): The polynomial model amplitudes.
        d1 (float): The constant offset term. #FIX - I don't think this should be here.
        d2 (float): The slope in sensitivity with the PSF width in the x direction.
        d3 (float): The slope in sensitivity with the PSF width in the y direction.
        s1 (float): The amplitude of the heaviside step function.
        s2 (float): The location of the step in the heaviside function.
        m1 (float): The slope in sensitivity over time with respect to time[0].
        p1_1--p25_1 (float): The 1st order PLD coefficients for 3x3 or 5x5 PLD stamps.
        p1_2--p25_2 (float): The 2nd order PLD coefficients for 3x3 or 5x5 PLD stamps.
        gpAmp (float): The natural logarithm of the GP covariance amplitude.
        gpLx (float): The natural logarithm of the GP covariance lengthscale in x.
        gpLy (float): The natural logarithm of the GP covariance lengthscale in y.
        sigF (float): The white noise in units of F_star.
        predictGp (bool, optional): Should the GP make predictions (True, default), or just return the GP (useful for lnlike).
        returnGp (bool, optional): Should the GP model return the GP object (True, useful for lnlike) or not (False, default).

    Returns:
        ndarray: The modelled flux variations due to the astrophysical model modified by the detector model.

    """

    mode = signal_input[-1]
    if 'poly' in mode.lower():
        return signal_poly(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                           c1,  c2,  c3,  c4,  c5,  c6,  c7,  c8,  c9, c10, c11, c12, c13, c14, c15, c16, c17,
                           c18, c19, c20, c21, d1,  d2,  d3,  s1,  s2, m1)
    elif 'pld' in mode.lower():
        return signal_PLD(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                          p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1, p10_1, p11_1, p12_1, p13_1, p14_1, p15_1,
                          p16_1, p17_1, p18_1, p19_1, p20_1, p21_1, p22_1, p23_1, p24_1, p25_1,
                          p1_2, p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, p8_2, p9_2, p10_2, p11_2, p12_2, p13_2, p14_2, p15_2,
                          p16_2, p17_2, p18_2, p19_2, p20_2, p21_2, p22_2, p23_2, p24_2, p25_2,
                          s1, s2, m1, sigF)
    elif 'bliss' in mode.lower():
        return signal_bliss(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                            d1, d2, d3, s1, s2, m1)
    elif 'gp' in mode.lower():
        return signal_GP(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                         d1, d2, d3, s1, s2, m1,
                         gpAmp, gpLx, gpLy, sigF,
                         predictGp, returnGp)

######################################################################################
#THESE ARE THE INDIVIDUAL SIGNAL FUNCTIONS WHICH MODEL THE FULL SIGNAL

def signal_poly(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                c1,  c2,  c3,  c4,  c5,  c6, c7,  c8,  c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, 
                d1, d2, d3, s1, s2, m1):
    """Model the flux variations as a product of astrophysical varations multiplied by a 2D polynomial detector sensitivity model.

    Args:
        signal_input (tuple): (flux, time, xdata, ydata, psfwx, psfwy, mode) with dtypes
            (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, string).
        t0 (float): Time of inferior conjunction.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        a (float): Semi-major axis (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        ecosw (float): Eccentricity multiplied by the cosine of the longitude of periastron (value between -1 and 1).
        esinw (float): Eccentricity multiplied by the sine of the longitude of periastron (value between -1 and 1).
        q1 (float): Limb darkening coefficient 1, parametrized to range between 0 and 1.
        q2 (float): Limb darkening coefficient 2, parametrized to range between 0 and 1.
        fp (float): Planet-to-star flux ratio.
        A (float): Amplitude of the first-order cosine term.
        B (float): Amplitude of the first-order sine term.
        C (float): Amplitude of the second-order cosine term. Default=0.
        D (float): Amplitude of the second-order sine term. Default=0.
        r2 (float): Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
        r2off (float): Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.
        c1--c21 (float): The polynomial model amplitudes.
        d1 (float): The constant offset term. #FIX - I don't think this should be here.
        d2 (float): The slope in sensitivity with the PSF width in the x direction.
        d3 (float): The slope in sensitivity with the PSF width in the y direction.
        s1 (float): The amplitude of the heaviside step function.
        s2 (float): The location of the step in the heaviside function.
        m1 (float): The slope in sensitivity over time with respect to time[0].

    Returns:
        ndarray: The modelled flux variations due to the astrophysical model modified by the detector model.

    """
    
    #flux, time, xdata, ydata, psfwx, psfwy, mode = signal_input
    
    time = signal_input[1]
    xdata = signal_input[2]
    ydata = signal_input[3]
    mode = signal_input[-1]
    
    astr   = astro_models.ideal_lightcurve(time, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, 
                                           A, B, C, D, r2, r2off)
    model = astr*detec_model_poly((xdata, ydata, mode), c1,  c2,  c3,  c4,  c5,  c6, c7,  c8,  c9, c10, 
                                           c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21)
    
    if 'psfw' in mode.lower():
        psfwidths = signal_input[4:6]
        model *= detec_model_PSFW(psfwidths, d1, d2, d3)
    
    if 'hside' in mode.lower():
        model *= hside(time, s1, s2)
        
    if 'tslope' in mode.lower():
        model *= tslope(time, m1)
  
    return model

def signal_PLD(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
               p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1, p10_1, p11_1, p12_1, p13_1, p14_1, p15_1,
               p16_1, p17_1, p18_1, p19_1, p20_1, p21_1, p22_1, p23_1, p24_1, p25_1,
               p1_2, p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, p8_2, p9_2, p10_2, p11_2, p12_2, p13_2, p14_2, p15_2,
               p16_2, p17_2, p18_2, p19_2, p20_2, p21_2, p22_2, p23_2, p24_2, p25_2,
               s1, s2, m1, sigF):
    """Model the flux variations as a product of astrophysical varations multiplied by a PLD sensitivity model.
    
    Args:
        signal_input (tuple): (flux, time, Pgroup, mode) with dtypes (ndarray, ndarray, ndarray, string).
        t0 (float): Time of inferior conjunction.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        a (float): Semi-major axis (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        ecosw (float): Eccentricity multiplied by the cosine of the longitude of periastron (value between -1 and 1).
        esinw (float): Eccentricity multiplied by the sine of the longitude of periastron (value between -1 and 1).
        q1 (float): Limb darkening coefficient 1, parametrized to range between 0 and 1.
        q2 (float): Limb darkening coefficient 2, parametrized to range between 0 and 1.
        fp (float): Planet-to-star flux ratio.
        A (float): Amplitude of the first-order cosine term.
        B (float): Amplitude of the first-order sine term.
        C (float): Amplitude of the second-order cosine term. Default=0.
        D (float): Amplitude of the second-order sine term. Default=0.
        r2 (float): Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
        r2off (float): Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.
        p1_1--p25_1 (float): The 1st order PLD coefficients for 3x3 or 5x5 PLD stamps.
        p1_2--p25_2 (float): The 2nd order PLD coefficients for 3x3 or 5x5 PLD stamps.
        s1 (float): The amplitude of the heaviside step function.
        s2 (float): The location of the step in the heaviside function.
        m1 (float): The slope in sensitivity over time with respect to time[0].
        sigF (float): The white noise in units of F_star.
        
    Returns:
        ndarray: The modelled flux variations due to the astrophysical model modified by the detector model.

    """
    
    
    #flux, time, Pgroup, mode = signal_input
    time   = signal_input[1]
    Pgroup = signal_input[2]
    mode   = signal_input[-1]
    
    astroModel   = astro_models.ideal_lightcurve(time, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, 
                                           A, B, C, D, r2, r2off)
    detec  = detec_model_PLD((Pgroup, mode), p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1,
                             p10_1, p11_1, p12_1, p13_1, p14_1, p15_1, p16_1, p17_1, p18_1, p19_1,
                             p20_1, p21_1, p22_1, p23_1, p24_1, p25_1,
                             p1_2, p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, p8_2, p9_2, p10_2, p11_2,
                             p12_2, p13_2, p14_2, p15_2, p16_2, p17_2, p18_2, p19_2, p20_2, p21_2,
                             p22_2, p23_2, p24_2, p25_2)
    
    model = astroModel*detec
    
    if 'hside' in mode.lower():
        model *= hside(time, s1, s2)
    
    if 'tslope' in mode.lower():
        model *= tslope(time, m1)
    
    return model

def signal_bliss(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off, 
                 d1, d2, d3, s1, s2, m1):
    """Model the flux variations as a product of astrophysical varations multiplied by a BLISS detector sensitivity model.

    Args:
        signal_input (tuple): (flux, time, psfxw, psfyw, nBin, nData, knotNdata, low_bnd_x,
            up_bnd_x, low_bnd_y, up_bnd_y, LL_dist, LR_dist, UL_dist, UR_dist,
            delta_xo, delta_yo, knot_nrst_x, knot_nrst_y, knot_nrst_lin, BLS, NNI,
            knots_x_mesh, knots_y_mesh, tmask_good_knotNdata, mode) with dtypes (????). # FIX dtypes!
        t0 (float): Time of inferior conjunction.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        a (float): Semi-major axis (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        ecosw (float): Eccentricity multiplied by the cosine of the longitude of periastron (value between -1 and 1).
        esinw (float): Eccentricity multiplied by the sine of the longitude of periastron (value between -1 and 1).
        q1 (float): Limb darkening coefficient 1, parametrized to range between 0 and 1.
        q2 (float): Limb darkening coefficient 2, parametrized to range between 0 and 1.
        fp (float): Planet-to-star flux ratio.
        A (float): Amplitude of the first-order cosine term.
        B (float): Amplitude of the first-order sine term.
        C (float): Amplitude of the second-order cosine term. Default=0.
        D (float): Amplitude of the second-order sine term. Default=0.
        r2 (float): Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
        r2off (float): Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.
        d1 (float): The constant offset term. #FIX - I don't think this should be here.
        d2 (float): The slope in sensitivity with the PSF width in the x direction.
        d3 (float): The slope in sensitivity with the PSF width in the y direction.
        s1 (float): The amplitude of the heaviside step function.
        s2 (float): The location of the step in the heaviside function.
        m1 (float): The slope in sensitivity over time with respect to time[0].
        
    Returns:
        ndarray: The modelled flux variations due to the astrophysical model modified by the detector model.

    """
    
    time = signal_input[1]
    mode = signal_input[-1]
    
    astroModel = astro_models.ideal_lightcurve(time, t0, per, rp, a, inc, ecosw, esinw, q1, q2,
                                               fp, A, B, C, D, r2, r2off)
    
    model = astroModel*detec_model_bliss(signal_input, astroModel)
    
    if 'psfw' in mode.lower():
        psfwidths = signal_input[4:6]
        model *= detec_model_PSFW(psfwidths, d1, d2, d3)
    
    if 'hside' in mode.lower():
        model *= hside(time, s1, s2)
    
    if 'tslope' in mode.lower():
        model *= tslope(time, m1)
    
    return model

def signal_GP(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
              d1, d2, d3, s1, s2, m1,
              gpAmp, gpLx, gpLy, sigF,
              predictGp=True, returnGp=False):
    """Model the flux variations as a product of astrophysical varations multiplied by a GP detector sensitivity model.

    Args:
        signal_input (tuple): (flux, time, xdata, ydata, psfwx, psfwy, mode) with dtypes
            (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, string).
        t0 (float): Time of inferior conjunction.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        a (float): Semi-major axis (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        ecosw (float): Eccentricity multiplied by the cosine of the longitude of periastron (value between -1 and 1).
        esinw (float): Eccentricity multiplied by the sine of the longitude of periastron (value between -1 and 1).
        q1 (float): Limb darkening coefficient 1, parametrized to range between 0 and 1.
        q2 (float): Limb darkening coefficient 2, parametrized to range between 0 and 1.
        fp (float): Planet-to-star flux ratio.
        A (float): Amplitude of the first-order cosine term.
        B (float): Amplitude of the first-order sine term.
        C (float): Amplitude of the second-order cosine term. Default=0.
        D (float): Amplitude of the second-order sine term. Default=0.
        r2 (float): Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
        r2off (float): Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.
        d1 (float): The constant offset term. #FIX - I don't think this should be here.
        d2 (float): The slope in sensitivity with the PSF width in the x direction.
        d3 (float): The slope in sensitivity with the PSF width in the y direction.
        s1 (float): The amplitude of the heaviside step function.
        s2 (float): The location of the step in the heaviside function.
        m1 (float): The slope in sensitivity over time with respect to time[0].
        gpAmp (float): The natural logarithm of the GP covariance amplitude.
        gpLx (float): The natural logarithm of the GP covariance lengthscale in x.
        gpLy (float): The natural logarithm of the GP covariance lengthscale in y.
        sigF (float): The white noise in units of F_star.
        predictGp (bool, optional): Should the GP make predictions (True, default), or just return the GP (useful for lnlike).
        returnGp (bool, optional): Should the GP model return the GP object (True, useful for lnlike) or not (False, default).

    Returns:
        ndarray: The modelled flux variations due to the astrophysical model modified by the detector model.

    """
    
    #flux, time, xdata, ydata, psfwx, psfwy, mode = signal_input
    
    flux, time, xdata, ydata = signal_input[:4]
    mode = signal_input[-1]
    
    model = astro_models.ideal_lightcurve(time, t0, per, rp, a, inc,
                                          ecosw, esinw, q1, q2, fp,
                                          A, B, C, D, r2, r2off)
    
    if 'psfw' in mode.lower():
        psfwidths = signal_input[4:6]
        model *= detec_model_PSFW(psfwidths, d1, d2, d3)
    
    if 'hside' in mode.lower():
        model *= hside(time, s1, s2)
    
    if 'tslope' in mode.lower():
        model *= tslope(time, m1)
    
    if predictGp:
        detec_input = (flux, xdata, ydata, time, returnGp, model)
        returnVar = detec_model_GP(detec_input, gpAmp, gpLx, gpLy, sigF)
        if returnGp:
            detecModel, gp = returnVar
        else:
            detecModel = returnVar
        
        model *= detecModel
    else:
        if returnGp:
            gp = george.GP(np.exp(gpAmp)*george.kernels.ExpSquaredKernel(np.exp([gpLx, gpLy]), ndim=2, axes=[0, 1]))#,
#                            mean=mean_model)#, solver=george.HODLRSolver, tol=1e-8)
    
            gp.compute(np.array([xdata, ydata]).T, sigF)
    
    if returnGp:
        return model, gp
    else:
        return model


######################################################################################
#THESE ARE THE INDIVIDUAL DETECTOR MODEL FUNCTIONS WHICH MODEL JUST THE DETECTOR SYSTEMATICS

def detec_model_poly(detec_inputs, c1, c2, c3, c4, c5, c6, c7=0, c8=0, c9=0, c10=0, c11=0, 
                     c12=0, c13=0, c14=0, c15=0, c16=0, c17=0, c18=0, c19=0, c20=0, c21=0):
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

def detec_model_PSFW(input_data, d1=1, d2=0, d3=0):
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

def hside(time, s1, s2):
    """Model the detector systematics with a heaviside step function at one AOR break.

    Args:
        time (ndarray): The time.
        s1 (float): The amplitude of the heaviside step function.
        s2 (float): The location of the step in the heaviside function.

    Returns:
        ndarray: The flux variations due to the detector systematics.

    """
    x = time - s2
    return s1*np.heaviside(x, 0.0) + 1

def tslope(time, m1):
    """Model the detector systematics with a simple slope in time.

    Args:
        time (ndarray): The time.
        m1 (float): The slope in sensitivity over time with respect to time[0].

    Returns:
        ndarray: The flux variations due to the detector systematics.

    """
    return 1+(time-time[0])*m1

def detec_model_PLD(input_data, p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1,
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
    
    detec = np.array(detec)
    
    return np.dot(detec, pixels).reshape(-1)

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

def detec_model_GP(input_data, gpAmp, gpLx, gpLy, sigF):
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
    
    flux, xdata, ydata, time, returnGp, astroModel = input_data
    
    gp = george.GP(np.exp(gpAmp)*george.kernels.ExpSquaredKernel(np.exp([gpLx, gpLy]), ndim=2, axes=[0, 1]))#, solver=george.HODLRSolver, tol=1e-8)
    
    gp.compute(np.array([xdata, ydata]).T, sigF)
    
    mu, _ = gp.predict(flux-astroModel, np.array([xdata, ydata]).T)
    
    mu = 1.+mu/astroModel
    
    if returnGp:
        return mu, gp
    else:
        return mu
