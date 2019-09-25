import numpy as np
import george
from george.modeling import Model

import astro_models

class gpAstroModel(Model):
    def __init__(self, astrofunc, p0_astro):
        self.p0_astro = p0_astro
        self.astrofunc = astrofunc
    
    def get_value(self, inputs):
        xdata, ydata, time = inputs.T
        return self.astrofunc(time, *self.p0_astro)



def detec_model_poly(detec_inputs, c1, c2, c3, c4, c5, c6, c7=0, c8=0, c9=0, c10=0, c11=0, 
                     c12=0, c13=0, c14=0, c15=0, c16=0, c17=0, c18=0, c19=0, c20=0, c21=0):
    
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
    px, py = input_data
    pw     = np.vstack((np.ones_like(px), px, py))
    syst   = np.array([d1, d2, d3])
    return np.dot(syst[np.newaxis,:], pw).reshape(-1)

def hside(time, s1, s2):
    x = time - s2
    return s1*np.heaviside(x, 0.0) + 1

def tslope(time, m1):
    return 1+(time-time[0])*m1

def detec_model_bliss(signal_input, astroModel): 
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
    '''
    Output:
        detec  = BLISS/NNI sensitivity values at each knots
    '''
    return detec

def detec_model_GP_simpler(input_data, gpAmp, gpLx, gpLy, sigF):
    
    flux, xdata, ydata, time, returnGp, astroModel = input_data
    
    gp = george.GP(np.exp(gpAmp)*george.kernels.ExpSquaredKernel(np.exp([gpLx, gpLy]), ndim=2, axes=[0, 1]))#, solver=george.HODLRSolver, tol=1e-8)
    
    gp.compute(np.array([xdata, ydata]).T, sigF)
    
    mu, _ = gp.predict(flux-astroModel, np.array([xdata, ydata]).T)
    
    mu = 1.+mu/astroModel
    
    if returnGp:
        return mu, gp
    else:
        return mu
    
def detec_model_GP(input_data, gpAmp, gpLx, gpLy, sigF):
    
    flux, xdata, ydata, time, returnGp, p0_astro, astrofunc = input_data
    
    mean_model = gpAstroModel(astrofunc, p0_astro)
    
    gp = george.GP(np.exp(gpAmp)*george.kernels.ExpSquaredKernel(np.exp([gpLx, gpLy]), ndim=3, axes=[0, 1]),
                   mean=mean_model)#, solver=george.HODLRSolver, tol=1e-8)
    
    gp.compute(np.array([xdata, ydata, time]).T, sigF)
    
    mu, _ = gp.predict(flux, np.array([xdata, ydata, time]).T)
    
    mu /= astrofunc(time, *p0_astro)
    
    if returnGp:
        return mu, gp
    else:
        return mu

######################################################################################
#THIS IS THE MAIN SIGNAL FUNCTION WHICH BRANCHES OUT TO THE CORRECT SIGNAL FUNCTION
def signal(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                c1,  c2,  c3,  c4,  c5,  c6, c7,  c8,  c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, 
                d1, d2, d3, s1, s2, m1, 
                gpAmp, gpLx, gpLy, sigF,
                predictGp=True, returnGp=False):

    mode = signal_input[-1]
    if 'poly' in mode.lower():
        return signal_poly(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                           c1,  c2,  c3,  c4,  c5,  c6,  c7,  c8,  c9, c10, c11, c12, c13, c14, c15, c16, c17,
                           c18, c19, c20, c21, d1,  d2,  d3,  s1,  s2, m1, sigF)
    elif 'bliss' in mode.lower():
        return signal_bliss(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                            d1, d2, d3, s1, s2, m1, sigF)
    elif 'gp' in mode.lower():
        return signal_GP(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                         d1, d2, d3, s1, s2, m1,
                         gpAmp, gpLx, gpLy, sigF,
                         predictGp, returnGp)

######################################################################################

def signal_poly(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                c1,  c2,  c3,  c4,  c5,  c6, c7,  c8,  c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, 
                d1, d2, d3, s1, s2, m1):
    
    #flux, time, xdata, ydata, psfwx, psfwy, fluxerr, mode = signal_input
    
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

def signal_bliss(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off, 
                 d1, d2, d3, s1, s2, m1):
    
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

def signal_GP_simpler(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
              d1, d2, d3, s1, s2, m1,
              gpAmp, gpLx, gpLy, sigF,
              predictGp=True, returnGp=False):
    
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
    
def signal_GP(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
              d1, d2, d3, s1, s2, m1,
              gpAmp, gpLx, gpLy, sigF,
              predictGp=True, returnGp=False):
    
    #flux, time, xdata, ydata, psfwx, psfwy, mode = signal_input
    
    flux, time, xdata, ydata = signal_input[:4]
    mode = signal_input[-1]
    
    model = 1.
    
    if 'psfw' in mode.lower():
        psfwidths = signal_input[4:6]
        model *= detec_model_PSFW(psfwidths, d1, d2, d3)
    
    if 'hside' in mode.lower():
        model *= hside(time, s1, s2)
    
    if 'tslope' in mode.lower():
        model *= tslope(time, m1)
        
    astrofunc = lambda time, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off: \
                    model*astro_models.ideal_lightcurve(time, t0, per, rp, a, inc,
                                                        ecosw, esinw, q1, q2, fp,
                                                        A, B, C, D, r2, r2off)
    
    p0_astro = np.array([t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off])
    
    if predictGp:
        detec_input = (flux, xdata, ydata, time, returnGp, p0_astro, astrofunc)
        returnVar = detec_model_GP(detec_input, gpAmp, gpLx, gpLy, sigF)
        if returnGp:
            detecModel, gp = returnVar
        else:
            detecModel = returnVar
        
        model = astrofunc(time, p0_astro)*detecModel
    else:
        if returnGp:
            mean_model = gpAstroModel(astrofunc, p0_astro)
            
            gp = george.GP(np.exp(gpAmp)*george.kernels.ExpSquaredKernel(np.exp([gpLx, gpLy]), ndim=3, axes=[0, 1]),
                           mean=mean_model)#, solver=george.HODLRSolver, tol=1e-8)
    
            gp.compute(np.array([xdata, ydata, time]).T, sigF)
    
    if returnGp:
        return model, gp
    else:
        return model
    