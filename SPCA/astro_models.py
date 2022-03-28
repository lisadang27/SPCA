import numpy as np
import batman

def transit_model(time, t0, per, rp, a, inc, ecc, w, u1, u2):
    """Get a model transit lightcurve.

    Args:
        time (ndarray): Array of times at which to calculate the model.
        t0 (float): Time of inferior conjunction.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        a (float): Semi-major axis (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        ecc (float): Orbital eccentricity.
        w (float): Longitude of periastron (in degrees).
        u1 (float): Limb darkening coefficient 1.
        u2 (float): Limb darkening coefficient 2.

    Returns:
        tuple: flux, t_secondary, anom (ndarray, float, ndarray).
            Model transit lightcurve, time of secondary eclipse,
    
    """
    
    #make object to store transit parameters
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.limb_dark = "quadratic"
    params.u = [u1, u2]

    #initialize model
    m = batman.TransitModel(params, time)
    flux = m.light_curve(params)
    t_secondary = m.get_t_secondary(params)
    # anom is in radians!
    anom       = m.get_true_anomaly()

    return flux, t_secondary, anom



def eclipse(time, t0, per, rp, a, inc, ecc, w, fp, t_sec):
    """Get a model secondary eclipse lightcurve.

    Args:
        time (ndarray): Array of times at which to calculate the model.
        t0 (float): Time of inferior conjunction.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        a (float): Semi-major axis (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        ecc (float): Orbital eccentricity.
        w (float): Longitude of periastron (in degrees).
        fp (float): Planet-to-star flux ratio.
        t_sec (float): Time of secondary eclipse.

    Returns:
        ndarray: flux. The model eclipse light curve.
    
    """
    
    #make object to store transit parameters
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.limb_dark = "uniform"
    params.u = []
    params.fp = fp
    params.t_secondary = t_sec
    
    #initialize model
    m = batman.TransitModel(params, time, transittype="secondary")
    flux = m.light_curve(params)

    return flux



def area_noOffset(time, t_sec, per, rp, inc, r2):
    """Model the variations in projected area of a bi-axial ellipsoid over time, assuming elongated axis is the sub-stellar axis.

    Args:
        time (ndarray): Array of times at which to calculate the model.
        t_sec (float): Time of secondary eclipse.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        r2 (float): Planet radius along sub-stellar axis (in units of stellar radii).

    Returns:
        ndarray: Modelled projected area of the ellipsoid over time.
    
    """
    
    t = time - t_sec
    orbFreq = 2*np.pi/per
    #calculate the orbital phase (assumes the planet is tidally locked)
    phi = (orbFreq*t-np.pi)%(2*np.pi)
    #convert inclination to radians for numpy
    inc = inc*np.pi/180
    
    return np.pi*np.sqrt(rp**2*np.sin(inc)**2*(r2**2*np.sin(phi)**2 + rp**2*np.cos(phi)**2)
                         + rp**2*r2**2*np.cos(inc)**2)/(np.pi*rp**2)



def area(time, t_sec, per, rp, inc, r2, r2off):
    """Model the variations in projected area of a bi-axial ellipsoid over time, without assuming elongated axis is the sub-stellar axis.

    Args:
        time (ndarray): Array of times at which to calculate the model.
        t_sec (float): Time of secondary eclipse.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        r2 (float): Planet radius along sub-stellar axis (in units of stellar radii).
        r2off (float): Angle to the elongated axis with respect to the sub-stellar axis (in degrees).

    Returns:
        ndarray: Modelled projected area of the ellipsoid over time.
    
    """
    
    t = time - t_sec
    orbFreq = 2*np.pi/per
    #calculate the orbital phase (assumes the planet is tidally locked)
    phi = (orbFreq*t-np.pi)%(2*np.pi)
    #effectively rotate the elongated axis by changing phi
    phi -= r2off*np.pi/180
    #convert inclination to radians for numpy
    inc = inc*np.pi/180
    
    return np.pi*np.sqrt(rp**2*np.sin(inc)**2*(r2**2*np.sin(phi)**2 + rp**2*np.cos(phi)**2)
                         + rp**2*r2**2*np.cos(inc)**2)/(np.pi*rp**2)


def phase_variation(time, t_sec, per, anom, ecc, w, A, B, C=0, D=0):
    """Model first- or second-order sinusoidal phase variations.

    Args:
        time (ndarray): Array of times at which to calculate the model.
        t_sec (float): Time of secondary eclipse.
        per (float): Orbital period.
        anom (ndarray): The true anomaly over time.
        ecc (float): Orbital eccentricity.
        w (float): Longitude of periastron (in degrees).
        A (float): Amplitude of the first-order cosine term.
        B (float): Amplitude of the first-order sine term.
        C (float, optional): Amplitude of the second-order cosine term. Default=0.
        D (float, optional): Amplitude of the second-order sine term. Default=0.

    Returns:
        ndarray: Modelled phase variations.
    
    """
    
    #calculate the orbital phase
    if ecc == 0.:
        #the planet is on a circular orbit
        t    = time - t_sec
        freq = 2.*np.pi/per
        phi  = (freq*t)
    else:
        #the planet is on an eccentric orbit
        phi  = anom + w*np.pi/180. + np.pi/2.
    
    #calculate the phase variations
    if C==0. and D==0.:
        #Skip multiplying by a bunch of zeros to speed up fitting
        phaseVars = 1. + A*(np.cos(phi)-1.) + B*np.sin(phi)
    else:
        phaseVars = 1. + A*(np.cos(phi)-1.) + B*np.sin(phi) + C*(np.cos(2.*phi)-1.) + D*np.sin(2.*phi)
    
    return phaseVars




def fplanet_model(time, anom, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec, A, B, C=0., D=0., r2=None, r2off=None):
    """Model observed flux coming from the planet over time.

    Args:
        time (ndarray): Array of times at which to calculate the model.
        anom (ndarray): The true anomaly over time.
        t0 (float): Time of inferior conjunction.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        a (float): Semi-major axis (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        ecc (float): Orbital eccentricity.
        w (float): Longitude of periastron (in degrees).
        u1 (float): Limb darkening coefficient 1.
        u2 (float): Limb darkening coefficient 2.
        fp (float): Planet-to-star flux ratio.
        t_sec (float): Time of secondary eclipse.
        A (float): Amplitude of the first-order cosine term.
        B (float): Amplitude of the first-order sine term.
        C (float, optional): Amplitude of the second-order cosine term. Default=0.
        D (float, optional): Amplitude of the second-order sine term. Default=0.
        r2 (float, optional): Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
        r2off (float, optional): Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.

    Returns:
        ndarray: Observed flux coming from planet over time.
    
    """
    
    fplanet = phase_variation(time, t_sec, per, anom, ecc, w, A, B, C, D)
    
    if r2!=rp and r2!=None:
        #if you are using r2off or non-90 inclination, the transit radius isn't going to be rp anymore.
        #rp will just be the minimum physical radius
        rEcl = rp*area(t_sec, t_sec, per, rp, inc, r2, r2off)
        fplanet *= area(time, t_sec, per, rp, inc, r2, r2off)
    else:
        rEcl = rp
    
    eclip = eclipse(time, t0, per, rEcl, a, inc, ecc, w, fp, t_sec)
    fplanet *= (eclip - 1)
    
    return fplanet


def ideal_lightcurve(time, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C=0, D=0, r2=None, r2off=None):
    """Model observed flux coming from the star+planet system over time.

    Args:
        time (ndarray): Array of times at which to calculate the model.
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
        C (float, optional): Amplitude of the second-order cosine term. Default=0.
        D (float, optional): Amplitude of the second-order sine term. Default=0.
        r2 (float, optional): Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
        r2off (float, optional): Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.

    Returns:
        ndarray: Observed flux coming from star+planet system over time.
    
    """
    
    if ecosw==0 and esinw==0:
        ecc = 0
        w = 0
    else:
        ecc = np.sqrt(ecosw**2 + esinw**2)
        #longitude of periastron needs to be in degrees for batman!
        w   = np.arctan2(esinw, ecosw)*180./np.pi
    
    #convert q1 and q2 limb darkening parameterization to u1 and u2 used by batman
    u1  = 2*np.sqrt(q1)*q2
    u2  = np.sqrt(q1)*(1-2*q2)
    
    if r2!=rp and r2!=None:
        #if you are using r2off or non-90 inclination, the transit radius isn't going to be rp anymore.
        #rp will just be the minimum physical radius
        rTrans = rp*area(t0, t0, per, rp, inc, r2, r2off)
    else:
        rTrans = rp
    
    transit, t_sec, anom = transit_model(time, t0, per, rTrans, a, inc, ecc, w, u1, u2)
    fplanet = fplanet_model(time, anom, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec, A, B, C, D, r2, r2off)
    
    return transit + fplanet



def check_phase(checkPhasePhis, A, B, C=0, D=0):
    """Check if the phasecurve ever dips below zero, implying non-physical negative flux coming from the planet.

    Args:
        phis (ndarray): Array of phases in radians at which to calculate the model, e.g. phis=np.linspace(-np.pi,np.pi,1000).
        A (float): Amplitude of the first-order cosine term.
        B (float): Amplitude of the first-order sine term.
        C (float, optional): Amplitude of the second-order cosine term. Default=0.
        D (float, optional): Amplitude of the second-order sine term. Default=0.

    Returns:
        bool: True if lightcurve implies non-physical negative flux coming from the planet, False otherwise.
    
    """
    toggle = False
    if toggle:
        if not (-90 < np.arctan2(B, A)*180/np.pi < 90):
            return -np.inf
    
        if C==0 and D==0:
            #avoid wasting time by multiplying by a bunch of zeros
            negative = np.any(1 + A*(np.cos(checkPhasePhis)-1) + B*np.sin(checkPhasePhis) < 0)
        else: 
            negative =  np.any(1 + A*(np.cos(checkPhasePhis)-1) + B*np.sin(checkPhasePhis)
                            + C*(np.cos(2*checkPhasePhis)-1) + D*np.sin(2*checkPhasePhis) < 0)
        
        if negative:
            return -np.inf
        else:
            return 0
    else:
        return 0
