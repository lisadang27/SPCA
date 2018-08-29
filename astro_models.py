import numpy as np
import batman

def transit_model(time, t0, per, rp, a, inc, ecc, w, u1, u2):
    '''
    Get a model transit lightcurve.

    Parameters
    ----------

    :type time : ndarray
    :param time: Array of times at which to calculate the model.

    :type t0 : float
    :param t0: Time of inferior conjunction

    :type per : float
    :param per: Orbital period

    :type rp : float
    :param rp: Planet radius (in units of stellar radii)

    :type a : float
    :param a: Semi-major axis (in units of stellar radii)

    :type inc : float
    :param inc: Orbital inclination (in degrees)

    :type ecc : float
    :param ecc: Eccentricity

    :type w : float
    :param w: Longitude of periastron (in degrees)

    :type u1 : float
    :param u1: Limb darkening coefficient

    :type u2 : float
    :param u2: Limb darkening coefficient

    Returns
    -------

    :return: flux, t_secondary, anom - (ndarray, float, ndarray) - Model transit lightcurve,
                                                                   time of secondary eclipse,
                                                                   and true anomaly
    '''
    
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
    '''
    Get a model secondary eclipse lightcurve.

    Parameters
    ----------

    :type time : ndarray
    :param time: Array of times at which to calculate the model.

    :type t0 : float
    :param t0: Time of inferior conjunction

    :type per : float
    :param per: Orbital period

    :type rp : float
    :param rp: Planet radius (in units of stellar radii)

    :type a : float
    :param a: Semi-major axis (in units of stellar radii)

    :type inc : float
    :param inc: Orbital inclination (in degrees)

    :type ecc : float
    :param ecc: Eccentricity

    :type w : float
    :param w: Longitude of periastron (in degrees)
    
    :type fp : float
    :param fp: Planet-to-star flux ratio
    
    :type t_sec : float
    :param t_sec: Time of secondary eclipse

    Returns
    -------

    :return: flux - (ndarray) - Model eclipse light curve
    '''
    
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
    '''
    Model the variations in projected area of a bi-axial ellipsoid over time, assuming elongated axis is the sub-stellar axis.

    Parameters
    ----------

    :type time : ndarray
    :param time: Array of times at which to calculate the model.
    
    :type t_sec : float
    :param t_sec: Time of secondary eclipse

    :type per : float
    :param per: Orbital period

    :type rp : float
    :param rp: Planet radius along dawn-dusk axis (in units of stellar radii)

    :type inc : float
    :param inc: Orbital inclination (in degrees)

    :type r2 : float
    :param r2: Planet radius along sub-stellar axis (in units of stellar radii)

    Returns
    -------

    :return: areas - (ndarray) - Modelled projected area of the ellipsoid over time
    '''
    
    t = time - t_sec
    orbFreq = 2*np.pi/per
    #calculate the orbital phase (assumes the planet is tidally locked)
    phi = (orbFreq*t-np.pi)%(2*np.pi)
    #convert inclination to radians for numpy
    inc = inc*np.pi/180
    
    return np.pi*np.sqrt(rp**2*np.sin(inc)**2*(r2**2*np.sin(phi)**2 + rp**2*np.cos(phi)**2)
                         + rp**2*r2**2*np.cos(inc)**2)/(np.pi*rp**2)



def area(time, t_sec, per, rp, inc, r2, r2off):
    '''
    Model the variations in projected area of a bi-axial ellipsoid over time, without assuming elongated axis is the sub-stellar axis.

    Parameters
    ----------

    :type time : ndarray
    :param time: Array of times at which to calculate the model.
    
    :type t_sec : float
    :param t_sec: Time of secondary eclipse

    :type per : float
    :param per: Orbital period

    :type rp : float
    :param rp: Planet radius along dawn-dusk axis (in units of stellar radii)

    :type inc : float
    :param inc: Orbital inclination (in degrees)

    :type r2 : float
    :param r2: Planet radius along sub-stellar axis (in units of stellar radii)
    
    :type r2off : float
    :param r2off: Angle to the elongated axis with respect to the sub-stellar axis (in degrees)

    Returns
    -------

    :return: areas - (ndarray) - Modelled projected area of the ellipsoid over time
    '''
    
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
    '''
    Model first- or second-order sinusoidal phase variations.

    Parameters
    ----------

    :type time : ndarray
    :param time: Array of times at which to calculate the model.
    
    :type t_sec : float
    :param t_sec: Time of secondary eclipse.

    :type per : float
    :param per: Orbital period.

    :type anom : ndarray
    :param anom: The true anomaly over time.
    
    :type ecc : float
    :param ecc: Eccentricity

    :type w : float
    :param w: Longitude of periastron (in degrees).

    :type A : float
    :param A: Amplitude of the first-order cosine term.
    
    :type B : float
    :param B: Amplitude of the first-order sine term.
    
    :type C : float, optional
    :param C: Amplitude of the second-order cosine term. Default=0.
    
    :type D : float, optional
    :param D: Amplitude of the second-order sine term. Default=0.

    Returns
    -------

    :return: phaseVars - (ndarray) - Modelled phase variations
    '''
    
    #calculate the orbital phase
    if ecc == 0:
        #the planet is on a circular orbit
        t    = time - t_sec
        freq = 2*np.pi/per
        phi  = (freq*t)
    else:
        #the planet is on an eccentric orbit
        phi  = anom + np.deg2rad(w) + np.pi/2
    
    #calculate the phase variations
    if C==0 and D==0:
        #Skip multiplying by a bunch of zeros to speed up fitting
        phaseVars = 1 + A*(np.cos(phi)-1) + B*np.sin(phi)
    else:
        phaseVars = 1 + A*(np.cos(phi)-1) + B*np.sin(phi) + C*(np.cos(2*phi)-1) + D*np.sin(2*phi)
    
    return phaseVars




def fplanet_model(time, anom, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec, A, B, C=0, D=0, r2=None, r2off=None):
    '''
    Model observed flux coming from the planet over time.

    Parameters
    ----------

    :type time : ndarray
    :param time: Array of times at which to calculate the model.
    
    :type anom : ndarray
    :param anom: The true anomaly over time.
    
    :type t0 : float
    :param t0: Time of inferior conjunction

    :type per : float
    :param per: Orbital period

    :type rp : float
    :param rp: Planet radius (in units of stellar radii)

    :type a : float
    :param a: Semi-major axis (in units of stellar radii)

    :type inc : float
    :param inc: Orbital inclination (in degrees)

    :type ecc : float
    :param ecc: Eccentricity

    :type w : float
    :param w: Longitude of periastron (in degrees)

    :type u1 : float
    :param u1: Limb darkening coefficient

    :type u2 : float
    :param u2: Limb darkening coefficient
    
    :type fp : float
    :param fp: Planet-to-star flux ratio
    
    :type t_sec : float
    :param t_sec: Time of secondary eclipse.
    
    :type A : float
    :param A: Amplitude of the first-order cosine term.
    
    :type B : float
    :param B: Amplitude of the first-order sine term.
    
    :type C : float, optional
    :param C: Amplitude of the second-order cosine term. Default=0.
    
    :type D : float, optional
    :param D: Amplitude of the second-order sine term. Default=0.
    
    :type r2 : float, optional
    :param r2: Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
    
    :type r2off : float, optional
    :param r2off: Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.

    Returns
    -------

    :return: fplanet - (ndarray) - Observed flux coming from planet over time.
    '''
    
    phase = phase_variation(time, t_sec, per, anom, ecc, w, A, B, C, D)
    eclip = eclipse(time, t0, per, rp, a, inc, ecc, w, fp, t_sec)
    
    fplanet = phase*(eclip - 1)
    if r2 != rp and r2 != None:
        fplanet *= area(time, t_sec, per, rp, inc, r2, r2off)
    
    return fplanet



def ideal_lightcurve(time, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C=0, D=0, r2=None, r2off=None):
    '''
    Model observed flux coming from the star+planet system over time.

    Parameters
    ----------

    :type time : ndarray
    :param time: Array of times at which to calculate the model.
        
    :type t0 : float
    :param t0: Time of inferior conjunction

    :type per : float
    :param per: Orbital period

    :type rp : float
    :param rp: Planet radius (in units of stellar radii)

    :type a : float
    :param a: Semi-major axis (in units of stellar radii)

    :type inc : float
    :param inc: Orbital inclination (in degrees)

    :type ecosw : float
    :param ecc: Eccentricity multiplied by the cosine of the longitude of periastron (value between -1 and 1)

    :type esinw : float
    :param w: Eccentricity multiplied by the sine of the longitude of periastron (value between -1 and 1)

    :type q1 : float
    :param q1: Limb darkening coefficient, parametrized to range between 0 and 1

    :type q2 : float
    :param q2: Limb darkening coefficient, parametrized to range between 0 and 1
    
    :type fp : float
    :param fp: Planet-to-star flux ratio
    
    :type A : float
    :param A: Amplitude of the first-order cosine term.
    
    :type B : float
    :param B: Amplitude of the first-order sine term.
    
    :type C : float, optional
    :param C: Amplitude of the second-order cosine term. Default=0.
    
    :type D : float, optional
    :param D: Amplitude of the second-order sine term. Default=0.
    
    :type r2 : float, optional
    :param r2: Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
    
    :type r2off : float, optional
    :param r2off: Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.

    Returns
    -------

    :return: lightcurve - (ndarray) - Observed flux coming from star+planet system over time.
    '''
    
    if ecosw==0 and esinw==0:
        ecc = 0
        w = 0
    else:
        ecc = np.sqrt(ecosw**2 + esinw**2)
        #longitude of periastron needs to be in degrees for batman!
        w   = np.rad2deg(np.arctan2(esinw, ecosw))
    
    #convert q1 and q2 limb darkening parameterization to u1 and u2 used by batman
    u1  = 2*np.sqrt(q1)*q2
    u2  = np.sqrt(q1)*(1-2*q2)
    
    transit, t_sec, anom = transit_model(time, t0, per, rp, a, inc, ecc, w, u1, u2)
    fplanet = fplanet_model(time, anom, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec, A, B, C, D, r2, r2off)
    
    return transit + fplanet




def check_phase(phis, A, B, C=0, D=0):
    '''
    Check if the phasecurve ever dips below zero, implying non-physical negative flux coming from the planet.

    Parameters
    ----------

    :type phis : ndarray
    :param phis: Array of phases at which to calculate the model, e.g. phis=np.linspace(-np.pi,np.pi,1000).
    
    :type A : float
    :param A: Amplitude of the first-order cosine term.
    
    :type B : float
    :param B: Amplitude of the first-order sine term.
    
    :type C : float, optional
    :param C: Amplitude of the second-order cosine term. Default=0.
    
    :type D : float, optional
    :param D: Amplitude of the second-order sine term. Default=0.
    
    Returns
    -------

    :return: lightcurveBad - (bool) - True if lightcurve implies non-physical negative flux coming from the planet.
    '''
    
    if C==0 and D==0:
        #avoid wasting time by multiplying by a bunch of zeros
        phase = 1 + A*(np.cos(phis)-1) + B*np.sin(phis)
    else: 
        phase = 1 + A*(np.cos(phis)-1) + B*np.sin(phis) + C*(np.cos(2*phis)-1) + D*np.sin(2*phis)
    
    return np.any(phase < 0)
