# INS_GNSS_function

# Jeffrey Mays

import numpy as np

RTOD = 180.0 / np.pi
DTOR = np.pi / 180.0

def LLA_to_NED(lat, long, alt, x_ECEF, x_ref):

    # x_ref is reference ECEF frame
    R_NED_to_ECEF = np.array([[-np.sin(lat)*np.cos(long),
                               -np.sin(long),
                               -np.cos(lat)*np.cos(long)],
                              [-np.sin(lat)*np.sin(long),
                               np.cos(long),
                               -np.cos(lat)*np.sin(long)],
                              [np.cos(lat),
                               0.0,
                               -np.sin(lat)]])
    residual = np.array([[x_ECEF[0]-x_ref[0]],
                         [x_ECEF[1]-x_ref[1]],
                         [x_ECEF[2]-x_ref[2]]])
#    x_NED = R_NED_to_ECEF.T*(x_ECEF-x_ref)
    x_NED = np.matmul(R_NED_to_ECEF.T, residual)
    return(x_NED)


def ECEF_to_LLA(r_eb_e):
    
    # r_eb_e ECEF frame
    f = 1.0/298.257223563
    R0 = 6378137.0
    e = np.sqrt(f*(2.0-f))
    R_P = R0*(1.0-f)
    long = ma.atan2(r_eb_e[1], r_eb_e[0]) # longitude
    p = np.sqrt(r_eb_e[0]**2+r_eb_e[1]**2)
    E = np.sqrt(R0**2-R_P**2)
    F = 54.0*(R_P*r_eb_e[2])**2
    G = p**2+(1.0-e**2)*r_eb_e[2]**2-(e*E)**2
    c = e**4*F*p**2/(G**3)
    s = (1.0+c+np.sqrt(c**2+2*c))**(1.0/3.0)
    P = (F/(3*G**2))/((s+1.0/s+1.0)**2)
    Q = np.sqrt(1.0+2.0*e**4*P)
    k1 = (-P*e**2*p)/(1.0+Q)
    k2 = 0.5*R0**2*(1.0+1.0/Q)
    k3 = -P*(1.0-e**2)*(r_eb_e[2]**2)/(Q*(1.0+Q))
    k4 = -0.5*P*p**2
    k5 = p-e**2*(k1+np.sqrt(k2+k3+k4))
    U = np.sqrt(k5**2+r_eb_e[2]**2)
    V = np.sqrt(k5**2+(1.0-e**2)*r_eb_e[2]**2)
    alt = U*(1.0-R_P**2/(R0*V))
    z0 = (R_P**2*r_eb_e[2])/(R0*V)
    ep = R0*e/R_P
    lat = ma.atan((r_eb_e[2]+z0*ep**2)/p)
    return lat, long, alt


def LLA_to_ECEF(lat, long, alt):

    R0 = 6378137.0  # WGS84 Equatorial radius in meters
    e = 0.0818191908425  # WGS84 eccentricity
    # calculate transverse radius of curvature (2.105)
    R_E = R0 / np.sqrt(1.0-(e*np.sin(lat))**2)
    # Convert position using (2.112)
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_long = np.cos(long)
    sin_long = np.sin(long)
    r_eb_e = np.array([[(R_E + alt) * cos_lat * cos_long],
                       [(R_E + alt) * cos_lat * sin_long],
                       [((1.0 - e**2) * R_E + alt) * sin_lat]])
    return r_eb_e


def ECI_to_ECEF(xp, yp, Omega, i):
    x = xp*ma.cos(Omega) - yp*ma.cos(i)*ma.sin(Omega)
    y = xp*ma.sin(Omega) + yp*ma.cos(i)*ma.cos(Omega)
    z = yp*ma.sin(i)
    r_eb_e = np.array([x, y, z])
    return r_eb_e