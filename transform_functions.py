# tranform_functions

# Jeffrey Mays

import numpy as np
import math as ma

RTOD = 180.0 / np.pi
DTOR = np.pi / 180.0


def skew(a):

    # Skew semetric matrix
    #
    # INPUTS:
    #   a       3 by 1 vector of numbers
    #
    # OUTPUTS
    #   A       3 by 3 skew matrix

    A = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return A


def RPY(eul):

    # Euler angle Diff EQ
    #
    # INPUTS:
    #   eul     3 by 1 set of Euler angles (phi, theta, psi) (rad)
    #
    # OUTPUTS
    #   A       3 by 3 matrix for Euler angle integration

    sin_phi = np.sin(eul[0])
    cos_phi = np.cos(eul[0])
    sin_theta = np.sin(eul[1])
    cos_theta = np.cos(eul[1])

    A = np.array([[1.0, sin_phi*sin_theta, cos_phi*sin_theta],
                  [0.0, cos_phi*cos_theta, -sin_phi*cos_theta],
                  [0.0, sin_phi, cos_phi]])
    A = 1.0/cos_theta*A
    return A


def CTM_to_Euler(C):

    # Compute Euler angles from transform matrix
    #
    # INPUTS:
    #   C     3 by 3 matrix C_b_n
    #
    # OUTPUTS
    #   Euler    Euler angle vector (phi, theta, psi) (rad)

    C = C.T  # change input so these are correct
    phi = float(ma.atan2(C[1, 2], C[2, 2]))
    theta = float(-ma.asin(C[0, 2]))
    psi = float(ma.atan2(C[0, 1], C[0, 0]))
    Euler = np.zeros((1, 3))
    Euler = np.array([phi, theta, psi])
    return Euler


def Euler_to_CTM(eul):

    # Create Coordinate transform matrix
    #
    # INPUTS:
    #   eul     Euler angle vector (phi, theta, psi) (rad)
    #
    # OUTPUTS
    #   C.T     Body to Nav Rotation Matrix (no transpose is opposite)

    sin_phi = float(np.sin(eul[0]))
    cos_phi = float(np.cos(eul[0]))
    sin_theta = float(np.sin(eul[1]))
    cos_theta = float(np.cos(eul[1]))
    sin_psi = float(np.sin(eul[2]))
    cos_psi = float(np.cos(eul[2]))

    C = np.zeros((3, 3))
    C[0, 0] = cos_theta * cos_psi
    C[0, 1] = cos_theta * sin_psi
    C[0, 2] = -sin_theta
    C[1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
    C[1, 1] = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
    C[1, 2] = sin_phi * cos_theta
    C[2, 0] = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi
    C[2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi
    C[2, 2] = cos_phi * cos_theta
    return C.T  # body to NED


def LLA_to_NED(lat, long, alt, x_ECEF, x_ref):

    # LLA to NED using reference
    #
    # INPUTS:
    #   lat     Lattitude (rad)
    #   long    Longitude (rad)
    #   alt     altitude (m)
    #   x_ECEF  Position of interest in ECEF frame (m)
    #   x_ref   Reference ECEF position for NED (m)
    #
    # OUTPUTS
    #   x_NED   Position in NED frame based on x_ref

    R_NED_to_ECEF = np.array([[-np.sin(lat)*np.cos(long),
                               -np.sin(long),
                               -np.cos(lat)*np.cos(long)],
                              [-np.sin(lat)*np.sin(long),
                               np.cos(long),
                               -np.cos(lat)*np.sin(long)],
                              [np.cos(lat),
                               0.0,
                               -np.sin(lat)]])
    residual = np.array([[float(x_ECEF[0])-float(x_ref[0])],
                         [float(x_ECEF[1])-float(x_ref[1])],
                         [float(x_ECEF[2])-float(x_ref[2])]])
    x_NED = np.matmul(R_NED_to_ECEF.T, residual)
    return(x_NED)


def ECEF_to_LLA(r_eb_e):

    # ECEF to LLA frame
    #
    # INPUTS:
    #   r_eb_e  ECEF frame coordinate (m)
    #
    # OUTPUTS
    #   lat     Lattitude (rad)
    #   long    Longitude (rad)
    #   alt     altitude (m)

    f = 1.0/298.257223563
    R0 = 6378137.0
    e = np.sqrt(f*(2.0-f))
    R_P = R0*(1.0-f)
    long = ma.atan2(r_eb_e[1], r_eb_e[0])
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

    # LLA to ECEF frame
    #
    # INPUTS:
    #   lat     Lattitude (rad)
    #   long    Longitude (rad)
    #   alt     altitude (m)
    #
    # OUTPUTS
    #   x   ECEF x position (m)
    #   y   ECEF y position (m)
    #   z   ECEF z position (m)

    R0 = 6378137.0  # WGS84 Equatorial radius in meters
    f = 1/298.257223563
    e = np.sqrt(f*(2.0 - f))  # WGS84 eccentricity
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
    x = r_eb_e[0]
    y = r_eb_e[1]
    z = r_eb_e[2]
    return x, y, z

def ECEF_to_NED(x_ecef, latR, longR, alt):

    # ECEF to NED frame (Using differnt method)
    #
    # INPUTS:
    #   x_ecef  ECEF position of interest (m)
    #   lat     Lattitude (rad)
    #   long    Longitude (rad)
    #   alt     altitude (m)
    #
    # OUTPUTS
    #   NED     NED position based on LLA coordinates (m)

    x, y, z = LLA_to_ECEF(latR, longR, alt)
    ref_e = np.array([x, y, z])
    delta_E = np.zeros((3, 1))
    delta_E[0] = x_ecef[0] - ref_e[0]
    delta_E[1] = x_ecef[1] - ref_e[1]
    delta_E[2] = x_ecef[2] - ref_e[2]
    ENU = np.zeros((3, 1))
    ENU[0] = -np.sin(longR)*delta_E[0] + np.cos(longR)*delta_E[1]
    ENU[1] = -np.sin(latR)*np.cos(longR)*delta_E[0] - np.sin(latR)*np.sin(longR)*delta_E[1] + np.cos(latR)*delta_E[2]
    ENU[2] = -np.cos(latR)*np.cos(longR)*delta_E[0] + np.cos(latR)*np.sin(longR)*delta_E[1] + np.sin(latR)*delta_E[2]
    C = np.zeros((3, 3))
    C[0, 1] = 1.0   # ENU to NED Rotation Matrix
    C[1, 0] = 1.0
    C[2, 2] = -1.0
    NED = np.zeros((3, 1))
    NED = C@ENU
    return NED

def ECI_to_ECEF(xp, yp, Omega, i):

    # ECI to ECEF frame
    #
    # INPUTS:
    #   xp
    #   yp
    #   Omega
    #   i
    #
    # OUTPUTS
    #   r_eb_e  ECEF frame coordinate (m)

    x = xp*ma.cos(Omega) - yp*ma.cos(i)*ma.sin(Omega)
    y = xp*ma.sin(Omega) + yp*ma.cos(i)*ma.cos(Omega)
    z = yp*ma.sin(i)
    r_eb_e = np.array([x, y, z])
    return r_eb_e