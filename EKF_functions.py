# EKF Functions

# Jeffrey Mays

import numpy as np
import scipy.linalg
import transform_functions as trans

RTOD = 180.0 / np.pi
DTOR = np.pi / 180.0
a = 6378137.0  # Earth's radius (m)
f = 1.0/298.257223563
e = np.sqrt(f*(2.0-f))
mu = 3.986005E14  # m^3/s^2
omega_ie = 7.2921151467E-05  # Earth ROT rate (rad/s)


def INS_Equations_NED(tor_i, old_latR, old_longR, old_alt, old_v_eb_n,
                      old_C_b_n, meas_omega_ib_b, meas_f_ib_b):

    # INS update Equations from k-1 to k
    #
    # INPUTS:
    #   tor_i               Delta T (In this instant, INS Hz Rate) (sec)
    #   old_latR            Previous estimate Latitude (rad)
    #   old_longR           Previous estimate Longtiude (rad)
    #   old_alt             Previous estimate altitude (m)
    #   old_v_eb_n          Previous estimate velocity (m/sec)
    #   old_C_b_n           Previous estimate DCM Matrix
    #   meas_omega_ib_b     Previous body angular velocity (rad/sec)
    #   meas_f_ib_b         Previous body acceleration (m/sec^2)
    #
    # OUTPUTS
    #   new_C_b_n       INS estimate DCM Matrix
    #   new_v_eb_n      INS estimate velocity (m/sec)
    #   new_latR        INS estimate Latitude (rad)
    #   new_longR       INS estimate Longtiude (rad)
    #   new_alt         INS estimate altitude (m)

    # Gravity accel in NED frame
    dum1 = (1.0 + 0.0019311853*(np.sin(old_latR))**2.0)
    g0 = 9.7803253359/np.sqrt(1-f*(2.0-f)*(np.sin(old_latR))**2.0)*dum1
    dum2 = 3.0*(old_alt/a)**2.0
    ch = 1.0-2.0*(1.0+f+(a**3.0*(1-f)*omega_ie**2.0)/(mu))*(old_alt/a)+dum2
    g_n = np.array([0.0, 0.0, ch*g0])

    # Earth rotation rate expressed in navigation frame
    omega_ie_n = omega_ie*np.array([np.cos(old_latR), 0.0, -np.sin(old_latR)])

    # Transport Rate
    RN = a*(1.0-e**2)/(1.0-e**2.0*(np.sin(old_latR))**2.0)**1.5
    RE = a/np.sqrt(1.0-e**2.0*(np.sin(old_latR))**2.0)
    dum4 = float(old_v_eb_n[1]/(RE+old_alt))
    dum5 = float(-old_v_eb_n[0]/(RN+old_alt))
    dum6 = float(-old_v_eb_n[1]*np.tan(old_latR)/(RE+old_alt))
    omega_en_n = np.array([dum4, dum5, dum6])

    # Attitude Update
    omega_in_b = old_C_b_n.T@(omega_ie_n + omega_en_n)
    meas_omega_nb_b = meas_omega_ib_b - omega_in_b
    eulk_1 = trans.CTM_to_Euler(old_C_b_n)
    Ak_1 = trans.RPY(eulk_1)
    eulk = eulk_1 + tor_i * Ak_1 @ meas_omega_nb_b
    new_C_b_n = trans.Euler_to_CTM(eulk)

    # Velocity Update
    meas_f_ib_n = old_C_b_n@meas_f_ib_b
    dum3 = 2.0*omega_ie_n-omega_en_n
    a_eb_n = meas_f_ib_n + g_n - trans.skew(dum3)@old_v_eb_n
    new_v_eb_n = old_v_eb_n + tor_i*a_eb_n
    new_v_eb_n[2] = old_v_eb_n[2]

    # Position Update
    old_P_lla = np.array([old_latR, old_longR, old_alt])
    T = np.array([[1.0/(RN+old_alt), 0.0, 0.0],
                  [0.0, 1.0/((RE+old_alt)*np.cos(old_latR)), 0.0],
                  [0.0, 0.0, -1.0]])
    new_P_lla = old_P_lla + tor_i*T@old_v_eb_n
    new_latR = new_P_lla[0]
    new_longR = new_P_lla[1]
    new_alt = old_alt
    return new_C_b_n, new_v_eb_n, new_latR, new_longR, new_alt


def Get_dem_EKF_matrices(KF_param, tor_s, old_latR, old_alt, old_v_eb_n,
                         old_C_b_n, meas_f_ib_b):

    # Retreive EKF Matrices for current epoch k
    #
    # INPUTS:
    #   KF_Param        Kalman Filter parameters
    #   tor_s           Delta T (In this instant, INS Hz Rate) (sec)
    #   old_latR        Previous estimate Latitude (rad)
    #   old_alt         Previous estimate altitude (m)
    #   old_v_eb_n      Previous estimate velocity (m/sec)
    #   old_C_b_n       Previous estimate DCM Matrix
    #   meas_f_ib_b     Previous body acceleration (m/sec^2)
    #
    # OUTPUTS
    #   F       Discretized linearized INS error equations
    #   Q       Noise Covariance Matrix
    #   H       Measurement Matrix
    #   R       Measurement Covariance Matrix

    # Obtain the EKF Matrices based on previous state
    A = np.zeros((15, 15))
    F = np.zeros((15, 15))
    M = np.zeros((15, 12))
    Q = np.zeros((15, 15))
    U = np.zeros((12, 12))
    H = np.zeros((6, 15))
    H[0:6, 0:6] = np.identity(6)
    R = np.zeros((6, 6))

    # Earth rotation rate expressed in navigation frame
    omega_ie_n = omega_ie*np.array([np.cos(old_latR), 0.0, -np.sin(old_latR)])

    # Transport Rate
    RN = a*(1.0-e**2)/(1.0-e**2.0*(np.sin(old_latR))**2.0)**1.5
    RE = a/np.sqrt(1.0-e**2.0*(np.sin(old_latR))**2.0)
    dum4 = float(old_v_eb_n[1]/(RE+old_alt))
    dum5 = float(-old_v_eb_n[0]/(RN+old_alt))
    dum6 = float(-old_v_eb_n[1]*np.tan(old_latR)/(RE+old_alt))
    omega_en_n = np.array([dum4, dum5, dum6])

    # Gravity accel in NED frame
    dum1 = (1.0 + 0.0019311853*(np.sin(old_latR))**2.0)
    g0 = 9.7803253359/np.sqrt(1-f*(2.0-f)*(np.sin(old_latR))**2.0)*dum1
    dum2 = 3.0*(old_alt/a)**2.0
    ch = 1.0-2.0*(1.0+f+(a**3.0*(1-f)*omega_ie**2.0)/(mu))*(old_alt/a)+dum2
    g_n = np.array([0.0, 0.0, ch*g0])

    # State Matrix (Continuous)
    A[0:3, 0:3] = -trans.skew(omega_en_n)
    A[3:6, 0:3] = g_n[2]/a*np.array([[-1.0, 0.0, 0.0],
                                     [0.0, -1.0, 0.0],
                                     [0.0, 0.0, 2.0]])
    A[0:3, 3:6] = np.identity(3)
    A[3:6, 3:6] = -trans.skew(2.0*omega_ie_n+omega_en_n)
    A[3:6, 6:9] = trans.skew(old_C_b_n@meas_f_ib_b)
    A[6:9, 6:9] = -trans.skew(omega_ie_n + omega_en_n)
    A[3:6, 9:12] = old_C_b_n
    A[9:12, 9:12] = -1.0/KF_param.accel_TC_bias*np.identity(3)
    A[6:9, 12:15] = -old_C_b_n
    A[12:15, 12:15] = -1.0/KF_param.gyro_TC_bias*np.identity(3)
    F = scipy.linalg.expm(A*tor_s)  # Discretize A matrix

    # Noise Covariance Matrix
    M[3:6, 0:3] = old_C_b_n
    M[6:9, 3:6] = -old_C_b_n
    M[9:15, 6:12] = np.identity(6)

    sigma_mu_a2 = 2.0*(KF_param.accel_markov_bias_sigma*g_n[2])**2.0/KF_param.accel_TC_bias
    sigma_mu_g2 = 2.0*(KF_param.gyro_markov_bias_sigma)**2.0/KF_param.gyro_TC_bias
    U[0:3, 0:3] = np.identity(3)*(KF_param.accel_meas_sigma*g_n[2])**2.0
    U[3:6, 3:6] = np.identity(3)*(KF_param.gyro_meas_sigma)**2.0
    U[6:9, 6:9] = np.identity(3)*sigma_mu_a2
    U[9:12, 9:12] = np.identity(3)*sigma_mu_g2

    Q = (np.identity(15)+tor_s*A)@(tor_s*M@U@M.T)

    # GNSS measurement covariance matrix
    R[0:3, 0:3] = np.identity(3)*KF_param.GPS_NED_pos_sigma**2.0
    R[3:6, 3:6] = np.identity(3)*KF_param.GPS_NED_vel_sigma**2.0

    return F, Q, H, R


def initialize_P_EKF(KF_param):

    # Initial Error Covariance Matrix for nav EKF
    #
    # INPUTS:
    #   KF_Param    Kalman Filter parameters
    #
    # OUTPUTS
    #   P           (initial) Error Covariance Matrix

    # Intitialize state covariance matrix
    P = np.zeros((15, 15))
    P[0:3, 0:3] = np.identity(3)*KF_param.GPS_NED_pos_sigma**2.0
    P[3:6, 3:6] = 10.0*np.identity(3)*KF_param.GPS_NED_vel_sigma**2.0
    P[6:9, 6:9] = np.identity(3)*KF_param.init_att_unc**2.0
    P[9:12, 9:12] = 10.0*np.identity(3)*(KF_param.accel_markov_bias_sigma*9.81)**2.0
    P[12:15, 12:15] = 10.0*np.identity(3)*KF_param.gyro_markov_bias_sigma**2.0
    return 10.0*P


def prediction_step(F, P, Q):

    # Kalman Filter prediction step
    #
    # INPUTS:
    #   F       Discretized linearized INS error equations
    #   P       Error Covariance Matrix
    #   Q       Noise Covariance Matrix
    #
    # OUTPUTS
    #   P       Updated Error Covariance Matrix

    FP = F@P
    P = F@P@F.T+Q
    
    return P, FP


def correction_step(P, H, R):

    # Kalman Filter correction step
    #
    # INPUTS:
    #   P       Error Covariance Matrix
    #   H       Measurement Matrix
    #   R       Measurement Covariance Matrix
    #
    # OUTPUTS
    #   P       Updated Error Covariance Matrix
    #   K       Kalman Gain

    S = scipy.linalg.inv(H@P@H.T+R)
    K = P@H.T@S
    P = (np.identity(15)-K@H)@P
    P = 0.5*(P + P.T)
    return P, K


def innovation(K, est_ned, meas_ned, est_v_eb_n, meas_v_eb_n):

    # Calculate Innovation and multiply by Kalman gain
    #
    # INPUTS:
    #   K               Kalman Gain
    #   est_ned         INS estimate of NED position (m)
    #   meas_ned        GPS estimate of NED position (m)
    #   est_v_eb_n      INS estimate of velocity (m/sec)
    #   meas_v_eb_n     GPS estimate of velcoity (m/sec)
    #
    # OUTPUTS
    #   delta_x         Kalman gain times innovation

    delta_x = np.zeros((15, 1))
    delta_y = np.zeros((6, 1))
    delta_y[0] = est_ned[0] - meas_ned[0]
    delta_y[1] = est_ned[1] - meas_ned[1]
    delta_y[2] = est_ned[2] - meas_ned[2]
    delta_y[3] = est_v_eb_n[0] - meas_v_eb_n[0]
    delta_y[4] = est_v_eb_n[1] - meas_v_eb_n[1]
    delta_y[5] = est_v_eb_n[2] - meas_v_eb_n[2]
    delta_x = K@delta_y
    return delta_x


def correct_NED_errors(delta_x, est_C_b_n, est_IMU_bias, old_latR,
                       est_latR, est_longR, est_alt, new_meas_alt,
                       est_v_eb_n, new_meas_v_eb_n):

    # Update INS solution by estimating errors
    #
    # INPUTS:
    #   delta_x:            Kalman Gain times Innovation
    #   est_C_b_n           INS estimated DCM
    #   est_IMU_bias        Previous IMU bias (row1: accel, row2: gyro)
    #   est_latR            Previous Latitude (rad)
    #   est_latR            INS estimate Latitude (rad)
    #   est_longR           INS estimate Longitude (rad)
    #   est_alt             INS estimate altitude (m)
    #   new_meas_alt        GPS estimate altitude (m)
    #   est_v_eb_n          INS esitmate velocity (m/sec)
    #   new_meas_v_eb_n     GPS esimate velocity (m/sec)
    #
    # OUTPUTS
    #   est_latR            NEW EKF latitude estimate (rad)
    #   est_longR           NEW EKF longitude estimate (rad)
    #   est_alt             NEW EKF altitude estiamte (m)
    #   est_v_eb_n          NEW EKF velocity (m/sec)
    #   est_C_b_n           New EKF DCM

    # DCM Update
    dumMat = np.zeros((3, 3))
    dumMat[0:3, 0:3] = trans.skew(delta_x[6:9])
    est_C_b_n = (np.identity(3) + dumMat)@est_C_b_n

    # IMU Bias Update
    est_IMU_bias[0, 0] = est_IMU_bias[0, 0] - float(delta_x[9])
    est_IMU_bias[0, 1] = est_IMU_bias[0, 1] - float(delta_x[10])
    est_IMU_bias[0, 2] = est_IMU_bias[0, 2] - float(delta_x[11])
    est_IMU_bias[1, 0] = est_IMU_bias[1, 0] - float(delta_x[12])
    est_IMU_bias[1, 1] = est_IMU_bias[1, 1] - float(delta_x[13])
    est_IMU_bias[1, 2] = est_IMU_bias[1, 2] - float(delta_x[14])

    # Transport Rate (previous step needed)
    RN = a*(1.0-e**2)/(1.0-e**2.0*(np.sin(old_latR))**2.0)**1.5
    RE = a/np.sqrt(1.0-e**2.0*(np.sin(old_latR))**2.0)

    # Position and Velocity update (avoid vertical channel instability)
    est_latR = est_latR - float(delta_x[0]/(RN+est_alt))
    est_longR = est_longR - float(delta_x[1]/((RE+est_alt)*np.cos(est_latR)))
    est_alt = new_meas_alt
    est_v_eb_n[0] = est_v_eb_n[0] - float(delta_x[3])
    est_v_eb_n[1] = est_v_eb_n[1] - float(delta_x[4])
    est_v_eb_n[2] = new_meas_v_eb_n
    return est_latR, est_longR, est_alt, est_v_eb_n, est_C_b_n, est_IMU_bias
