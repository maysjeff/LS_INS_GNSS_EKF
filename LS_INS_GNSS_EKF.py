# Jeffrey Mays
# INS/GNSS Error State Extended Kalman Filter
# April 27, 2020

import progressbar
import numpy as np
import matplotlib.pyplot as plt
import read_INS_GNSS_data as ins_gnss_data
import transform_functions as trans
import EKF_functions as EKF

plt.close('all')


# Constants
RTOD = 180.0 / np.pi  # radians to degrees
DTOR = np.pi / 180.0  # degrees to radians


# %% Read Data Into Python

print('Reading data from IMU & GPS...')

in_profile_data = ins_gnss_data.get_all_INS_GNSS_data(True)
no_epochs = len(in_profile_data)

print('Read data into in_profile_array')

# %% Initialize Navigation States

print(' ')
print('Initializing Navigation States...')
print(' ')


class data_packet:
    def __init__(self):
        # GPS position measurement standard deviation per axis
        self.GPS_NED_pos_sigma = 3.0  # m
        # GPS velocity measurement standard deviation per axis
        self.GPS_NED_vel_sigma = 0.2  # m/s
        # Attiude uncert
        self.init_att_unc = 10.0 * DTOR  # rad

        # Initial accelerometer bias per axis
        self.init_accel_bias = np.array([0.25, 0.077, -0.12])  # m/s^2
        # Accelerometer Markov bias standard deviation
        self.accel_markov_bias_sigma = 0.0005  # Times G!!!
        # Accelerometer time constant bias
        self.accel_TC_bias = 300.0  # sec
        # Accelerometer measurement noise standard deviation
        self.accel_meas_sigma = 0.12  # TIMES G!!!

        # Initial gyroscope bias per axis
        self.init_gyro_bias = np.array([2.4E-4, -1.3E-4, 5.6E-4])  # rad/s
        # Gyroscope Markov bias standard deviation
        self.gyro_markov_bias_sigma = 0.3 * DTOR
        # Gyroscope time constant bias
        self.gyro_TC_bias = 300.0  # sec
        # Gyroscope measurement noise standard deviation
        self.gyro_meas_sigma = 0.95 * DTOR  # deg/s


KF_param = data_packet()

# Initialize true navigation solution to first reading
old_time = in_profile_data[0][0]                # current time
old_latR = in_profile_data[0][1]                # lat is first GPS reading
old_longR = in_profile_data[0][2]               # Long is first GPS reading
old_alt = in_profile_data[0][3]                 # alt is first GPS reading
old_eul_nb = np.zeros((3, 1))                   # Euler angles are [0 0 0]^T
old_C_b_n = trans.Euler_to_CTM(old_eul_nb).T    # old CTM
old_v_eb_n = in_profile_data[0][4:7]            # vel ned is first GPS reading
first_omega_ib_b = in_profile_data[0][7:10]     # first gyro measurement
first_f_ib_b = in_profile_data[0][10:13]        # first accel measurement

# Find NED frame ECEF reference
latR_base = old_latR
longR_base = old_longR
alt_base = 0.0  # assume base is at zero alt?
xB, yB, zB = trans.LLA_to_ECEF(latR_base, longR_base, alt_base)
ECEF_ref = np.array([xB, yB, zB])

# Initialize IMU bias states
est_IMU_bias = np.array([KF_param.init_accel_bias.T,
                         KF_param.init_gyro_bias.T])

# Initialize output profile data array
out_profile_data = np.zeros((no_epochs, 19))
out_profile_data[0][0] = old_time
out_profile_data[0][1:4] = old_eul_nb.T*RTOD
out_profile_data[0][4:7] = old_v_eb_n.T
out_profile_data[0][7] = old_latR
out_profile_data[0][8] = old_longR
out_profile_data[0][9] = old_alt
out_profile_data[0][10] = 0.0  # Initial NED frame pos
out_profile_data[0][11] = 0.0
out_profile_data[0][12] = -old_alt  # - Down
out_profile_data[0][13:16] = est_IMU_bias[0]
out_profile_data[0][16:19] = est_IMU_bias[1]

GPS_NED = np.zeros((no_epochs, 3))
INS_NED = np.zeros((no_epochs, 3))
v_eb_b = np.zeros((no_epochs, 3))
AOA = np.zeros((no_epochs, 1))
BETA = np.zeros((no_epochs, 1))

# Initialize GNSS timing model
GNSS_epoch_interval = 1.0  # Hz of GNSS Receiver
time_last_GNSS = old_time
GNSS_epoch = 1

print('Initial Conditions:')
print('===================')
print('Start time: ' + str(old_time))
print('Latitude (rad): ' + str(old_latR))
print('Longitude (rad): ' + str(old_longR))
print('UAV altitude (m): ' + str(old_alt))
print('UAV attitude (rad): ' + str(old_eul_nb.T))
print('UAV vel_ned (m/s): ' + str(old_v_eb_n.T))
print(' ')


# %% Compute Noise Covariance Matrices

P = EKF.initialize_P_EKF(KF_param)

# %% Error State EKF

bar = progressbar.ProgressBar(maxval=no_epochs,
                              widgets=[progressbar.Bar('=', '[', ']'),
                                       ' ', progressbar.Percentage()])
bar.start()

for epoch in range(1, no_epochs):

    # Input data from motion profile array (use epoch k-1 to compute epoch k)
    time = in_profile_data[epoch][0]
    meas_latR = in_profile_data[epoch][1]
    meas_longR = in_profile_data[epoch][2]
    meas_alt = in_profile_data[epoch][3]
    meas_v_eb_n = in_profile_data[epoch][4:7]
    meas_omega_ib_b = in_profile_data[epoch][7:10]
    meas_f_ib_b = in_profile_data[epoch][10:13]

    # Time interval of INS
    tor_i = time - old_time

    # Correct IMU errors
    meas_f_ib_b = meas_f_ib_b + est_IMU_bias[0]
    meas_omega_ib_b = meas_omega_ib_b + est_IMU_bias[1]

    # Update estimated INS Solution
    est_C_b_n, est_v_eb_n, est_latR, est_longR, est_alt = EKF.INS_Equations_NED(tor_i, old_latR, old_longR, old_alt, old_v_eb_n, old_C_b_n, meas_omega_ib_b, meas_f_ib_b)

    # Compute INS solution in NED frame
    xE, yE, zE = trans.LLA_to_ECEF(est_latR, est_longR, est_alt)
    r_eb_e = np.zeros((3, 1))
    r_eb_e = np.array([xE, yE, zE])
    est_ned = np.zeros((3, 1))
    est_ned = trans.ECEF_to_NED(r_eb_e, latR_base, longR_base, alt_base)
    INS_NED[epoch] = est_ned.T  # Vector for plotting purposes

    # Convert CURRENT GPS measured pos to NED (for innovation)
    new_meas_latR = in_profile_data[epoch][1]
    new_meas_longR = in_profile_data[epoch][2]
    new_meas_alt = in_profile_data[epoch][3]
    new_meas_v_eb_n = in_profile_data[epoch][6]
    xE, yE, zE = trans.LLA_to_ECEF(new_meas_latR, new_meas_longR, new_meas_alt)
    r_eb_e = np.zeros((3, 1))
    r_eb_e = np.array([xE, yE, zE])
    GPS_ned = np.zeros((3, 1))
    GPS_ned = trans.ECEF_to_NED(r_eb_e, latR_base, longR_base, alt_base)
    GPS_NED[epoch] = GPS_ned.T  # Vector for plotting purposes

    # Linearize and discretize error model
    F, Q, H, R = EKF.Get_dem_EKF_matrices(KF_param, tor_i, old_latR, old_alt,
                                          old_v_eb_n, old_C_b_n, meas_f_ib_b)

    # Compute prediction step
    P = EKF.prediction_step(F, P, Q)

    if (time - time_last_GNSS) >= GNSS_epoch_interval:
        GNSS_epoch = GNSS_epoch + 1
        tor_s = time - time_last_GNSS  # EKF time interval
        time_last_GNSS = time

        # Compute correction step for errors
        P, K = EKF.correction_step(P, H, R)

        # Calculate Innovation
        delta_x = EKF.innovation(K, est_ned, GPS_ned, est_v_eb_n, meas_v_eb_n)

        # Update INS solution with estimated errors
        est_latR, est_longR, est_alt, est_v_eb_n, est_C_b_n = EKF.correct_NED_errors(delta_x, est_C_b_n, est_IMU_bias, old_latR, est_latR, est_longR, est_alt, new_meas_alt, est_v_eb_n, new_meas_v_eb_n)

    # Convert LLA position solution to NED
    xE, yE, zE = trans.LLA_to_ECEF(est_latR, est_longR, est_alt)
    r_eb_e = np.zeros((3, 1))
    r_eb_e = np.array([xE, yE, zE])
    P_n = trans.LLA_to_NED(latR_base, longR_base, alt_base, r_eb_e, ECEF_ref)

    # Record ouput data record
    out_profile_data[epoch][0] = time
    out_profile_data[epoch][1:4] = trans.CTM_to_Euler(est_C_b_n).T*RTOD
    out_profile_data[epoch][4:7] = est_v_eb_n.T
    out_profile_data[epoch][7] = est_latR
    out_profile_data[epoch][8] = est_longR
    out_profile_data[epoch][9] = est_alt
    out_profile_data[epoch][10] = P_n[0]
    out_profile_data[epoch][11] = P_n[1]
    out_profile_data[epoch][12] = P_n[2]
    out_profile_data[epoch][13:16] = est_IMU_bias[0]
    out_profile_data[epoch][16:19] = est_IMU_bias[1]

    # Compute body frame and aero angles
    v_eb_b[epoch] = (est_C_b_n.T@est_v_eb_n).T
    AOA[epoch] = np.arctan(v_eb_b[epoch][2]/v_eb_b[epoch][0])
    BETA[epoch] = np.arcsin(v_eb_b[epoch][1]/(np.sqrt(v_eb_b[epoch][0]**2.0 +
                            v_eb_b[epoch][1]**2.0+v_eb_b[epoch][2]**2.0)))

    # Reset old values
    old_time = time
    old_C_b_n = est_C_b_n
    old_v_eb_n = est_v_eb_n
    old_latR = est_latR
    old_longR = est_longR
    old_alt = est_alt

    bar.update(epoch)
bar.finish()

# %% Plot North-East Trajectory

plt.figure(figsize=(12, 7.5))
plt.plot(out_profile_data[:, 11], out_profile_data[:, 10], 'b', label='Pos.')
plt.plot(0, 0, 'ro', label='Start', MarkerSize=15)
plt.plot(out_profile_data[no_epochs-1, 11], out_profile_data[no_epochs-1, 10],
         'r*', label='End', MarkerSize=15)
plt.xlim(-200, 200)
plt.ylim(-100, 150)
plt.grid()
plt.legend()
plt.xlabel('East (m)')
plt.ylabel('North (m)')

# plt.figure(figsize=(12, 7.5))
# plt.scatter(GPS_NED[:, 1], GPS_NED[:, 0])
# plt.scatter(INS_NED[:, 1], INS_NED[:, 0])
# plt.grid()
# plt.legend()
# plt.xlabel('East (m)')
# plt.ylabel('North (m)')
# plt.show()


# %% Plot Height time history

plt.figure(figsize=(10, 4))
plt.plot(out_profile_data[:, 0], -out_profile_data[:, 12], 'b', label='Down')
plt.grid()
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Altitude NED (m)')

# %% Plot NED Velocity and Body-frame Velocity time history

plt.figure(figsize=(10, 4))
plt.plot(out_profile_data[:, 0], out_profile_data[:, 4], 'b', label='North')
plt.plot(out_profile_data[:, 0], out_profile_data[:, 5], 'k', label='East')
plt.plot(out_profile_data[:, 0], out_profile_data[:, 6], 'r', label='Down')
plt.grid()
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Velocity (m/s)')

plt.figure(figsize=(10, 4))
plt.plot(out_profile_data[:, 0], v_eb_b[:, 0], 'b', label='U')
plt.plot(out_profile_data[:, 0], v_eb_b[:, 1], 'k', label='V')
plt.plot(out_profile_data[:, 0], v_eb_b[:, 2], 'r', label='W')
plt.grid()
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Velocity (m/s)')

plt.figure(figsize=(10, 4))
plt.plot(out_profile_data[:, 0], AOA[:, 0]*RTOD, 'b', label='Angle of Attack')
plt.plot(out_profile_data[:, 0], BETA[:, 0]*RTOD, 'r', label='Sideslip')
plt.xlim(330, 850)
plt.ylim(-30, 30)
plt.grid()
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Aero Angle (deg)')

# %% Plot Euler Angles time history

fig, eul = plt.subplots(3, figsize=(10, 8))
eul[0].plot(out_profile_data[:, 0], out_profile_data[:, 1], 'b', label='Phi')
eul[0].grid()
eul[0].set_ylabel('Phi (deg)')
eul[0].legend()

eul[1].plot(out_profile_data[:, 0], out_profile_data[:, 2], 'k', label='Theta')
eul[1].grid()
eul[1].set_ylabel('Theta (deg)')
eul[1].legend()

eul[2].plot(out_profile_data[:, 0], out_profile_data[:, 3], 'r', label='Psi')
eul[2].grid()
eul[2].set_xlabel('Time (sec)')
eul[2].set_ylabel('Psi (deg)')
eul[2].legend()

# %% Plot biases time history

fig, a = plt.subplots(6, figsize=(14, 12))
a[0].plot(out_profile_data[:, 0], out_profile_data[:, 13], 'b', label='X_b')
a[0].grid()
a[0].set_xlabel('Time (sec)')
a[0].set_ylabel('A_bias (m/s^2)')
a[0].legend()

a[1].plot(out_profile_data[:, 0], out_profile_data[:, 14], 'k', label='Y_b')
a[1].grid()
a[1].set_xlabel('Time (sec)')
a[1].set_ylabel('A_bias (m/s^2)')
a[1].legend()

a[2].plot(out_profile_data[:, 0], out_profile_data[:, 15], 'r', label='Z_b')
a[2].grid()
a[2].set_xlabel('Time (sec)')
a[2].set_ylabel('A_bias (m/s^2)')
a[2].legend()

a[3].plot(out_profile_data[:, 0], out_profile_data[:, 16], 'b', label='X_b')
a[3].grid()
a[3].set_xlabel('Time (sec)')
a[3].set_ylabel('G_bias(deg/s)')
a[3].legend()

a[4].plot(out_profile_data[:, 0], out_profile_data[:, 17], 'k', label='Y_b')
a[4].grid()
a[4].set_xlabel('Time (sec)')
a[4].set_ylabel('G_bias(deg/s)')
a[4].legend()

a[5].plot(out_profile_data[:, 0], out_profile_data[:, 18], 'r', label='Z_b')
a[5].grid()
a[5].set_xlabel('Time (sec)')
a[5].set_ylabel('G_bias(deg/s)')
a[5].legend()
