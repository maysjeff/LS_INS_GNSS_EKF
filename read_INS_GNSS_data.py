# Data Read Functions

# Jeffrey Mays

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read GPS & IMU data
def get_GPS_LLA_data(folder):
    # col 0: Latitude (rad)
    # col 1: Longitude (rad)
    # col 2: Altitude (m)
    gps_pos_data = pd.read_csv(folder+'/gps_pos_lla.txt',
                               sep=',', header=None)
    gps_pos_data.columns = ['Lat (rad)', 'Long (rad)', 'Alt (m)']
    return gps_pos_data


def get_GPS_vel_ned_data(folder):
    # col 0: North x_dot (m/s)
    # col 1: East x_dot (m/s)
    # col 2: Down x_dot (m/s)
    gps_vel_data = pd.read_csv(folder+'/gps_vel_ned.txt',
                               sep=',', header=None)
    gps_vel_data.columns = ['v_N (m/s)', 'v_E (m/s)', 'v_D (m/s)']
    return gps_vel_data


def get_IMU_data(folder):
    # col 0: time (sec)
    # col 1: meas_omega_ib_b P (rad/s)
    # col 2: meas_omega_ib_b Q (rad/s)
    # col 3: meas_omega_ib_b R (rad/s)
    # col 4: meas_f_ib_b X (m/s^2)
    # col 5: meas_f_ib_b Y (m/s^2)
    # col 6: meas_f_ib_b Z (m/s^2)
    IMU_data = pd.read_csv(folder+'/imu.txt',
                           sep=',', header=None)
    IMU_data.columns = ['time (sec)', 'P', 'Q', 'R', 'X', 'Y', 'Z']
    return IMU_data


def get_all_INS_GNSS_data(flag):
    GPS_LLA_data = get_GPS_LLA_data('data')
    GPS_VEL_NED_data = get_GPS_vel_ned_data('data')
    IMU_data = get_IMU_data('data')

    # Set up and fill vectors/matrices
    meas_GPS_pos_lla = np.zeros((len(GPS_LLA_data), 3))
    meas_GPS_vel_ned = np.zeros((len(GPS_VEL_NED_data), 3))
    meas_omega_ib_b = np.zeros((len(IMU_data), 3))
    meas_f_ib_b = np.zeros((len(IMU_data), 3))
    in_profile_data = np.zeros((13, len(GPS_LLA_data)))  # Input data array

    time_series = IMU_data.loc[:]['time (sec)'].to_numpy()
    meas_omega_ib_b[:, 0] = IMU_data.loc[:]['P'].to_numpy()  # Meas roll rate
    meas_omega_ib_b[:, 1] = IMU_data.loc[:]['Q'].to_numpy()  # Meas pitch rate
    meas_omega_ib_b[:, 2] = IMU_data.loc[:]['R'].to_numpy()  # Meas yaw rate
    meas_f_ib_b[:, 0] = IMU_data.loc[:]['X'].to_numpy()  # Measured X accel
    meas_f_ib_b[:, 1] = IMU_data.loc[:]['Y'].to_numpy()  # Measured Y accel
    meas_f_ib_b[:, 2] = IMU_data.loc[:]['Z'].to_numpy()  # Measured Z accel

    meas_GPS_pos_lla[:, 0] = GPS_LLA_data.loc[:]['Lat (rad)'].to_numpy()
    meas_GPS_pos_lla[:, 1] = GPS_LLA_data.loc[:]['Long (rad)'].to_numpy()
    meas_GPS_pos_lla[:, 2] = GPS_LLA_data.loc[:]['Alt (m)'].to_numpy()

    meas_GPS_vel_ned[:, 0] = GPS_VEL_NED_data.loc[:]['v_N (m/s)'].to_numpy()
    meas_GPS_vel_ned[:, 1] = GPS_VEL_NED_data.loc[:]['v_E (m/s)'].to_numpy()
    meas_GPS_vel_ned[:, 2] = GPS_VEL_NED_data.loc[:]['v_D (m/s)'].to_numpy()

    in_profile_data = np.array([time_series, meas_GPS_pos_lla[:, 0],
                                meas_GPS_pos_lla[:, 1], meas_GPS_pos_lla[:, 2],
                                meas_GPS_vel_ned[:, 0], meas_GPS_vel_ned[:, 1],
                                meas_GPS_vel_ned[:, 2], meas_omega_ib_b[:, 0],
                                meas_omega_ib_b[:, 1], meas_omega_ib_b[:, 2],
                                meas_f_ib_b[:, 0], meas_f_ib_b[:, 1],
                                meas_f_ib_b[:, 2]]).T
    print(' ')
    print('in_profile_array contains: ')
    print('col 0: Time series (sec)')
    print('col 1: Latitude (rad)')
    print('col 2: Longitude (rad)')
    print('col 3: Altitude (m)')
    print('col 4: GPS Velocity, v_N (m/s)')
    print('col 5: GPS Velocity, v_E (m/s)')
    print('col 6: GPS Velocity, v_D (m/s)')
    print('col 7: Measured body-frame roll rate, P (rad/s)')
    print('col 8: Measured body-frame pitch rate, Q (rad/s)')
    print('col 9: Measured body-frame yaw rate, R (rad/s)')
    print('col 10: Measured body-frame X Accel, X (m/s^2)')
    print('col 11: Measured body-frame Y Accel, Y (m/s^2)')
    print('col 12: Measured body-frame Z Accel, Z (m/s^2)')
    print(' ')

    if flag is True:
        # Plot data
        fig, a = plt.subplots(4, figsize=(10, 10))
        a[0].plot(time_series, meas_omega_ib_b[:, 0], 'b',
                  label='Meas. Roll Rate')
        a[0].plot(time_series, meas_omega_ib_b[:, 1], 'k',
                  label='Meas. Pitch Rate')
        a[0].plot(time_series, meas_omega_ib_b[:, 2], 'r',
                  label='Meas. Yaw Rate')
        a[0].grid()
        a[0].set_xlabel('Time (s)')
        a[0].set_ylabel('Angular Velocity (rad/s)')
        a[0].legend()

        a[1].plot(time_series, meas_f_ib_b[:, 0], 'b',
                  label='Meas. X accel')
        a[1].plot(time_series, meas_f_ib_b[:, 1], 'k',
                  label='Meas. Y accel')
        a[1].plot(time_series, meas_f_ib_b[:, 2], 'r',
                  label='Meas. Z accel')
        a[1].grid()
        a[1].set_xlabel('Time (s)')
        a[1].set_ylabel('Acceleration (m/s^2)')
        a[1].legend()

        a[2].plot(time_series, meas_GPS_vel_ned[:, 0], 'b',
                  label='Meas. GPS North Velocity')
        a[2].plot(time_series, meas_GPS_vel_ned[:, 1], 'k',
                  label='Meas. GPS East Velocity')
        a[2].plot(time_series, meas_GPS_vel_ned[:, 2], 'r',
                  label='Meas. GPS Down Velocity')
        a[2].grid()
        a[2].set_xlabel('Time (s)')
        a[2].set_ylabel('Velocity (m/s)')
        a[2].legend()

        a[3].plot(time_series, meas_GPS_pos_lla[:, 2], 'b',
                  label='Meas. GPS Altitude')
        a[3].grid()
        a[3].set_xlabel('Time (s)')
        a[3].set_ylabel('GPS Altitude (m)')
        a[3].legend()

    return in_profile_data
