from array import array
from operator import gt
import sys, os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
import math 
from scipy.interpolate import interp1d
import quaternion as quat
from pyquaternion import Quaternion

color_x = '#CC1EEA'
color_y = '#06AE97'
color_z = '#FFA409'

color_dragon_ekom = 'tab:blue'
color_rgbde = 'tab:orange'

SMALL_SIZE = 18
MEDIUM_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def computeEuclideanDistance(x1, y1, z1, x2, y2, z2):
    list_dist = []
    for xp1,yp1,zp1,xp2,yp2,zp2 in zip(x1,y1,z1,x2,y2,z2):
        if np.isneginf(xp2) or np.isneginf(yp2) or np.isneginf(zp2) or np.isnan(xp2) or np.isnan(yp2) or np.isnan(zp2):
            continue
        else:
            list_dist.append(math.sqrt((xp2-xp1)*(xp2-xp1)+(yp2-yp1)*(yp2-yp1)+(zp2-zp1)*(zp2-zp1)))
    array_dist = np.array(list_dist)
    return array_dist

def computeQuaternionError(qx1, qy1, qz1, qw1, qx2, qy2, qz2, qw2):

    list_q_error = []
    for qxp1,qyp1,qzp1,qwp1,qxp2,qyp2,qzp2,qwp2 in zip(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2):
        normalizer1 = math.sqrt(qwp1*qwp1+qxp1*qxp1+qyp1*qyp1+qzp1*qzp1)
        qxp1 = qxp1/normalizer1
        qyp1 = qyp1/normalizer1
        qzp1 = qzp1/normalizer1
        qwp1 = qwp1/normalizer1

        normalizer2 = math.sqrt(qwp2*qwp2+qxp2*qxp2+qyp2*qyp2+qzp2*qzp2)
        qxp2 = qxp2/normalizer2
        qyp2 = qyp2/normalizer2
        qzp2 = qzp2/normalizer2
        qwp2 = qwp2/normalizer2
        inner_product = qwp1*qwp2+qxp1*qxp2+qyp1*qyp2+qzp1*qzp2
        quaternion_angle = np.arccos(np.clip(2*inner_product*inner_product-1, -1, 1))
        if np.isnan(quaternion_angle):
            continue
        else:
            list_q_error.append(quaternion_angle)

    array_q_error = np.array(list_q_error)
    return array_q_error 

def resampling_by_interpolate(time_samples, x_values, y_values):
    f_neareast = interp1d(x_values, y_values, kind='nearest', fill_value="extrapolate")
    resampled = f_neareast(time_samples)
    return resampled

def quaternion_to_euler_angle(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """

        roll_list = []
        pitch_list = []
        yaw_list = []

        degrees = 57.2958

        for qx,qy,qz,qw in zip(x,y,z,w):

            t0 = +2.0 * (qw * qx + qy * qz)
            t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
            roll_x = math.atan2(t0, t1)
        
            t2 = +2.0 * (qw * qy - qz * qx)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = math.asin(t2)
        
            t3 = +2.0 * (qw * qz + qx * qy)
            t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
            yaw_z = math.atan2(t3, t4)

            roll_list.append(roll_x*degrees)
            pitch_list.append(pitch_y*degrees)  
            yaw_list.append(yaw_z*degrees)

            roll_array = np.array(roll_list)
            pitch_array = np.array(pitch_list)
            yaw_array = np.array(yaw_list)
     
        return roll_array, pitch_array, yaw_array # in degrees

def cleanEuler(angle, angle_type): 

    # angle = angle[~np.isnan(angle)]
    diff_arrays = angle[1:-1]-angle[0:-2]
    prev_x = 0
    th = 180
    filtered_angles_list = [] 
    diff_list=[]
    for idx, x in enumerate(angle):
        if idx == 0:
            if angle_type==2 and x < 0:
                x+=360 
            prev_x = x
        else:
            diff = abs(x - prev_x)
            diff_list.append(diff)
            if diff > th:
                x += 360
            else:
                if angle_type==2 and x<0:
                    x += 360
            prev_x = x
        filtered_angles_list.append(x)

    return(np.array(filtered_angles_list))
    
# --------------------------------------------------------------------------- DRAGON ---------------------------------------------------------------------------------------

filePath_dataset = '/data/dragon/'
dragon_gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_translation_x_1_m_s_2/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_translation_y_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_translation_z_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_roll_4rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_pitch_4rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_yaw_4rad_s_2/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

dragon_gt_trans_x['t'] = (dragon_gt_trans_x['t']-dragon_gt_trans_x['t'][0])*10
dragon_gt_trans_x['x'] = dragon_gt_trans_x['x']*0.01
dragon_gt_trans_x['y'] = dragon_gt_trans_x['y']*0.01
dragon_gt_trans_x['z'] = dragon_gt_trans_x['z']*0.01

dragon_gt_trans_y['t'] = (dragon_gt_trans_y['t']-dragon_gt_trans_y['t'][0])*10
dragon_gt_trans_y['x'] = dragon_gt_trans_y['x']*0.01
dragon_gt_trans_y['y'] = dragon_gt_trans_y['y']*0.01
dragon_gt_trans_y['z'] = dragon_gt_trans_y['z']*0.01

dragon_gt_trans_z['t'] = (dragon_gt_trans_z['t']-dragon_gt_trans_z['t'][0])*10
dragon_gt_trans_z['x'] = dragon_gt_trans_z['x']*0.01
dragon_gt_trans_z['y'] = dragon_gt_trans_z['y']*0.01
dragon_gt_trans_z['z'] = dragon_gt_trans_z['z']*0.01

dragon_gt_roll['t'] = (dragon_gt_roll['t']-dragon_gt_roll['t'][0])*10
dragon_gt_roll['x'] = dragon_gt_roll['x']*0.01
dragon_gt_roll['y'] = dragon_gt_roll['y']*0.01
dragon_gt_roll['z'] = dragon_gt_roll['z']*0.01

dragon_gt_pitch['t'] = (dragon_gt_pitch['t']-dragon_gt_pitch['t'][0])*10
dragon_gt_pitch['x'] = dragon_gt_pitch['x']*0.01
dragon_gt_pitch['y'] = dragon_gt_pitch['y']*0.01
dragon_gt_pitch['z'] = dragon_gt_pitch['z']*0.01

dragon_gt_yaw['t'] = (dragon_gt_yaw['t']-dragon_gt_yaw['t'][0])*10
dragon_gt_yaw['x'] = dragon_gt_yaw['x']*0.01
dragon_gt_yaw['y'] = dragon_gt_yaw['y']*0.01
dragon_gt_yaw['z'] = dragon_gt_yaw['z']*0.01

dragon_ekom_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_ekom_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_ekom_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_ekom_roll = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_ekom_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_ekom_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

dragon_ekom_trans_x_resampled_x = resampling_by_interpolate(dragon_gt_trans_x['t'], dragon_ekom_trans_x['t'], dragon_ekom_trans_x['x'])
dragon_ekom_trans_x_resampled_y = resampling_by_interpolate(dragon_gt_trans_x['t'], dragon_ekom_trans_x['t'], dragon_ekom_trans_x['y'])
dragon_ekom_trans_x_resampled_z = resampling_by_interpolate(dragon_gt_trans_x['t'], dragon_ekom_trans_x['t'], dragon_ekom_trans_x['z'])
dragon_ekom_trans_x_resampled_qx = resampling_by_interpolate(dragon_gt_trans_x['t'], dragon_ekom_trans_x['t'], dragon_ekom_trans_x['qx'])
dragon_ekom_trans_x_resampled_qy = resampling_by_interpolate(dragon_gt_trans_x['t'], dragon_ekom_trans_x['t'], dragon_ekom_trans_x['qy'])
dragon_ekom_trans_x_resampled_qz = resampling_by_interpolate(dragon_gt_trans_x['t'], dragon_ekom_trans_x['t'], dragon_ekom_trans_x['qz'])
dragon_ekom_trans_x_resampled_qw = resampling_by_interpolate(dragon_gt_trans_x['t'], dragon_ekom_trans_x['t'], dragon_ekom_trans_x['qw'])

dragon_ekom_trans_y_resampled_x = resampling_by_interpolate(dragon_gt_trans_y['t'], dragon_ekom_trans_y['t'], dragon_ekom_trans_y['x'])
dragon_ekom_trans_y_resampled_y = resampling_by_interpolate(dragon_gt_trans_y['t'], dragon_ekom_trans_y['t'], dragon_ekom_trans_y['y'])
dragon_ekom_trans_y_resampled_z = resampling_by_interpolate(dragon_gt_trans_y['t'], dragon_ekom_trans_y['t'], dragon_ekom_trans_y['z'])
dragon_ekom_trans_y_resampled_qx = resampling_by_interpolate(dragon_gt_trans_y['t'], dragon_ekom_trans_y['t'], dragon_ekom_trans_y['qx'])
dragon_ekom_trans_y_resampled_qy = resampling_by_interpolate(dragon_gt_trans_y['t'], dragon_ekom_trans_y['t'], dragon_ekom_trans_y['qy'])
dragon_ekom_trans_y_resampled_qz = resampling_by_interpolate(dragon_gt_trans_y['t'], dragon_ekom_trans_y['t'], dragon_ekom_trans_y['qz'])
dragon_ekom_trans_y_resampled_qw = resampling_by_interpolate(dragon_gt_trans_y['t'], dragon_ekom_trans_y['t'], dragon_ekom_trans_y['qw'])

dragon_ekom_trans_z_resampled_x = resampling_by_interpolate(dragon_gt_trans_z['t'], dragon_ekom_trans_z['t'], dragon_ekom_trans_z['x'])
dragon_ekom_trans_z_resampled_y = resampling_by_interpolate(dragon_gt_trans_z['t'], dragon_ekom_trans_z['t'], dragon_ekom_trans_z['y'])
dragon_ekom_trans_z_resampled_z = resampling_by_interpolate(dragon_gt_trans_z['t'], dragon_ekom_trans_z['t'], dragon_ekom_trans_z['z'])
dragon_ekom_trans_z_resampled_qx = resampling_by_interpolate(dragon_gt_trans_z['t'], dragon_ekom_trans_z['t'], dragon_ekom_trans_z['qx'])
dragon_ekom_trans_z_resampled_qy = resampling_by_interpolate(dragon_gt_trans_z['t'], dragon_ekom_trans_z['t'], dragon_ekom_trans_z['qy'])
dragon_ekom_trans_z_resampled_qz = resampling_by_interpolate(dragon_gt_trans_z['t'], dragon_ekom_trans_z['t'], dragon_ekom_trans_z['qz'])
dragon_ekom_trans_z_resampled_qw = resampling_by_interpolate(dragon_gt_trans_z['t'], dragon_ekom_trans_z['t'], dragon_ekom_trans_z['qw'])

dragon_ekom_roll_resampled_x = resampling_by_interpolate(dragon_gt_roll['t'], dragon_ekom_roll['t'], dragon_ekom_roll['x'])
dragon_ekom_roll_resampled_y = resampling_by_interpolate(dragon_gt_roll['t'], dragon_ekom_roll['t'], dragon_ekom_roll['y'])
dragon_ekom_roll_resampled_z = resampling_by_interpolate(dragon_gt_roll['t'], dragon_ekom_roll['t'], dragon_ekom_roll['z'])
dragon_ekom_roll_resampled_qx = resampling_by_interpolate(dragon_gt_roll['t'], dragon_ekom_roll['t'], dragon_ekom_roll['qx'])
dragon_ekom_roll_resampled_qy = resampling_by_interpolate(dragon_gt_roll['t'], dragon_ekom_roll['t'], dragon_ekom_roll['qy'])
dragon_ekom_roll_resampled_qz = resampling_by_interpolate(dragon_gt_roll['t'], dragon_ekom_roll['t'], dragon_ekom_roll['qz'])
dragon_ekom_roll_resampled_qw = resampling_by_interpolate(dragon_gt_roll['t'], dragon_ekom_roll['t'], dragon_ekom_roll['qw'])

dragon_ekom_pitch_resampled_x = resampling_by_interpolate(dragon_gt_pitch['t'], dragon_ekom_pitch['t'], dragon_ekom_pitch['x'])
dragon_ekom_pitch_resampled_y = resampling_by_interpolate(dragon_gt_pitch['t'], dragon_ekom_pitch['t'], dragon_ekom_pitch['y'])
dragon_ekom_pitch_resampled_z = resampling_by_interpolate(dragon_gt_pitch['t'], dragon_ekom_pitch['t'], dragon_ekom_pitch['z'])
dragon_ekom_pitch_resampled_qx = resampling_by_interpolate(dragon_gt_pitch['t'], dragon_ekom_pitch['t'], dragon_ekom_pitch['qx'])
dragon_ekom_pitch_resampled_qy = resampling_by_interpolate(dragon_gt_pitch['t'], dragon_ekom_pitch['t'], dragon_ekom_pitch['qy'])
dragon_ekom_pitch_resampled_qz = resampling_by_interpolate(dragon_gt_pitch['t'], dragon_ekom_pitch['t'], dragon_ekom_pitch['qz'])
dragon_ekom_pitch_resampled_qw = resampling_by_interpolate(dragon_gt_pitch['t'], dragon_ekom_pitch['t'], dragon_ekom_pitch['qw'])

dragon_ekom_yaw_resampled_x = resampling_by_interpolate(dragon_gt_yaw['t'], dragon_ekom_yaw['t'], dragon_ekom_yaw['x'])
dragon_ekom_yaw_resampled_y = resampling_by_interpolate(dragon_gt_yaw['t'], dragon_ekom_yaw['t'], dragon_ekom_yaw['y'])
dragon_ekom_yaw_resampled_z = resampling_by_interpolate(dragon_gt_yaw['t'], dragon_ekom_yaw['t'], dragon_ekom_yaw['z'])
dragon_ekom_yaw_resampled_qx = resampling_by_interpolate(dragon_gt_yaw['t'], dragon_ekom_yaw['t'], dragon_ekom_yaw['qx'])
dragon_ekom_yaw_resampled_qy = resampling_by_interpolate(dragon_gt_yaw['t'], dragon_ekom_yaw['t'], dragon_ekom_yaw['qy'])
dragon_ekom_yaw_resampled_qz = resampling_by_interpolate(dragon_gt_yaw['t'], dragon_ekom_yaw['t'], dragon_ekom_yaw['qz'])
dragon_ekom_yaw_resampled_qw = resampling_by_interpolate(dragon_gt_yaw['t'], dragon_ekom_yaw['t'], dragon_ekom_yaw['qw'])

dragon_ekom_trans_x_alpha,dragon_ekom_trans_x_beta,dragon_ekom_trans_x_gamma = quaternion_to_euler_angle(dragon_ekom_trans_x['qw'], dragon_ekom_trans_x['qx'], dragon_ekom_trans_x['qy'], dragon_ekom_trans_x['qz'])
dragon_gt_trans_x_alpha,dragon_gt_trans_x_beta,dragon_gt_trans_x_gamma = quaternion_to_euler_angle(dragon_gt_trans_x['qw'], dragon_gt_trans_x['qx'], dragon_gt_trans_x['qy'], dragon_gt_trans_x['qz'])

dragon_ekom_trans_x_alpha_cleaned = cleanEuler(dragon_ekom_trans_x_alpha,0)
dragon_ekom_trans_x_beta_cleaned = cleanEuler(dragon_ekom_trans_x_beta,1)
dragon_ekom_trans_x_gamma_cleaned = cleanEuler(dragon_ekom_trans_x_gamma,2)

dragon_gt_trans_x_alpha_cleaned = cleanEuler(dragon_gt_trans_x_alpha,0)
dragon_gt_trans_x_beta_cleaned = cleanEuler(dragon_gt_trans_x_beta,1)
dragon_gt_trans_x_gamma_cleaned = cleanEuler(dragon_gt_trans_x_gamma,1)

dragon_ekom_trans_y_alpha,dragon_ekom_trans_y_beta,dragon_ekom_trans_y_gamma = quaternion_to_euler_angle(dragon_ekom_trans_y['qw'], dragon_ekom_trans_y['qx'], dragon_ekom_trans_y['qy'], dragon_ekom_trans_y['qz'])
dragon_gt_trans_y_alpha,dragon_gt_trans_y_beta,dragon_gt_trans_y_gamma = quaternion_to_euler_angle(dragon_gt_trans_y['qw'], dragon_gt_trans_y['qx'], dragon_gt_trans_y['qy'], dragon_gt_trans_y['qz'])

dragon_ekom_trans_y_alpha_cleaned = cleanEuler(dragon_ekom_trans_y_alpha,0)
dragon_ekom_trans_y_beta_cleaned = cleanEuler(dragon_ekom_trans_y_beta,1)
dragon_ekom_trans_y_gamma_cleaned = cleanEuler(dragon_ekom_trans_y_gamma,2)

dragon_gt_trans_y_alpha_cleaned = cleanEuler(dragon_gt_trans_y_alpha,0)
dragon_gt_trans_y_beta_cleaned = cleanEuler(dragon_gt_trans_y_beta,1)
dragon_gt_trans_y_gamma_cleaned = cleanEuler(dragon_gt_trans_y_gamma,2)

dragon_ekom_trans_z_alpha,dragon_ekom_trans_z_beta,dragon_ekom_trans_z_gamma = quaternion_to_euler_angle(dragon_ekom_trans_z['qw'], dragon_ekom_trans_z['qx'], dragon_ekom_trans_z['qy'], dragon_ekom_trans_z['qz'])
dragon_gt_trans_z_alpha,dragon_gt_trans_z_beta,dragon_gt_trans_z_gamma = quaternion_to_euler_angle(dragon_gt_trans_z['qw'], dragon_gt_trans_z['qx'], dragon_gt_trans_z['qy'], dragon_gt_trans_z['qz'])

dragon_ekom_trans_z_alpha_cleaned = cleanEuler(dragon_ekom_trans_z_alpha,0)
dragon_ekom_trans_z_beta_cleaned = cleanEuler(dragon_ekom_trans_z_beta,1)
dragon_ekom_trans_z_gamma_cleaned = cleanEuler(dragon_ekom_trans_z_gamma,2)

dragon_gt_trans_z_alpha_cleaned = cleanEuler(dragon_gt_trans_z_alpha,0)
dragon_gt_trans_z_beta_cleaned = cleanEuler(dragon_gt_trans_z_beta,1)
dragon_gt_trans_z_gamma_cleaned = cleanEuler(dragon_gt_trans_z_gamma,2)

dragon_ekom_roll_alpha,dragon_ekom_roll_beta,dragon_ekom_roll_gamma = quaternion_to_euler_angle(dragon_ekom_roll['qw'], dragon_ekom_roll['qx'], dragon_ekom_roll['qy'], dragon_ekom_roll['qz'])
dragon_gt_roll_alpha,dragon_gt_roll_beta,dragon_gt_roll_gamma = quaternion_to_euler_angle(dragon_gt_roll['qw'], dragon_gt_roll['qx'], dragon_gt_roll['qy'], dragon_gt_roll['qz'])

dragon_ekom_roll_alpha_cleaned = cleanEuler(dragon_ekom_roll_alpha,0)
dragon_ekom_roll_beta_cleaned = cleanEuler(dragon_ekom_roll_beta,1)
dragon_ekom_roll_gamma_cleaned = cleanEuler(dragon_ekom_roll_gamma,2)

dragon_gt_roll_alpha_cleaned = cleanEuler(dragon_gt_roll_alpha,0)
dragon_gt_roll_beta_cleaned = cleanEuler(dragon_gt_roll_beta,1)
dragon_gt_roll_gamma_cleaned = cleanEuler(dragon_gt_roll_gamma,2)

dragon_ekom_pitch_alpha,dragon_ekom_pitch_beta,dragon_ekom_pitch_gamma = quaternion_to_euler_angle(dragon_ekom_pitch['qw'], dragon_ekom_pitch['qx'], dragon_ekom_pitch['qy'], dragon_ekom_pitch['qz'])
dragon_gt_pitch_alpha,dragon_gt_pitch_beta,dragon_gt_pitch_gamma = quaternion_to_euler_angle(dragon_gt_pitch['qw'], dragon_gt_pitch['qx'], dragon_gt_pitch['qy'], dragon_gt_pitch['qz'])

dragon_ekom_pitch_alpha_cleaned = cleanEuler(dragon_ekom_pitch_alpha,0)
dragon_ekom_pitch_beta_cleaned = cleanEuler(dragon_ekom_pitch_beta,1)
dragon_ekom_pitch_gamma_cleaned = cleanEuler(dragon_ekom_pitch_gamma,2)

dragon_gt_pitch_alpha_cleaned = cleanEuler(dragon_gt_pitch_alpha,0)
dragon_gt_pitch_beta_cleaned = cleanEuler(dragon_gt_pitch_beta,1)
dragon_gt_pitch_gamma_cleaned = cleanEuler(dragon_gt_pitch_gamma,2)

dragon_ekom_yaw_alpha,dragon_ekom_yaw_beta,dragon_ekom_yaw_gamma = quaternion_to_euler_angle(dragon_ekom_yaw['qw'], dragon_ekom_yaw['qx'], dragon_ekom_yaw['qy'], dragon_ekom_yaw['qz'])
dragon_gt_yaw_alpha,dragon_gt_yaw_beta,dragon_gt_yaw_gamma = quaternion_to_euler_angle(dragon_gt_yaw['qw'], dragon_gt_yaw['qx'], dragon_gt_yaw['qy'], dragon_gt_yaw['qz'])

dragon_ekom_yaw_alpha_cleaned = cleanEuler(dragon_ekom_yaw_alpha,0)
dragon_ekom_yaw_beta_cleaned = cleanEuler(dragon_ekom_yaw_beta,1)
dragon_ekom_yaw_gamma_cleaned = cleanEuler(dragon_ekom_yaw_gamma,2)

dragon_gt_yaw_alpha_cleaned = cleanEuler(dragon_gt_yaw_alpha,0)
dragon_gt_yaw_beta_cleaned = cleanEuler(dragon_gt_yaw_beta,1)
dragon_gt_yaw_gamma_cleaned = cleanEuler(dragon_gt_yaw_gamma,2)

# fig_summary, axs = plt.subplots(4,3)
# fig_summary.set_size_inches(18, 12)
# axs[0,0].plot(dragon_ekom_trans_x['t'], dragon_ekom_trans_x['x'], color=color_x, label='x')
# axs[0,0].plot(dragon_ekom_trans_x['t'], dragon_ekom_trans_x['y'], color=color_y, label='y')
# axs[0,0].plot(dragon_ekom_trans_x['t'], dragon_ekom_trans_x['z'], color=color_z, label='z')
# axs[0,0].plot(dragon_gt_trans_x['t'], dragon_gt_trans_x['x'], color=color_x, ls='--')
# axs[0,0].plot(dragon_gt_trans_x['t'], dragon_gt_trans_x['y'], color=color_y, ls='--')
# axs[0,0].plot(dragon_gt_trans_x['t'], dragon_gt_trans_x['z'], color=color_z, ls='--')
# axs[0,1].plot(dragon_ekom_trans_y['t'], dragon_ekom_trans_y['x'], color=color_x, label='x')
# axs[0,1].plot(dragon_ekom_trans_y['t'], dragon_ekom_trans_y['y'], color=color_y, label='y')
# axs[0,1].plot(dragon_ekom_trans_y['t'], dragon_ekom_trans_y['z'], color=color_z, label='z')
# axs[0,1].plot(dragon_gt_trans_y['t'], dragon_gt_trans_y['x'], color=color_x, ls='--')
# axs[0,1].plot(dragon_gt_trans_y['t'], dragon_gt_trans_y['y'], color=color_y, ls='--')
# axs[0,1].plot(dragon_gt_trans_y['t'], dragon_gt_trans_y['z'], color=color_z, ls='--')
# axs[0,2].plot(dragon_ekom_trans_z['t'], dragon_ekom_trans_z['x'], color=color_x, label='x')
# axs[0,2].plot(dragon_ekom_trans_z['t'], dragon_ekom_trans_z['y'], color=color_y, label='y')
# axs[0,2].plot(dragon_ekom_trans_z['t'], dragon_ekom_trans_z['z'], color=color_z, label='z')
# axs[0,2].plot(dragon_gt_trans_z['t'], dragon_gt_trans_z['x'], color=color_x, ls='--')
# axs[0,2].plot(dragon_gt_trans_z['t'], dragon_gt_trans_z['y'], color=color_y, ls='--')
# axs[0,2].plot(dragon_gt_trans_z['t'], dragon_gt_trans_z['z'], color=color_z, ls='--')
# axs[2,0].plot(dragon_ekom_roll['t'], dragon_ekom_roll['x'], color=color_x, label='x')
# axs[2,0].plot(dragon_ekom_roll['t'], dragon_ekom_roll['y'], color=color_y, label='y')
# axs[2,0].plot(dragon_ekom_roll['t'], dragon_ekom_roll['z'], color=color_z, label='z')
# axs[2,0].plot(dragon_gt_roll['t'], dragon_gt_roll['x'], color=color_x, ls='--')
# axs[2,0].plot(dragon_gt_roll['t'], dragon_gt_roll['y'], color=color_y, ls='--')
# axs[2,0].plot(dragon_gt_roll['t'], dragon_gt_roll['z'], color=color_z, ls='--')
# axs[2,1].plot(dragon_ekom_pitch['t'], dragon_ekom_pitch['x'], color=color_x, label='x')
# axs[2,1].plot(dragon_ekom_pitch['t'], dragon_ekom_pitch['y'], color=color_y, label='y')
# axs[2,1].plot(dragon_ekom_pitch['t'], dragon_ekom_pitch['z'], color=color_z, label='z')
# axs[2,1].plot(dragon_gt_pitch['t'], dragon_gt_pitch['x'], color=color_x, ls='--')
# axs[2,1].plot(dragon_gt_pitch['t'], dragon_gt_pitch['y'], color=color_y, ls='--')
# axs[2,1].plot(dragon_gt_pitch['t'], dragon_gt_pitch['z'], color=color_z, ls='--')
# axs[2,2].plot(dragon_ekom_yaw['t'], dragon_ekom_yaw['x'], color=color_x, label='x')
# axs[2,2].plot(dragon_ekom_yaw['t'], dragon_ekom_yaw['y'], color=color_y, label='y')
# axs[2,2].plot(dragon_ekom_yaw['t'], dragon_ekom_yaw['z'], color=color_z, label='z')
# axs[2,2].plot(dragon_gt_yaw['t'], dragon_gt_yaw['x'], color=color_x, ls='--')
# axs[2,2].plot(dragon_gt_yaw['t'], dragon_gt_yaw['y'], color=color_y, ls='--')
# axs[2,2].plot(dragon_gt_yaw['t'], dragon_gt_yaw['z'], color=color_z, ls='--')
# axs[1,0].plot(dragon_ekom_trans_x['t'], dragon_ekom_trans_x_alpha_cleaned, color=color_x, label='qx')
# axs[1,0].plot(dragon_ekom_trans_x['t'], dragon_ekom_trans_x_beta_cleaned, color=color_y, label='qy')
# axs[1,0].plot(dragon_ekom_trans_x['t'], dragon_ekom_trans_x_gamma_cleaned, color=color_z, label='qz')
# axs[1,0].plot(dragon_gt_trans_x['t'], dragon_gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
# axs[1,0].plot(dragon_gt_trans_x['t'], dragon_gt_trans_x_beta_cleaned, color=color_y, ls = '--')
# axs[1,0].plot(dragon_gt_trans_x['t'], dragon_gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
# axs[1,1].plot(dragon_ekom_trans_y['t'], dragon_ekom_trans_y_alpha_cleaned, color=color_x, label='qx')
# axs[1,1].plot(dragon_ekom_trans_y['t'], dragon_ekom_trans_y_beta_cleaned, color=color_y, label='qy')
# axs[1,1].plot(dragon_ekom_trans_y['t'], dragon_ekom_trans_y_gamma_cleaned, color=color_z, label='qz')
# axs[1,1].plot(dragon_gt_trans_y['t'], dragon_gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
# axs[1,1].plot(dragon_gt_trans_y['t'], dragon_gt_trans_y_beta_cleaned, color=color_y, ls = '--')
# axs[1,1].plot(dragon_gt_trans_y['t'], dragon_gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
# axs[1,2].plot(dragon_ekom_trans_z['t'], dragon_ekom_trans_z_alpha_cleaned, color=color_x, label='qx')
# axs[1,2].plot(dragon_ekom_trans_z['t'], dragon_ekom_trans_z_beta_cleaned, color=color_y, label='qy')
# axs[1,2].plot(dragon_ekom_trans_z['t'], dragon_ekom_trans_z_gamma_cleaned, color=color_z, label='qz')
# axs[1,2].plot(dragon_gt_trans_z['t'], dragon_gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
# axs[1,2].plot(dragon_gt_trans_z['t'], dragon_gt_trans_z_beta_cleaned, color=color_y, ls = '--')
# axs[1,2].plot(dragon_gt_trans_z['t'], dragon_gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
# axs[3,0].plot(dragon_ekom_roll['t'], dragon_ekom_roll_alpha_cleaned, color=color_x, label='qx')
# axs[3,0].plot(dragon_ekom_roll['t'], dragon_ekom_roll_beta_cleaned, color=color_y, label='qy')
# axs[3,0].plot(dragon_ekom_roll['t'], dragon_ekom_roll_gamma_cleaned, color=color_z, label='qz')
# axs[3,0].plot(dragon_gt_roll['t'], dragon_gt_roll_alpha_cleaned, color=color_x, ls = '--')
# axs[3,0].plot(dragon_gt_roll['t'], dragon_gt_roll_beta_cleaned, color=color_y, ls = '--')
# axs[3,0].plot(dragon_gt_roll['t'], dragon_gt_roll_gamma_cleaned, color=color_z, ls = '--')
# axs[3,1].plot(dragon_ekom_pitch['t'], dragon_ekom_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - dragon_ekom")
# axs[3,1].plot(dragon_ekom_pitch['t'], dragon_ekom_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - dragon_ekom")
# axs[3,1].plot(dragon_ekom_pitch['t'], dragon_ekom_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - dragon_ekom")
# axs[3,1].plot(dragon_gt_pitch['t'], dragon_gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - dragon_gt")
# axs[3,1].plot(dragon_gt_pitch['t'], dragon_gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - dragon_gt")
# axs[3,1].plot(dragon_gt_pitch['t'], dragon_gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - dragon_gt")
# axs[3,2].plot(dragon_ekom_yaw['t'], dragon_ekom_yaw_alpha_cleaned, color=color_x, label='x or roll')
# axs[3,2].plot(dragon_ekom_yaw['t'], dragon_ekom_yaw_beta_cleaned, color=color_y, label='y or pitch')
# axs[3,2].plot(dragon_ekom_yaw['t'], dragon_ekom_yaw_gamma_cleaned, color=color_z, label='z or yaw')
# axs[3,2].plot(dragon_gt_yaw['t'], dragon_gt_yaw_alpha_cleaned, color=color_x, ls = '--')
# axs[3,2].plot(dragon_gt_yaw['t'], dragon_gt_yaw_beta_cleaned, color=color_y, ls = '--')
# axs[3,2].plot(dragon_gt_yaw['t'], dragon_gt_yaw_gamma_cleaned, color=color_z, ls = '--')
# for i in range(0, 4):
#     for j in range(0, 3):
#         axs[i,j].set_xlim([-2, 50])
# axs[0,0].set_ylim(-0.3,  1)
# axs[0,1].set_ylim(-0.3,  1)
# axs[0,2].set_ylim(-0.3,  1)
# axs[2,0].set_ylim(-0.1,  0.9)
# axs[2,1].set_ylim(-0.1,  0.9)
# axs[2,2].set_ylim(-0.1,  0.9)
# axs[1,0].set_ylim(-120,  200)
# axs[1,1].set_ylim(-120,  200)
# axs[1,2].set_ylim(-120,  200)
# axs[3,0].set_ylim(-200,  300)
# axs[3,1].set_ylim(-200,  300)
# axs[3,2].set_ylim(-200,  300)
# axs[0,0].set_xticks([])
# axs[1,0].set_xticks([])
# axs[2,0].set_xticks([])
# axs[0,1].set_xticks([])
# axs[1,1].set_xticks([])
# axs[2,1].set_xticks([])
# axs[0,2].set_xticks([])
# axs[1,2].set_xticks([])
# axs[2,2].set_xticks([])
# axs[0,0].set_xticklabels([])
# axs[1,0].set_xticklabels([])
# axs[2,0].set_xticklabels([])
# axs[0,1].set_xticklabels([])
# axs[1,1].set_xticklabels([])
# axs[2,1].set_xticklabels([])
# axs[0,2].set_xticklabels([])
# axs[1,2].set_xticklabels([])
# axs[2,2].set_xticklabels([])
# axs[0,1].set_yticklabels([])
# axs[0,2].set_yticklabels([])
# axs[1,1].set_yticklabels([])
# axs[1,2].set_yticklabels([])
# axs[2,1].set_yticklabels([])
# axs[2,2].set_yticklabels([])
# axs[3,1].set_yticklabels([])
# axs[3,2].set_yticklabels([])
# axs[0,0].set(ylabel='Position [m]')
# axs[1,0].set(ylabel='Euler angles [deg]')
# axs[2,0].set(ylabel='Position [m]')
# axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
# axs[3,1].set(xlabel='Time [s]')
# axs[3,2].set(xlabel='Time [s]')
# axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
#           fancybox=True, shadow=True, ncol=3)
# fig_summary.align_ylabels(axs[:, 0])
# fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
# plt.show()

dragon_ekom_error_trans_x = computeEuclideanDistance(dragon_gt_trans_x['x'], dragon_gt_trans_x['y'], dragon_gt_trans_x['z'], dragon_ekom_trans_x_resampled_x, dragon_ekom_trans_x_resampled_y, dragon_ekom_trans_x_resampled_z)
dragon_ekom_error_trans_y = computeEuclideanDistance(dragon_gt_trans_y['x'], dragon_gt_trans_y['y'], dragon_gt_trans_y['z'], dragon_ekom_trans_y_resampled_x, dragon_ekom_trans_y_resampled_y, dragon_ekom_trans_y_resampled_z)
dragon_ekom_error_trans_z = computeEuclideanDistance(dragon_gt_trans_z['x'], dragon_gt_trans_z['y'], dragon_gt_trans_z['z'], dragon_ekom_trans_z_resampled_x, dragon_ekom_trans_z_resampled_y, dragon_ekom_trans_z_resampled_z)
dragon_ekom_error_roll = computeEuclideanDistance(dragon_gt_roll['x'], dragon_gt_roll['y'], dragon_gt_roll['z'], dragon_ekom_roll_resampled_x, dragon_ekom_roll_resampled_y, dragon_ekom_roll_resampled_z)
dragon_ekom_error_pitch = computeEuclideanDistance(dragon_gt_pitch['x'], dragon_gt_pitch['y'], dragon_gt_pitch['z'], dragon_ekom_pitch_resampled_x, dragon_ekom_pitch_resampled_y, dragon_ekom_pitch_resampled_z)
dragon_ekom_error_yaw = computeEuclideanDistance(dragon_gt_yaw['x'], dragon_gt_yaw['y'], dragon_gt_yaw['z'], dragon_ekom_yaw_resampled_x, dragon_ekom_yaw_resampled_y, dragon_ekom_yaw_resampled_z)

dragon_ekom_q_angle_trans_x = computeQuaternionError(dragon_ekom_trans_x_resampled_qx, dragon_ekom_trans_x_resampled_qy, dragon_ekom_trans_x_resampled_qz, dragon_ekom_trans_x_resampled_qw, dragon_gt_trans_x['qx'], dragon_gt_trans_x['qy'], dragon_gt_trans_x['qz'], dragon_gt_trans_x['qw'])
dragon_ekom_q_angle_trans_y = computeQuaternionError(dragon_ekom_trans_y_resampled_qx, dragon_ekom_trans_y_resampled_qy, dragon_ekom_trans_y_resampled_qz, dragon_ekom_trans_y_resampled_qw, dragon_gt_trans_y['qx'], dragon_gt_trans_y['qy'], dragon_gt_trans_y['qz'], dragon_gt_trans_y['qw'])
dragon_ekom_q_angle_trans_z = computeQuaternionError(dragon_ekom_trans_z_resampled_qx, dragon_ekom_trans_z_resampled_qy, dragon_ekom_trans_z_resampled_qz, dragon_ekom_trans_z_resampled_qw, dragon_gt_trans_z['qx'], dragon_gt_trans_z['qy'], dragon_gt_trans_z['qz'], dragon_gt_trans_z['qw'])
dragon_ekom_q_angle_roll = computeQuaternionError(dragon_ekom_roll_resampled_qx, dragon_ekom_roll_resampled_qy, dragon_ekom_roll_resampled_qz, dragon_ekom_roll_resampled_qw, dragon_gt_roll['qx'], dragon_gt_roll['qy'], dragon_gt_roll['qz'], dragon_gt_roll['qw'])
dragon_ekom_q_angle_pitch = computeQuaternionError(dragon_ekom_pitch_resampled_qx, dragon_ekom_pitch_resampled_qy, dragon_ekom_pitch_resampled_qz, dragon_ekom_pitch_resampled_qw, dragon_gt_pitch['qx'], dragon_gt_pitch['qy'], dragon_gt_pitch['qz'], dragon_gt_pitch['qw'])
dragon_ekom_q_angle_yaw = computeQuaternionError(dragon_ekom_yaw_resampled_qx, dragon_ekom_yaw_resampled_qy, dragon_ekom_yaw_resampled_qz, dragon_ekom_yaw_resampled_qw, dragon_gt_yaw['qx'], dragon_gt_yaw['qy'], dragon_gt_yaw['qz'], dragon_gt_yaw['qw'])

dragon_ekom_position_errors = np.concatenate((dragon_ekom_error_trans_x, dragon_ekom_error_trans_y, dragon_ekom_error_trans_z, dragon_ekom_error_roll, dragon_ekom_error_pitch, dragon_ekom_error_yaw))
dragon_ekom_rotation_errors = np.concatenate((dragon_ekom_q_angle_trans_x, dragon_ekom_q_angle_trans_y, dragon_ekom_q_angle_trans_z, dragon_ekom_q_angle_roll, dragon_ekom_q_angle_pitch, dragon_ekom_q_angle_yaw))




# ---------------------------------------------------------------------------  GELATIN  ---------------------------------------------------------------------------------------

filePath_dataset = '/data/gelatin/'
gelatin_gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, '009_gelatin_translation_x_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, '009_gelatin_translation_y_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, '009_gelatin_translation_z_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_gt_roll = np.genfromtxt(os.path.join(filePath_dataset, '009_gelatin_roll_2_rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, '009_gelatin_pitch_2_rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, '009_gelatin_yaw_2_rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

gelatin_gt_trans_x['t'] = (gelatin_gt_trans_x['t']-gelatin_gt_trans_x['t'][0])*10
gelatin_gt_trans_x['x'] = gelatin_gt_trans_x['x']*0.01
gelatin_gt_trans_x['y'] = gelatin_gt_trans_x['y']*0.01
gelatin_gt_trans_x['z'] = gelatin_gt_trans_x['z']*0.01

gelatin_gt_trans_y['t'] = (gelatin_gt_trans_y['t']-gelatin_gt_trans_y['t'][0])*10
gelatin_gt_trans_y['x'] = gelatin_gt_trans_y['x']*0.01
gelatin_gt_trans_y['y'] = gelatin_gt_trans_y['y']*0.01
gelatin_gt_trans_y['z'] = gelatin_gt_trans_y['z']*0.01

gelatin_gt_trans_z['t'] = (gelatin_gt_trans_z['t']-gelatin_gt_trans_z['t'][0])*10
gelatin_gt_trans_z['x'] = gelatin_gt_trans_z['x']*0.01
gelatin_gt_trans_z['y'] = gelatin_gt_trans_z['y']*0.01
gelatin_gt_trans_z['z'] = gelatin_gt_trans_z['z']*0.01

gelatin_gt_roll['t'] = (gelatin_gt_roll['t']-gelatin_gt_roll['t'][0])*10
gelatin_gt_roll['x'] = gelatin_gt_roll['x']*0.01
gelatin_gt_roll['y'] = gelatin_gt_roll['y']*0.01
gelatin_gt_roll['z'] = gelatin_gt_roll['z']*0.01

gelatin_gt_pitch['t'] = (gelatin_gt_pitch['t']-gelatin_gt_pitch['t'][0])*10
gelatin_gt_pitch['x'] = gelatin_gt_pitch['x']*0.01
gelatin_gt_pitch['y'] = gelatin_gt_pitch['y']*0.01
gelatin_gt_pitch['z'] = gelatin_gt_pitch['z']*0.01

gelatin_gt_yaw['t'] = (gelatin_gt_yaw['t']-gelatin_gt_yaw['t'][0])*10
gelatin_gt_yaw['x'] = gelatin_gt_yaw['x']*0.01
gelatin_gt_yaw['y'] = gelatin_gt_yaw['y']*0.01
gelatin_gt_yaw['z'] = gelatin_gt_yaw['z']*0.01

gelatin_ekom_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_ekom_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_ekom_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_ekom_roll = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_ekom_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_ekom_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

gelatin_ekom_trans_x_resampled_x = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['x'])
gelatin_ekom_trans_x_resampled_y = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['y'])
gelatin_ekom_trans_x_resampled_z = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['z'])
gelatin_ekom_trans_x_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['qx'])
gelatin_ekom_trans_x_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['qy'])
gelatin_ekom_trans_x_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['qz'])
gelatin_ekom_trans_x_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['qw'])

gelatin_ekom_trans_y_resampled_x = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['x'])
gelatin_ekom_trans_y_resampled_y = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['y'])
gelatin_ekom_trans_y_resampled_z = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['z'])
gelatin_ekom_trans_y_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['qx'])
gelatin_ekom_trans_y_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['qy'])
gelatin_ekom_trans_y_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['qz'])
gelatin_ekom_trans_y_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['qw'])

gelatin_ekom_trans_z_resampled_x = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['x'])
gelatin_ekom_trans_z_resampled_y = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['y'])
gelatin_ekom_trans_z_resampled_z = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['z'])
gelatin_ekom_trans_z_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['qx'])
gelatin_ekom_trans_z_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['qy'])
gelatin_ekom_trans_z_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['qz'])
gelatin_ekom_trans_z_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['qw'])

gelatin_ekom_roll_resampled_x = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_ekom_roll['t'], gelatin_ekom_roll['x'])
gelatin_ekom_roll_resampled_y = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_ekom_roll['t'], gelatin_ekom_roll['y'])
gelatin_ekom_roll_resampled_z = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_ekom_roll['t'], gelatin_ekom_roll['z'])
gelatin_ekom_roll_resampled_qx = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_ekom_roll['t'], gelatin_ekom_roll['qx'])
gelatin_ekom_roll_resampled_qy = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_ekom_roll['t'], gelatin_ekom_roll['qy'])
gelatin_ekom_roll_resampled_qz = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_ekom_roll['t'], gelatin_ekom_roll['qz'])
gelatin_ekom_roll_resampled_qw = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_ekom_roll['t'], gelatin_ekom_roll['qw'])

gelatin_ekom_pitch_resampled_x = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_ekom_pitch['t'], gelatin_ekom_pitch['x'])
gelatin_ekom_pitch_resampled_y = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_ekom_pitch['t'], gelatin_ekom_pitch['y'])
gelatin_ekom_pitch_resampled_z = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_ekom_pitch['t'], gelatin_ekom_pitch['z'])
gelatin_ekom_pitch_resampled_qx = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_ekom_pitch['t'], gelatin_ekom_pitch['qx'])
gelatin_ekom_pitch_resampled_qy = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_ekom_pitch['t'], gelatin_ekom_pitch['qy'])
gelatin_ekom_pitch_resampled_qz = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_ekom_pitch['t'], gelatin_ekom_pitch['qz'])
gelatin_ekom_pitch_resampled_qw = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_ekom_pitch['t'], gelatin_ekom_pitch['qw'])

gelatin_ekom_yaw_resampled_x = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_ekom_yaw['t'], gelatin_ekom_yaw['x'])
gelatin_ekom_yaw_resampled_y = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_ekom_yaw['t'], gelatin_ekom_yaw['y'])
gelatin_ekom_yaw_resampled_z = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_ekom_yaw['t'], gelatin_ekom_yaw['z'])
gelatin_ekom_yaw_resampled_qx = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_ekom_yaw['t'], gelatin_ekom_yaw['qx'])
gelatin_ekom_yaw_resampled_qy = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_ekom_yaw['t'], gelatin_ekom_yaw['qy'])
gelatin_ekom_yaw_resampled_qz = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_ekom_yaw['t'], gelatin_ekom_yaw['qz'])
gelatin_ekom_yaw_resampled_qw = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_ekom_yaw['t'], gelatin_ekom_yaw['qw'])

gelatin_ekom_trans_x_alpha,gelatin_ekom_trans_x_beta,gelatin_ekom_trans_x_gamma = quaternion_to_euler_angle(gelatin_ekom_trans_x['qw'], gelatin_ekom_trans_x['qx'], gelatin_ekom_trans_x['qy'], gelatin_ekom_trans_x['qz'])
gelatin_gt_trans_x_alpha,gelatin_gt_trans_x_beta,gelatin_gt_trans_x_gamma = quaternion_to_euler_angle(gelatin_gt_trans_x['qw'], gelatin_gt_trans_x['qx'], gelatin_gt_trans_x['qy'], gelatin_gt_trans_x['qz'])

gelatin_ekom_trans_x_alpha_cleaned = cleanEuler(gelatin_ekom_trans_x_alpha,0)
gelatin_ekom_trans_x_beta_cleaned = cleanEuler(gelatin_ekom_trans_x_beta,1)
gelatin_ekom_trans_x_gamma_cleaned = cleanEuler(gelatin_ekom_trans_x_gamma,2)

gelatin_gt_trans_x_alpha_cleaned = cleanEuler(gelatin_gt_trans_x_alpha,0)
gelatin_gt_trans_x_beta_cleaned = cleanEuler(gelatin_gt_trans_x_beta,1)
gelatin_gt_trans_x_gamma_cleaned = cleanEuler(gelatin_gt_trans_x_gamma,1)

gelatin_ekom_trans_y_alpha,gelatin_ekom_trans_y_beta,gelatin_ekom_trans_y_gamma = quaternion_to_euler_angle(gelatin_ekom_trans_y['qw'], gelatin_ekom_trans_y['qx'], gelatin_ekom_trans_y['qy'], gelatin_ekom_trans_y['qz'])
gelatin_gt_trans_y_alpha,gelatin_gt_trans_y_beta,gelatin_gt_trans_y_gamma = quaternion_to_euler_angle(gelatin_gt_trans_y['qw'], gelatin_gt_trans_y['qx'], gelatin_gt_trans_y['qy'], gelatin_gt_trans_y['qz'])

gelatin_ekom_trans_y_alpha_cleaned = cleanEuler(gelatin_ekom_trans_y_alpha,0)
gelatin_ekom_trans_y_beta_cleaned = cleanEuler(gelatin_ekom_trans_y_beta,1)
gelatin_ekom_trans_y_gamma_cleaned = cleanEuler(gelatin_ekom_trans_y_gamma,2)

gelatin_gt_trans_y_alpha_cleaned = cleanEuler(gelatin_gt_trans_y_alpha,0)
gelatin_gt_trans_y_beta_cleaned = cleanEuler(gelatin_gt_trans_y_beta,1)
gelatin_gt_trans_y_gamma_cleaned = cleanEuler(gelatin_gt_trans_y_gamma,2)

gelatin_ekom_trans_z_alpha,gelatin_ekom_trans_z_beta,gelatin_ekom_trans_z_gamma = quaternion_to_euler_angle(gelatin_ekom_trans_z['qw'], gelatin_ekom_trans_z['qx'], gelatin_ekom_trans_z['qy'], gelatin_ekom_trans_z['qz'])
gelatin_gt_trans_z_alpha,gelatin_gt_trans_z_beta,gelatin_gt_trans_z_gamma = quaternion_to_euler_angle(gelatin_gt_trans_z['qw'], gelatin_gt_trans_z['qx'], gelatin_gt_trans_z['qy'], gelatin_gt_trans_z['qz'])

gelatin_ekom_trans_z_alpha_cleaned = cleanEuler(gelatin_ekom_trans_z_alpha,0)
gelatin_ekom_trans_z_beta_cleaned = cleanEuler(gelatin_ekom_trans_z_beta,1)
gelatin_ekom_trans_z_gamma_cleaned = cleanEuler(gelatin_ekom_trans_z_gamma,2)

gelatin_gt_trans_z_alpha_cleaned = cleanEuler(gelatin_gt_trans_z_alpha,0)
gelatin_gt_trans_z_beta_cleaned = cleanEuler(gelatin_gt_trans_z_beta,1)
gelatin_gt_trans_z_gamma_cleaned = cleanEuler(gelatin_gt_trans_z_gamma,2)

gelatin_ekom_roll_alpha,gelatin_ekom_roll_beta,gelatin_ekom_roll_gamma = quaternion_to_euler_angle(gelatin_ekom_roll['qw'], gelatin_ekom_roll['qx'], gelatin_ekom_roll['qy'], gelatin_ekom_roll['qz'])
gelatin_gt_roll_alpha,gelatin_gt_roll_beta,gelatin_gt_roll_gamma = quaternion_to_euler_angle(gelatin_gt_roll['qw'], gelatin_gt_roll['qx'], gelatin_gt_roll['qy'], gelatin_gt_roll['qz'])

gelatin_ekom_roll_alpha_cleaned = cleanEuler(gelatin_ekom_roll_alpha,0)
gelatin_ekom_roll_beta_cleaned = cleanEuler(gelatin_ekom_roll_beta,1)
gelatin_ekom_roll_gamma_cleaned = cleanEuler(gelatin_ekom_roll_gamma,2)

gelatin_gt_roll_alpha_cleaned = cleanEuler(gelatin_gt_roll_alpha,0)
gelatin_gt_roll_beta_cleaned = cleanEuler(gelatin_gt_roll_beta,1)
gelatin_gt_roll_gamma_cleaned = cleanEuler(gelatin_gt_roll_gamma,2)

gelatin_ekom_pitch_alpha,gelatin_ekom_pitch_beta,gelatin_ekom_pitch_gamma = quaternion_to_euler_angle(gelatin_ekom_pitch['qw'], gelatin_ekom_pitch['qx'], gelatin_ekom_pitch['qy'], gelatin_ekom_pitch['qz'])
gelatin_gt_pitch_alpha,gelatin_gt_pitch_beta,gelatin_gt_pitch_gamma = quaternion_to_euler_angle(gelatin_gt_pitch['qw'], gelatin_gt_pitch['qx'], gelatin_gt_pitch['qy'], gelatin_gt_pitch['qz'])

gelatin_ekom_pitch_alpha_cleaned = cleanEuler(gelatin_ekom_pitch_alpha,0)
gelatin_ekom_pitch_beta_cleaned = cleanEuler(gelatin_ekom_pitch_beta,1)
gelatin_ekom_pitch_gamma_cleaned = cleanEuler(gelatin_ekom_pitch_gamma,2)

gelatin_gt_pitch_alpha_cleaned = cleanEuler(gelatin_gt_pitch_alpha,0)
gelatin_gt_pitch_beta_cleaned = cleanEuler(gelatin_gt_pitch_beta,1)
gelatin_gt_pitch_gamma_cleaned = cleanEuler(gelatin_gt_pitch_gamma,2)

gelatin_ekom_yaw_alpha,gelatin_ekom_yaw_beta,gelatin_ekom_yaw_gamma = quaternion_to_euler_angle(gelatin_ekom_yaw['qw'], gelatin_ekom_yaw['qx'], gelatin_ekom_yaw['qy'], gelatin_ekom_yaw['qz'])
gelatin_gt_yaw_alpha,gelatin_gt_yaw_beta,gelatin_gt_yaw_gamma = quaternion_to_euler_angle(gelatin_gt_yaw['qw'], gelatin_gt_yaw['qx'], gelatin_gt_yaw['qy'], gelatin_gt_yaw['qz'])

gelatin_ekom_yaw_alpha_cleaned = cleanEuler(gelatin_ekom_yaw_alpha,0)
gelatin_ekom_yaw_beta_cleaned = cleanEuler(gelatin_ekom_yaw_beta,1)
gelatin_ekom_yaw_gamma_cleaned = cleanEuler(gelatin_ekom_yaw_gamma,2)

gelatin_gt_yaw_alpha_cleaned = cleanEuler(gelatin_gt_yaw_alpha,0)
gelatin_gt_yaw_beta_cleaned = cleanEuler(gelatin_gt_yaw_beta,1)
gelatin_gt_yaw_gamma_cleaned = cleanEuler(gelatin_gt_yaw_gamma,2)

# fig_summary, axs = plt.subplots(4,3)
# fig_summary.set_size_inches(18, 12)
# axs[0,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['x'], color=color_x, label='x')
# axs[0,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['y'], color=color_y, label='y')
# axs[0,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['z'], color=color_z, label='z')
# axs[0,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x['x'], color=color_x, ls='--')
# axs[0,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x['y'], color=color_y, ls='--')
# axs[0,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x['z'], color=color_z, ls='--')
# axs[0,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['x'], color=color_x, label='x')
# axs[0,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['y'], color=color_y, label='y')
# axs[0,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['z'], color=color_z, label='z')
# axs[0,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y['x'], color=color_x, ls='--')
# axs[0,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y['y'], color=color_y, ls='--')
# axs[0,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y['z'], color=color_z, ls='--')
# axs[0,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['x'], color=color_x, label='x')
# axs[0,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['y'], color=color_y, label='y')
# axs[0,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['z'], color=color_z, label='z')
# axs[0,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z['x'], color=color_x, ls='--')
# axs[0,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z['y'], color=color_y, ls='--')
# axs[0,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z['z'], color=color_z, ls='--')
# axs[2,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll['x'], color=color_x, label='x')
# axs[2,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll['y'], color=color_y, label='y')
# axs[2,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll['z'], color=color_z, label='z')
# axs[2,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll['x'], color=color_x, ls='--')
# axs[2,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll['y'], color=color_y, ls='--')
# axs[2,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll['z'], color=color_z, ls='--')
# axs[2,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch['x'], color=color_x, label='x')
# axs[2,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch['y'], color=color_y, label='y')
# axs[2,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch['z'], color=color_z, label='z')
# axs[2,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch['x'], color=color_x, ls='--')
# axs[2,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch['y'], color=color_y, ls='--')
# axs[2,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch['z'], color=color_z, ls='--')
# axs[2,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw['x'], color=color_x, label='x')
# axs[2,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw['y'], color=color_y, label='y')
# axs[2,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw['z'], color=color_z, label='z')
# axs[2,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw['x'], color=color_x, ls='--')
# axs[2,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw['y'], color=color_y, ls='--')
# axs[2,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw['z'], color=color_z, ls='--')
# axs[1,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x_alpha_cleaned, color=color_x, label='qx')
# axs[1,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x_beta_cleaned, color=color_y, label='qy')
# axs[1,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x_gamma_cleaned, color=color_z, label='qz')
# axs[1,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
# axs[1,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x_beta_cleaned, color=color_y, ls = '--')
# axs[1,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
# axs[1,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y_alpha_cleaned, color=color_x, label='qx')
# axs[1,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y_beta_cleaned, color=color_y, label='qy')
# axs[1,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y_gamma_cleaned, color=color_z, label='qz')
# axs[1,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
# axs[1,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y_beta_cleaned, color=color_y, ls = '--')
# axs[1,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
# axs[1,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z_alpha_cleaned, color=color_x, label='qx')
# axs[1,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z_beta_cleaned, color=color_y, label='qy')
# axs[1,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z_gamma_cleaned, color=color_z, label='qz')
# axs[1,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
# axs[1,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z_beta_cleaned, color=color_y, ls = '--')
# axs[1,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
# axs[3,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll_alpha_cleaned, color=color_x, label='qx')
# axs[3,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll_beta_cleaned, color=color_y, label='qy')
# axs[3,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll_gamma_cleaned, color=color_z, label='qz')
# axs[3,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll_alpha_cleaned, color=color_x, ls = '--')
# axs[3,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll_beta_cleaned, color=color_y, ls = '--')
# axs[3,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll_gamma_cleaned, color=color_z, ls = '--')
# axs[3,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - gelatin_ekom")
# axs[3,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - gelatin_ekom")
# axs[3,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - gelatin_ekom")
# axs[3,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - gelatin_gt")
# axs[3,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - gelatin_gt")
# axs[3,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - gelatin_gt")
# axs[3,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw_alpha_cleaned, color=color_x, label='x or roll')
# axs[3,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw_beta_cleaned, color=color_y, label='y or pitch')
# axs[3,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw_gamma_cleaned, color=color_z, label='z or yaw')
# axs[3,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw_alpha_cleaned, color=color_x, ls = '--')
# axs[3,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw_beta_cleaned, color=color_y, ls = '--')
# axs[3,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw_gamma_cleaned, color=color_z, ls = '--')
# for i in range(0, 4):
#     for j in range(0, 3):
#         axs[i,j].set_xlim([-2, 50])
# axs[0,0].set_ylim(-0.3,  1)
# axs[0,1].set_ylim(-0.3,  1)
# axs[0,2].set_ylim(-0.3,  1)
# axs[2,0].set_ylim(-0.1,  0.9)
# axs[2,1].set_ylim(-0.1,  0.9)
# axs[2,2].set_ylim(-0.1,  0.9)
# axs[1,0].set_ylim(-120,  200)
# axs[1,1].set_ylim(-120,  200)
# axs[1,2].set_ylim(-120,  200)
# axs[3,0].set_ylim(-200,  300)
# axs[3,1].set_ylim(-200,  300)
# axs[3,2].set_ylim(-200,  300)
# axs[0,0].set_xticks([])
# axs[1,0].set_xticks([])
# axs[2,0].set_xticks([])
# axs[0,1].set_xticks([])
# axs[1,1].set_xticks([])
# axs[2,1].set_xticks([])
# axs[0,2].set_xticks([])
# axs[1,2].set_xticks([])
# axs[2,2].set_xticks([])
# axs[0,0].set_xticklabels([])
# axs[1,0].set_xticklabels([])
# axs[2,0].set_xticklabels([])
# axs[0,1].set_xticklabels([])
# axs[1,1].set_xticklabels([])
# axs[2,1].set_xticklabels([])
# axs[0,2].set_xticklabels([])
# axs[1,2].set_xticklabels([])
# axs[2,2].set_xticklabels([])
# axs[0,1].set_yticklabels([])
# axs[0,2].set_yticklabels([])
# axs[1,1].set_yticklabels([])
# axs[1,2].set_yticklabels([])
# axs[2,1].set_yticklabels([])
# axs[2,2].set_yticklabels([])
# axs[3,1].set_yticklabels([])
# axs[3,2].set_yticklabels([])
# axs[0,0].set(ylabel='Position [m]')
# axs[1,0].set(ylabel='Euler angles [deg]')
# axs[2,0].set(ylabel='Position [m]')
# axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
# axs[3,1].set(xlabel='Time [s]')
# axs[3,2].set(xlabel='Time [s]')
# axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
#           fancybox=True, shadow=True, ncol=3)
# fig_summary.align_ylabels(axs[:, 0])
# fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
# plt.show()

gelatin_ekom_error_trans_x = computeEuclideanDistance(gelatin_gt_trans_x['x'], gelatin_gt_trans_x['y'], gelatin_gt_trans_x['z'], gelatin_ekom_trans_x_resampled_x, gelatin_ekom_trans_x_resampled_y, gelatin_ekom_trans_x_resampled_z)
gelatin_ekom_error_trans_y = computeEuclideanDistance(gelatin_gt_trans_y['x'], gelatin_gt_trans_y['y'], gelatin_gt_trans_y['z'], gelatin_ekom_trans_y_resampled_x, gelatin_ekom_trans_y_resampled_y, gelatin_ekom_trans_y_resampled_z)
gelatin_ekom_error_trans_z = computeEuclideanDistance(gelatin_gt_trans_z['x'], gelatin_gt_trans_z['y'], gelatin_gt_trans_z['z'], gelatin_ekom_trans_z_resampled_x, gelatin_ekom_trans_z_resampled_y, gelatin_ekom_trans_z_resampled_z)
gelatin_ekom_error_roll = computeEuclideanDistance(gelatin_gt_roll['x'], gelatin_gt_roll['y'], gelatin_gt_roll['z'], gelatin_ekom_roll_resampled_x, gelatin_ekom_roll_resampled_y, gelatin_ekom_roll_resampled_z)
gelatin_ekom_error_pitch = computeEuclideanDistance(gelatin_gt_pitch['x'], gelatin_gt_pitch['y'], gelatin_gt_pitch['z'], gelatin_ekom_pitch_resampled_x, gelatin_ekom_pitch_resampled_y, gelatin_ekom_pitch_resampled_z)
gelatin_ekom_error_yaw = computeEuclideanDistance(gelatin_gt_yaw['x'], gelatin_gt_yaw['y'], gelatin_gt_yaw['z'], gelatin_ekom_yaw_resampled_x, gelatin_ekom_yaw_resampled_y, gelatin_ekom_yaw_resampled_z)

gelatin_ekom_q_angle_trans_x = computeQuaternionError(gelatin_ekom_trans_x_resampled_qx, gelatin_ekom_trans_x_resampled_qy, gelatin_ekom_trans_x_resampled_qz, gelatin_ekom_trans_x_resampled_qw, gelatin_gt_trans_x['qx'], gelatin_gt_trans_x['qy'], gelatin_gt_trans_x['qz'], gelatin_gt_trans_x['qw'])
gelatin_ekom_q_angle_trans_y = computeQuaternionError(gelatin_ekom_trans_y_resampled_qx, gelatin_ekom_trans_y_resampled_qy, gelatin_ekom_trans_y_resampled_qz, gelatin_ekom_trans_y_resampled_qw, gelatin_gt_trans_y['qx'], gelatin_gt_trans_y['qy'], gelatin_gt_trans_y['qz'], gelatin_gt_trans_y['qw'])
gelatin_ekom_q_angle_trans_z = computeQuaternionError(gelatin_ekom_trans_z_resampled_qx, gelatin_ekom_trans_z_resampled_qy, gelatin_ekom_trans_z_resampled_qz, gelatin_ekom_trans_z_resampled_qw, gelatin_gt_trans_z['qx'], gelatin_gt_trans_z['qy'], gelatin_gt_trans_z['qz'], gelatin_gt_trans_z['qw'])
gelatin_ekom_q_angle_roll = computeQuaternionError(gelatin_ekom_roll_resampled_qx, gelatin_ekom_roll_resampled_qy, gelatin_ekom_roll_resampled_qz, gelatin_ekom_roll_resampled_qw, gelatin_gt_roll['qx'], gelatin_gt_roll['qy'], gelatin_gt_roll['qz'], gelatin_gt_roll['qw'])
gelatin_ekom_q_angle_pitch = computeQuaternionError(gelatin_ekom_pitch_resampled_qx, gelatin_ekom_pitch_resampled_qy, gelatin_ekom_pitch_resampled_qz, gelatin_ekom_pitch_resampled_qw, gelatin_gt_pitch['qx'], gelatin_gt_pitch['qy'], gelatin_gt_pitch['qz'], gelatin_gt_pitch['qw'])
gelatin_ekom_q_angle_yaw = computeQuaternionError(gelatin_ekom_yaw_resampled_qx, gelatin_ekom_yaw_resampled_qy, gelatin_ekom_yaw_resampled_qz, gelatin_ekom_yaw_resampled_qw, gelatin_gt_yaw['qx'], gelatin_gt_yaw['qy'], gelatin_gt_yaw['qz'], gelatin_gt_yaw['qw'])

gelatin_ekom_position_errors = np.concatenate((gelatin_ekom_error_trans_x, gelatin_ekom_error_trans_y, gelatin_ekom_error_trans_z, gelatin_ekom_error_roll, gelatin_ekom_error_pitch, gelatin_ekom_error_yaw))
gelatin_ekom_rotation_errors = np.concatenate((gelatin_ekom_q_angle_trans_x, gelatin_ekom_q_angle_trans_y, gelatin_ekom_q_angle_trans_z, gelatin_ekom_q_angle_roll, gelatin_ekom_q_angle_pitch, gelatin_ekom_q_angle_yaw))

labels = ['dragon', 'gelatin']
ticks=[0, 1]
medianprops = dict(color='white')

all_objects_position_errors = [dragon_ekom_position_errors, gelatin_ekom_position_errors]
all_objects_rotation_errors = [dragon_ekom_rotation_errors, gelatin_ekom_rotation_errors]
# new_quart_array = np.array(quart_vec_pos).transpose

fig15, ax1 = plt.subplots(1,2)
ax1[0].set_xlabel('Position error [m]', color='k')
ax1[1].set_xlabel('Rotation error [rad]', color='k')
res1 = ax1[0].boxplot(all_objects_position_errors, labels=labels, vert=False, showfliers=False,
                    patch_artist=True, medianprops=medianprops)
res2 = ax1[1].boxplot(all_objects_rotation_errors, vert=False, showfliers=False,
                    patch_artist=True, medianprops=medianprops)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(res1[element], color='k')
    plt.setp(res2[element], color='k')
ax1[1].set_yticklabels([])
ax1[1].set_yticks([])
colors=[color_x, color_y]
# color='white'
# colors = [color, color, color, color, color, color,color, color,color, color,color, color]
# patterns=[0,1,0,1,0,1,0,1,0,1,0,1]
for patch, color in zip(res1['boxes'], colors):
    patch.set_facecolor(color)
    # if pattern == 1:
    #     patch.set(hatch = '/')
for patch, color in zip(res2['boxes'], colors):
    patch.set_facecolor(color)
    # if pattern == 1:
    #     patch.set(hatch = '/')
fig15.subplots_adjust(wspace=0)
plt.show()
