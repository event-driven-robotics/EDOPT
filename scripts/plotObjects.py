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

color_ekom = '#5A8CFF'
color_rgbde = '#FF4782'

color_potted = '#005175' 
color_mustard = '#08FBFB'
color_gelatin = '#08FBD3'
color_tomato = '#5770CB'
color_dragon = color_ekom

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

dragon_ekom_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/new/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_ekom_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/new/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_ekom_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/new/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_ekom_roll = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/new/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_ekom_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/new/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_ekom_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/new/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

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

fig_summary, axs = plt.subplots(4,3)
fig_summary.set_size_inches(18, 12)
axs[0,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['x'], color=color_x, label='x')
axs[0,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['y'], color=color_y, label='y')
axs[0,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x['z'], color=color_z, label='z')
axs[0,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x['x'], color=color_x, ls='--')
axs[0,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x['y'], color=color_y, ls='--')
axs[0,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x['z'], color=color_z, ls='--')
axs[0,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['x'], color=color_x, label='x')
axs[0,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['y'], color=color_y, label='y')
axs[0,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y['z'], color=color_z, label='z')
axs[0,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y['x'], color=color_x, ls='--')
axs[0,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y['y'], color=color_y, ls='--')
axs[0,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y['z'], color=color_z, ls='--')
axs[0,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['x'], color=color_x, label='x')
axs[0,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['y'], color=color_y, label='y')
axs[0,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z['z'], color=color_z, label='z')
axs[0,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z['x'], color=color_x, ls='--')
axs[0,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z['y'], color=color_y, ls='--')
axs[0,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z['z'], color=color_z, ls='--')
axs[2,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll['x'], color=color_x, label='x')
axs[2,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll['y'], color=color_y, label='y')
axs[2,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll['z'], color=color_z, label='z')
axs[2,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll['x'], color=color_x, ls='--')
axs[2,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll['y'], color=color_y, ls='--')
axs[2,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll['z'], color=color_z, ls='--')
axs[2,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch['x'], color=color_x, label='x')
axs[2,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch['y'], color=color_y, label='y')
axs[2,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch['z'], color=color_z, label='z')
axs[2,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch['x'], color=color_x, ls='--')
axs[2,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch['y'], color=color_y, ls='--')
axs[2,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch['z'], color=color_z, ls='--')
axs[2,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw['x'], color=color_x, label='x')
axs[2,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw['y'], color=color_y, label='y')
axs[2,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw['z'], color=color_z, label='z')
axs[2,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw['x'], color=color_x, ls='--')
axs[2,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw['y'], color=color_y, ls='--')
axs[2,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw['z'], color=color_z, ls='--')
axs[1,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x_alpha_cleaned, color=color_x, label='qx')
axs[1,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x_beta_cleaned, color=color_y, label='qy')
axs[1,0].plot(gelatin_ekom_trans_x['t'], gelatin_ekom_trans_x_gamma_cleaned, color=color_z, label='qz')
axs[1,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
axs[1,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x_beta_cleaned, color=color_y, ls = '--')
axs[1,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
axs[1,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y_alpha_cleaned, color=color_x, label='qx')
axs[1,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y_beta_cleaned, color=color_y, label='qy')
axs[1,1].plot(gelatin_ekom_trans_y['t'], gelatin_ekom_trans_y_gamma_cleaned, color=color_z, label='qz')
axs[1,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
axs[1,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y_beta_cleaned, color=color_y, ls = '--')
axs[1,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
axs[1,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z_alpha_cleaned, color=color_x, label='qx')
axs[1,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z_beta_cleaned, color=color_y, label='qy')
axs[1,2].plot(gelatin_ekom_trans_z['t'], gelatin_ekom_trans_z_gamma_cleaned, color=color_z, label='qz')
axs[1,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
axs[1,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z_beta_cleaned, color=color_y, ls = '--')
axs[1,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
axs[3,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll_alpha_cleaned, color=color_x, label='qx')
axs[3,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll_beta_cleaned, color=color_y, label='qy')
axs[3,0].plot(gelatin_ekom_roll['t'], gelatin_ekom_roll_gamma_cleaned, color=color_z, label='qz')
axs[3,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll_alpha_cleaned, color=color_x, ls = '--')
axs[3,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll_beta_cleaned, color=color_y, ls = '--')
axs[3,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll_gamma_cleaned, color=color_z, ls = '--')
axs[3,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - gelatin_ekom")
axs[3,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - gelatin_ekom")
axs[3,1].plot(gelatin_ekom_pitch['t'], gelatin_ekom_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - gelatin_ekom")
axs[3,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - gelatin_gt")
axs[3,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - gelatin_gt")
axs[3,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - gelatin_gt")
axs[3,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw_alpha_cleaned, color=color_x, label='x or roll')
axs[3,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw_beta_cleaned, color=color_y, label='y or pitch')
axs[3,2].plot(gelatin_ekom_yaw['t'], gelatin_ekom_yaw_gamma_cleaned, color=color_z, label='z or yaw')
axs[3,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw_alpha_cleaned, color=color_x, ls = '--')
axs[3,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw_beta_cleaned, color=color_y, ls = '--')
axs[3,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw_gamma_cleaned, color=color_z, ls = '--')
for i in range(0, 4):
    for j in range(0, 3):
        axs[i,j].set_xlim([-2, 50])
axs[0,0].set_ylim(-0.3,  1)
axs[0,1].set_ylim(-0.3,  1)
axs[0,2].set_ylim(-0.3,  1)
axs[2,0].set_ylim(-0.1,  0.9)
axs[2,1].set_ylim(-0.1,  0.9)
axs[2,2].set_ylim(-0.1,  0.9)
axs[1,0].set_ylim(-120,  200)
axs[1,1].set_ylim(-120,  200)
axs[1,2].set_ylim(-120,  200)
axs[3,0].set_ylim(-200,  300)
axs[3,1].set_ylim(-200,  300)
axs[3,2].set_ylim(-200,  300)
axs[0,0].set_xticks([])
axs[1,0].set_xticks([])
axs[2,0].set_xticks([])
axs[0,1].set_xticks([])
axs[1,1].set_xticks([])
axs[2,1].set_xticks([])
axs[0,2].set_xticks([])
axs[1,2].set_xticks([])
axs[2,2].set_xticks([])
axs[0,0].set_xticklabels([])
axs[1,0].set_xticklabels([])
axs[2,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[1,1].set_xticklabels([])
axs[2,1].set_xticklabels([])
axs[0,2].set_xticklabels([])
axs[1,2].set_xticklabels([])
axs[2,2].set_xticklabels([])
axs[0,1].set_yticklabels([])
axs[0,2].set_yticklabels([])
axs[1,1].set_yticklabels([])
axs[1,2].set_yticklabels([])
axs[2,1].set_yticklabels([])
axs[2,2].set_yticklabels([])
axs[3,1].set_yticklabels([])
axs[3,2].set_yticklabels([])
axs[0,0].set(ylabel='Position [m]')
axs[1,0].set(ylabel='Euler angles [deg]')
axs[2,0].set(ylabel='Position [m]')
axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
axs[3,1].set(xlabel='Time [s]')
axs[3,2].set(xlabel='Time [s]')
axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=3)
fig_summary.align_ylabels(axs[:, 0])
fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
plt.show()

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


# ---------------------------------------------------------------------------  MUSTARD  ---------------------------------------------------------------------------------------

filePath_dataset = '/data/mustard/'
mustard_gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'translation_x_gt_velocity_1m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'translation_y_gt_velocity_1m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'translation_z_gt_velocity_1m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_gt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'roll_gt_velocity_2rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'pitch_gt_velocity_2rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'yaw_gt_velocity_2rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

mustard_gt_trans_x['t'] = (mustard_gt_trans_x['t']-mustard_gt_trans_x['t'][0])*10
mustard_gt_trans_x['x'] = mustard_gt_trans_x['x']*0.01
mustard_gt_trans_x['y'] = mustard_gt_trans_x['y']*0.01
mustard_gt_trans_x['z'] = mustard_gt_trans_x['z']*0.01

mustard_gt_trans_y['t'] = (mustard_gt_trans_y['t']-mustard_gt_trans_y['t'][0])*10
mustard_gt_trans_y['x'] = mustard_gt_trans_y['x']*0.01
mustard_gt_trans_y['y'] = mustard_gt_trans_y['y']*0.01
mustard_gt_trans_y['z'] = mustard_gt_trans_y['z']*0.01

mustard_gt_trans_z['t'] = (mustard_gt_trans_z['t']-mustard_gt_trans_z['t'][0])*10
mustard_gt_trans_z['x'] = mustard_gt_trans_z['x']*0.01
mustard_gt_trans_z['y'] = mustard_gt_trans_z['y']*0.01
mustard_gt_trans_z['z'] = mustard_gt_trans_z['z']*0.01

mustard_gt_roll['t'] = (mustard_gt_roll['t']-mustard_gt_roll['t'][0])*10
mustard_gt_roll['x'] = mustard_gt_roll['x']*0.01
mustard_gt_roll['y'] = mustard_gt_roll['y']*0.01
mustard_gt_roll['z'] = mustard_gt_roll['z']*0.01

mustard_gt_pitch['t'] = (mustard_gt_pitch['t']-mustard_gt_pitch['t'][0])*10
mustard_gt_pitch['x'] = mustard_gt_pitch['x']*0.01
mustard_gt_pitch['y'] = mustard_gt_pitch['y']*0.01
mustard_gt_pitch['z'] = mustard_gt_pitch['z']*0.01

mustard_gt_yaw['t'] = (mustard_gt_yaw['t']-mustard_gt_yaw['t'][0])*10
mustard_gt_yaw['x'] = mustard_gt_yaw['x']*0.01
mustard_gt_yaw['y'] = mustard_gt_yaw['y']*0.01
mustard_gt_yaw['z'] = mustard_gt_yaw['z']*0.01

mustard_ekom_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_ekom_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_ekom_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_ekom_roll = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_ekom_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_ekom_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

mustard_ekom_trans_x_resampled_x = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_ekom_trans_x['t'], mustard_ekom_trans_x['x'])
mustard_ekom_trans_x_resampled_y = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_ekom_trans_x['t'], mustard_ekom_trans_x['y'])
mustard_ekom_trans_x_resampled_z = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_ekom_trans_x['t'], mustard_ekom_trans_x['z'])
mustard_ekom_trans_x_resampled_qx = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_ekom_trans_x['t'], mustard_ekom_trans_x['qx'])
mustard_ekom_trans_x_resampled_qy = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_ekom_trans_x['t'], mustard_ekom_trans_x['qy'])
mustard_ekom_trans_x_resampled_qz = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_ekom_trans_x['t'], mustard_ekom_trans_x['qz'])
mustard_ekom_trans_x_resampled_qw = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_ekom_trans_x['t'], mustard_ekom_trans_x['qw'])

mustard_ekom_trans_y_resampled_x = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_ekom_trans_y['t'], mustard_ekom_trans_y['x'])
mustard_ekom_trans_y_resampled_y = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_ekom_trans_y['t'], mustard_ekom_trans_y['y'])
mustard_ekom_trans_y_resampled_z = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_ekom_trans_y['t'], mustard_ekom_trans_y['z'])
mustard_ekom_trans_y_resampled_qx = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_ekom_trans_y['t'], mustard_ekom_trans_y['qx'])
mustard_ekom_trans_y_resampled_qy = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_ekom_trans_y['t'], mustard_ekom_trans_y['qy'])
mustard_ekom_trans_y_resampled_qz = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_ekom_trans_y['t'], mustard_ekom_trans_y['qz'])
mustard_ekom_trans_y_resampled_qw = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_ekom_trans_y['t'], mustard_ekom_trans_y['qw'])

mustard_ekom_trans_z_resampled_x = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_ekom_trans_z['t'], mustard_ekom_trans_z['x'])
mustard_ekom_trans_z_resampled_y = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_ekom_trans_z['t'], mustard_ekom_trans_z['y'])
mustard_ekom_trans_z_resampled_z = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_ekom_trans_z['t'], mustard_ekom_trans_z['z'])
mustard_ekom_trans_z_resampled_qx = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_ekom_trans_z['t'], mustard_ekom_trans_z['qx'])
mustard_ekom_trans_z_resampled_qy = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_ekom_trans_z['t'], mustard_ekom_trans_z['qy'])
mustard_ekom_trans_z_resampled_qz = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_ekom_trans_z['t'], mustard_ekom_trans_z['qz'])
mustard_ekom_trans_z_resampled_qw = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_ekom_trans_z['t'], mustard_ekom_trans_z['qw'])

mustard_ekom_roll_resampled_x = resampling_by_interpolate(mustard_gt_roll['t'], mustard_ekom_roll['t'], mustard_ekom_roll['x'])
mustard_ekom_roll_resampled_y = resampling_by_interpolate(mustard_gt_roll['t'], mustard_ekom_roll['t'], mustard_ekom_roll['y'])
mustard_ekom_roll_resampled_z = resampling_by_interpolate(mustard_gt_roll['t'], mustard_ekom_roll['t'], mustard_ekom_roll['z'])
mustard_ekom_roll_resampled_qx = resampling_by_interpolate(mustard_gt_roll['t'], mustard_ekom_roll['t'], mustard_ekom_roll['qx'])
mustard_ekom_roll_resampled_qy = resampling_by_interpolate(mustard_gt_roll['t'], mustard_ekom_roll['t'], mustard_ekom_roll['qy'])
mustard_ekom_roll_resampled_qz = resampling_by_interpolate(mustard_gt_roll['t'], mustard_ekom_roll['t'], mustard_ekom_roll['qz'])
mustard_ekom_roll_resampled_qw = resampling_by_interpolate(mustard_gt_roll['t'], mustard_ekom_roll['t'], mustard_ekom_roll['qw'])

mustard_ekom_pitch_resampled_x = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_ekom_pitch['t'], mustard_ekom_pitch['x'])
mustard_ekom_pitch_resampled_y = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_ekom_pitch['t'], mustard_ekom_pitch['y'])
mustard_ekom_pitch_resampled_z = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_ekom_pitch['t'], mustard_ekom_pitch['z'])
mustard_ekom_pitch_resampled_qx = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_ekom_pitch['t'], mustard_ekom_pitch['qx'])
mustard_ekom_pitch_resampled_qy = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_ekom_pitch['t'], mustard_ekom_pitch['qy'])
mustard_ekom_pitch_resampled_qz = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_ekom_pitch['t'], mustard_ekom_pitch['qz'])
mustard_ekom_pitch_resampled_qw = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_ekom_pitch['t'], mustard_ekom_pitch['qw'])

mustard_ekom_yaw_resampled_x = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_ekom_yaw['t'], mustard_ekom_yaw['x'])
mustard_ekom_yaw_resampled_y = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_ekom_yaw['t'], mustard_ekom_yaw['y'])
mustard_ekom_yaw_resampled_z = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_ekom_yaw['t'], mustard_ekom_yaw['z'])
mustard_ekom_yaw_resampled_qx = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_ekom_yaw['t'], mustard_ekom_yaw['qx'])
mustard_ekom_yaw_resampled_qy = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_ekom_yaw['t'], mustard_ekom_yaw['qy'])
mustard_ekom_yaw_resampled_qz = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_ekom_yaw['t'], mustard_ekom_yaw['qz'])
mustard_ekom_yaw_resampled_qw = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_ekom_yaw['t'], mustard_ekom_yaw['qw'])

mustard_ekom_trans_x_alpha,mustard_ekom_trans_x_beta,mustard_ekom_trans_x_gamma = quaternion_to_euler_angle(mustard_ekom_trans_x['qw'], mustard_ekom_trans_x['qx'], mustard_ekom_trans_x['qy'], mustard_ekom_trans_x['qz'])
mustard_gt_trans_x_alpha,mustard_gt_trans_x_beta,mustard_gt_trans_x_gamma = quaternion_to_euler_angle(mustard_gt_trans_x['qw'], mustard_gt_trans_x['qx'], mustard_gt_trans_x['qy'], mustard_gt_trans_x['qz'])

mustard_ekom_trans_x_alpha_cleaned = cleanEuler(mustard_ekom_trans_x_alpha,0)
mustard_ekom_trans_x_beta_cleaned = cleanEuler(mustard_ekom_trans_x_beta,1)
mustard_ekom_trans_x_gamma_cleaned = cleanEuler(mustard_ekom_trans_x_gamma,2)

mustard_gt_trans_x_alpha_cleaned = cleanEuler(mustard_gt_trans_x_alpha,0)
mustard_gt_trans_x_beta_cleaned = cleanEuler(mustard_gt_trans_x_beta,1)
mustard_gt_trans_x_gamma_cleaned = cleanEuler(mustard_gt_trans_x_gamma,1)

mustard_ekom_trans_y_alpha,mustard_ekom_trans_y_beta,mustard_ekom_trans_y_gamma = quaternion_to_euler_angle(mustard_ekom_trans_y['qw'], mustard_ekom_trans_y['qx'], mustard_ekom_trans_y['qy'], mustard_ekom_trans_y['qz'])
mustard_gt_trans_y_alpha,mustard_gt_trans_y_beta,mustard_gt_trans_y_gamma = quaternion_to_euler_angle(mustard_gt_trans_y['qw'], mustard_gt_trans_y['qx'], mustard_gt_trans_y['qy'], mustard_gt_trans_y['qz'])

mustard_ekom_trans_y_alpha_cleaned = cleanEuler(mustard_ekom_trans_y_alpha,0)
mustard_ekom_trans_y_beta_cleaned = cleanEuler(mustard_ekom_trans_y_beta,1)
mustard_ekom_trans_y_gamma_cleaned = cleanEuler(mustard_ekom_trans_y_gamma,2)

mustard_gt_trans_y_alpha_cleaned = cleanEuler(mustard_gt_trans_y_alpha,0)
mustard_gt_trans_y_beta_cleaned = cleanEuler(mustard_gt_trans_y_beta,1)
mustard_gt_trans_y_gamma_cleaned = cleanEuler(mustard_gt_trans_y_gamma,2)

mustard_ekom_trans_z_alpha,mustard_ekom_trans_z_beta,mustard_ekom_trans_z_gamma = quaternion_to_euler_angle(mustard_ekom_trans_z['qw'], mustard_ekom_trans_z['qx'], mustard_ekom_trans_z['qy'], mustard_ekom_trans_z['qz'])
mustard_gt_trans_z_alpha,mustard_gt_trans_z_beta,mustard_gt_trans_z_gamma = quaternion_to_euler_angle(mustard_gt_trans_z['qw'], mustard_gt_trans_z['qx'], mustard_gt_trans_z['qy'], mustard_gt_trans_z['qz'])

mustard_ekom_trans_z_alpha_cleaned = cleanEuler(mustard_ekom_trans_z_alpha,0)
mustard_ekom_trans_z_beta_cleaned = cleanEuler(mustard_ekom_trans_z_beta,1)
mustard_ekom_trans_z_gamma_cleaned = cleanEuler(mustard_ekom_trans_z_gamma,2)

mustard_gt_trans_z_alpha_cleaned = cleanEuler(mustard_gt_trans_z_alpha,0)
mustard_gt_trans_z_beta_cleaned = cleanEuler(mustard_gt_trans_z_beta,1)
mustard_gt_trans_z_gamma_cleaned = cleanEuler(mustard_gt_trans_z_gamma,2)

mustard_ekom_roll_alpha,mustard_ekom_roll_beta,mustard_ekom_roll_gamma = quaternion_to_euler_angle(mustard_ekom_roll['qw'], mustard_ekom_roll['qx'], mustard_ekom_roll['qy'], mustard_ekom_roll['qz'])
mustard_gt_roll_alpha,mustard_gt_roll_beta,mustard_gt_roll_gamma = quaternion_to_euler_angle(mustard_gt_roll['qw'], mustard_gt_roll['qx'], mustard_gt_roll['qy'], mustard_gt_roll['qz'])

mustard_ekom_roll_alpha_cleaned = cleanEuler(mustard_ekom_roll_alpha,0)
mustard_ekom_roll_beta_cleaned = cleanEuler(mustard_ekom_roll_beta,1)
mustard_ekom_roll_gamma_cleaned = cleanEuler(mustard_ekom_roll_gamma,2)

mustard_gt_roll_alpha_cleaned = cleanEuler(mustard_gt_roll_alpha,0)
mustard_gt_roll_beta_cleaned = cleanEuler(mustard_gt_roll_beta,1)
mustard_gt_roll_gamma_cleaned = cleanEuler(mustard_gt_roll_gamma,2)

mustard_ekom_pitch_alpha,mustard_ekom_pitch_beta,mustard_ekom_pitch_gamma = quaternion_to_euler_angle(mustard_ekom_pitch['qw'], mustard_ekom_pitch['qx'], mustard_ekom_pitch['qy'], mustard_ekom_pitch['qz'])
mustard_gt_pitch_alpha,mustard_gt_pitch_beta,mustard_gt_pitch_gamma = quaternion_to_euler_angle(mustard_gt_pitch['qw'], mustard_gt_pitch['qx'], mustard_gt_pitch['qy'], mustard_gt_pitch['qz'])

mustard_ekom_pitch_alpha_cleaned = cleanEuler(mustard_ekom_pitch_alpha,0)
mustard_ekom_pitch_beta_cleaned = cleanEuler(mustard_ekom_pitch_beta,1)
mustard_ekom_pitch_gamma_cleaned = cleanEuler(mustard_ekom_pitch_gamma,2)

mustard_gt_pitch_alpha_cleaned = cleanEuler(mustard_gt_pitch_alpha,0)
mustard_gt_pitch_beta_cleaned = cleanEuler(mustard_gt_pitch_beta,1)
mustard_gt_pitch_gamma_cleaned = cleanEuler(mustard_gt_pitch_gamma,2)

mustard_ekom_yaw_alpha,mustard_ekom_yaw_beta,mustard_ekom_yaw_gamma = quaternion_to_euler_angle(mustard_ekom_yaw['qw'], mustard_ekom_yaw['qx'], mustard_ekom_yaw['qy'], mustard_ekom_yaw['qz'])
mustard_gt_yaw_alpha,mustard_gt_yaw_beta,mustard_gt_yaw_gamma = quaternion_to_euler_angle(mustard_gt_yaw['qw'], mustard_gt_yaw['qx'], mustard_gt_yaw['qy'], mustard_gt_yaw['qz'])

mustard_ekom_yaw_alpha_cleaned = cleanEuler(mustard_ekom_yaw_alpha,0)
mustard_ekom_yaw_beta_cleaned = cleanEuler(mustard_ekom_yaw_beta,1)
mustard_ekom_yaw_gamma_cleaned = cleanEuler(mustard_ekom_yaw_gamma,2)

mustard_gt_yaw_alpha_cleaned = cleanEuler(mustard_gt_yaw_alpha,0)
mustard_gt_yaw_beta_cleaned = cleanEuler(mustard_gt_yaw_beta,1)
mustard_gt_yaw_gamma_cleaned = cleanEuler(mustard_gt_yaw_gamma,2)

fig_summary, axs = plt.subplots(4,3)
fig_summary.set_size_inches(18, 12)
axs[0,0].plot(mustard_ekom_trans_x['t'], mustard_ekom_trans_x['x'], color=color_x, label='x')
axs[0,0].plot(mustard_ekom_trans_x['t'], mustard_ekom_trans_x['y'], color=color_y, label='y')
axs[0,0].plot(mustard_ekom_trans_x['t'], mustard_ekom_trans_x['z'], color=color_z, label='z')
axs[0,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x['x'], color=color_x, ls='--')
axs[0,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x['y'], color=color_y, ls='--')
axs[0,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x['z'], color=color_z, ls='--')
axs[0,1].plot(mustard_ekom_trans_y['t'], mustard_ekom_trans_y['x'], color=color_x, label='x')
axs[0,1].plot(mustard_ekom_trans_y['t'], mustard_ekom_trans_y['y'], color=color_y, label='y')
axs[0,1].plot(mustard_ekom_trans_y['t'], mustard_ekom_trans_y['z'], color=color_z, label='z')
axs[0,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y['x'], color=color_x, ls='--')
axs[0,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y['y'], color=color_y, ls='--')
axs[0,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y['z'], color=color_z, ls='--')
axs[0,2].plot(mustard_ekom_trans_z['t'], mustard_ekom_trans_z['x'], color=color_x, label='x')
axs[0,2].plot(mustard_ekom_trans_z['t'], mustard_ekom_trans_z['y'], color=color_y, label='y')
axs[0,2].plot(mustard_ekom_trans_z['t'], mustard_ekom_trans_z['z'], color=color_z, label='z')
axs[0,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z['x'], color=color_x, ls='--')
axs[0,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z['y'], color=color_y, ls='--')
axs[0,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z['z'], color=color_z, ls='--')
axs[2,0].plot(mustard_ekom_roll['t'], mustard_ekom_roll['x'], color=color_x, label='x')
axs[2,0].plot(mustard_ekom_roll['t'], mustard_ekom_roll['y'], color=color_y, label='y')
axs[2,0].plot(mustard_ekom_roll['t'], mustard_ekom_roll['z'], color=color_z, label='z')
axs[2,0].plot(mustard_gt_roll['t'], mustard_gt_roll['x'], color=color_x, ls='--')
axs[2,0].plot(mustard_gt_roll['t'], mustard_gt_roll['y'], color=color_y, ls='--')
axs[2,0].plot(mustard_gt_roll['t'], mustard_gt_roll['z'], color=color_z, ls='--')
axs[2,1].plot(mustard_ekom_pitch['t'], mustard_ekom_pitch['x'], color=color_x, label='x')
axs[2,1].plot(mustard_ekom_pitch['t'], mustard_ekom_pitch['y'], color=color_y, label='y')
axs[2,1].plot(mustard_ekom_pitch['t'], mustard_ekom_pitch['z'], color=color_z, label='z')
axs[2,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch['x'], color=color_x, ls='--')
axs[2,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch['y'], color=color_y, ls='--')
axs[2,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch['z'], color=color_z, ls='--')
axs[2,2].plot(mustard_ekom_yaw['t'], mustard_ekom_yaw['x'], color=color_x, label='x')
axs[2,2].plot(mustard_ekom_yaw['t'], mustard_ekom_yaw['y'], color=color_y, label='y')
axs[2,2].plot(mustard_ekom_yaw['t'], mustard_ekom_yaw['z'], color=color_z, label='z')
axs[2,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw['x'], color=color_x, ls='--')
axs[2,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw['y'], color=color_y, ls='--')
axs[2,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw['z'], color=color_z, ls='--')
axs[1,0].plot(mustard_ekom_trans_x['t'], mustard_ekom_trans_x_alpha_cleaned, color=color_x, label='qx')
axs[1,0].plot(mustard_ekom_trans_x['t'], mustard_ekom_trans_x_beta_cleaned, color=color_y, label='qy')
axs[1,0].plot(mustard_ekom_trans_x['t'], mustard_ekom_trans_x_gamma_cleaned, color=color_z, label='qz')
axs[1,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
axs[1,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x_beta_cleaned, color=color_y, ls = '--')
axs[1,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
axs[1,1].plot(mustard_ekom_trans_y['t'], mustard_ekom_trans_y_alpha_cleaned, color=color_x, label='qx')
axs[1,1].plot(mustard_ekom_trans_y['t'], mustard_ekom_trans_y_beta_cleaned, color=color_y, label='qy')
axs[1,1].plot(mustard_ekom_trans_y['t'], mustard_ekom_trans_y_gamma_cleaned, color=color_z, label='qz')
axs[1,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
axs[1,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y_beta_cleaned, color=color_y, ls = '--')
axs[1,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
axs[1,2].plot(mustard_ekom_trans_z['t'], mustard_ekom_trans_z_alpha_cleaned, color=color_x, label='qx')
axs[1,2].plot(mustard_ekom_trans_z['t'], mustard_ekom_trans_z_beta_cleaned, color=color_y, label='qy')
axs[1,2].plot(mustard_ekom_trans_z['t'], mustard_ekom_trans_z_gamma_cleaned, color=color_z, label='qz')
axs[1,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
axs[1,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z_beta_cleaned, color=color_y, ls = '--')
axs[1,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
axs[3,0].plot(mustard_ekom_roll['t'], mustard_ekom_roll_alpha_cleaned, color=color_x, label='qx')
axs[3,0].plot(mustard_ekom_roll['t'], mustard_ekom_roll_beta_cleaned, color=color_y, label='qy')
axs[3,0].plot(mustard_ekom_roll['t'], mustard_ekom_roll_gamma_cleaned, color=color_z, label='qz')
axs[3,0].plot(mustard_gt_roll['t'], mustard_gt_roll_alpha_cleaned, color=color_x, ls = '--')
axs[3,0].plot(mustard_gt_roll['t'], mustard_gt_roll_beta_cleaned, color=color_y, ls = '--')
axs[3,0].plot(mustard_gt_roll['t'], mustard_gt_roll_gamma_cleaned, color=color_z, ls = '--')
axs[3,1].plot(mustard_ekom_pitch['t'], mustard_ekom_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - mustard_ekom")
axs[3,1].plot(mustard_ekom_pitch['t'], mustard_ekom_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - mustard_ekom")
axs[3,1].plot(mustard_ekom_pitch['t'], mustard_ekom_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - mustard_ekom")
axs[3,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - mustard_gt")
axs[3,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - mustard_gt")
axs[3,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - mustard_gt")
axs[3,2].plot(mustard_ekom_yaw['t'], mustard_ekom_yaw_alpha_cleaned, color=color_x, label='x or roll')
axs[3,2].plot(mustard_ekom_yaw['t'], mustard_ekom_yaw_beta_cleaned, color=color_y, label='y or pitch')
axs[3,2].plot(mustard_ekom_yaw['t'], mustard_ekom_yaw_gamma_cleaned, color=color_z, label='z or yaw')
axs[3,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw_alpha_cleaned, color=color_x, ls = '--')
axs[3,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw_beta_cleaned, color=color_y, ls = '--')
axs[3,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw_gamma_cleaned, color=color_z, ls = '--')
for i in range(0, 4):
    for j in range(0, 3):
        axs[i,j].set_xlim([-2, 50])
axs[0,0].set_ylim(-0.3,  1)
axs[0,1].set_ylim(-0.3,  1)
axs[0,2].set_ylim(-0.3,  1)
axs[2,0].set_ylim(-0.1,  0.9)
axs[2,1].set_ylim(-0.1,  0.9)
axs[2,2].set_ylim(-0.1,  0.9)
axs[1,0].set_ylim(-120,  200)
axs[1,1].set_ylim(-120,  200)
axs[1,2].set_ylim(-120,  200)
axs[3,0].set_ylim(-200,  300)
axs[3,1].set_ylim(-200,  300)
axs[3,2].set_ylim(-200,  300)
axs[0,0].set_xticks([])
axs[1,0].set_xticks([])
axs[2,0].set_xticks([])
axs[0,1].set_xticks([])
axs[1,1].set_xticks([])
axs[2,1].set_xticks([])
axs[0,2].set_xticks([])
axs[1,2].set_xticks([])
axs[2,2].set_xticks([])
axs[0,0].set_xticklabels([])
axs[1,0].set_xticklabels([])
axs[2,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[1,1].set_xticklabels([])
axs[2,1].set_xticklabels([])
axs[0,2].set_xticklabels([])
axs[1,2].set_xticklabels([])
axs[2,2].set_xticklabels([])
axs[0,1].set_yticklabels([])
axs[0,2].set_yticklabels([])
axs[1,1].set_yticklabels([])
axs[1,2].set_yticklabels([])
axs[2,1].set_yticklabels([])
axs[2,2].set_yticklabels([])
axs[3,1].set_yticklabels([])
axs[3,2].set_yticklabels([])
axs[0,0].set(ylabel='Position [m]')
axs[1,0].set(ylabel='Euler angles [deg]')
axs[2,0].set(ylabel='Position [m]')
axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
axs[3,1].set(xlabel='Time [s]')
axs[3,2].set(xlabel='Time [s]')
axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=3)
fig_summary.align_ylabels(axs[:, 0])
fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
plt.show()

mustard_ekom_error_trans_x = computeEuclideanDistance(mustard_gt_trans_x['x'], mustard_gt_trans_x['y'], mustard_gt_trans_x['z'], mustard_ekom_trans_x_resampled_x, mustard_ekom_trans_x_resampled_y, mustard_ekom_trans_x_resampled_z)
mustard_ekom_error_trans_y = computeEuclideanDistance(mustard_gt_trans_y['x'], mustard_gt_trans_y['y'], mustard_gt_trans_y['z'], mustard_ekom_trans_y_resampled_x, mustard_ekom_trans_y_resampled_y, mustard_ekom_trans_y_resampled_z)
mustard_ekom_error_trans_z = computeEuclideanDistance(mustard_gt_trans_z['x'], mustard_gt_trans_z['y'], mustard_gt_trans_z['z'], mustard_ekom_trans_z_resampled_x, mustard_ekom_trans_z_resampled_y, mustard_ekom_trans_z_resampled_z)
mustard_ekom_error_roll = computeEuclideanDistance(mustard_gt_roll['x'], mustard_gt_roll['y'], mustard_gt_roll['z'], mustard_ekom_roll_resampled_x, mustard_ekom_roll_resampled_y, mustard_ekom_roll_resampled_z)
mustard_ekom_error_pitch = computeEuclideanDistance(mustard_gt_pitch['x'], mustard_gt_pitch['y'], mustard_gt_pitch['z'], mustard_ekom_pitch_resampled_x, mustard_ekom_pitch_resampled_y, mustard_ekom_pitch_resampled_z)
mustard_ekom_error_yaw = computeEuclideanDistance(mustard_gt_yaw['x'], mustard_gt_yaw['y'], mustard_gt_yaw['z'], mustard_ekom_yaw_resampled_x, mustard_ekom_yaw_resampled_y, mustard_ekom_yaw_resampled_z)

mustard_ekom_q_angle_trans_x = computeQuaternionError(mustard_ekom_trans_x_resampled_qx, mustard_ekom_trans_x_resampled_qy, mustard_ekom_trans_x_resampled_qz, mustard_ekom_trans_x_resampled_qw, mustard_gt_trans_x['qx'], mustard_gt_trans_x['qy'], mustard_gt_trans_x['qz'], mustard_gt_trans_x['qw'])
mustard_ekom_q_angle_trans_y = computeQuaternionError(mustard_ekom_trans_y_resampled_qx, mustard_ekom_trans_y_resampled_qy, mustard_ekom_trans_y_resampled_qz, mustard_ekom_trans_y_resampled_qw, mustard_gt_trans_y['qx'], mustard_gt_trans_y['qy'], mustard_gt_trans_y['qz'], mustard_gt_trans_y['qw'])
mustard_ekom_q_angle_trans_z = computeQuaternionError(mustard_ekom_trans_z_resampled_qx, mustard_ekom_trans_z_resampled_qy, mustard_ekom_trans_z_resampled_qz, mustard_ekom_trans_z_resampled_qw, mustard_gt_trans_z['qx'], mustard_gt_trans_z['qy'], mustard_gt_trans_z['qz'], mustard_gt_trans_z['qw'])
mustard_ekom_q_angle_roll = computeQuaternionError(mustard_ekom_roll_resampled_qx, mustard_ekom_roll_resampled_qy, mustard_ekom_roll_resampled_qz, mustard_ekom_roll_resampled_qw, mustard_gt_roll['qx'], mustard_gt_roll['qy'], mustard_gt_roll['qz'], mustard_gt_roll['qw'])
mustard_ekom_q_angle_pitch = computeQuaternionError(mustard_ekom_pitch_resampled_qx, mustard_ekom_pitch_resampled_qy, mustard_ekom_pitch_resampled_qz, mustard_ekom_pitch_resampled_qw, mustard_gt_pitch['qx'], mustard_gt_pitch['qy'], mustard_gt_pitch['qz'], mustard_gt_pitch['qw'])
mustard_ekom_q_angle_yaw = computeQuaternionError(mustard_ekom_yaw_resampled_qx, mustard_ekom_yaw_resampled_qy, mustard_ekom_yaw_resampled_qz, mustard_ekom_yaw_resampled_qw, mustard_gt_yaw['qx'], mustard_gt_yaw['qy'], mustard_gt_yaw['qz'], mustard_gt_yaw['qw'])

mustard_ekom_position_errors = np.concatenate((mustard_ekom_error_trans_x, mustard_ekom_error_trans_y, mustard_ekom_error_trans_z, mustard_ekom_error_roll, mustard_ekom_error_pitch, mustard_ekom_error_yaw))
mustard_ekom_rotation_errors = np.concatenate((mustard_ekom_q_angle_trans_x, mustard_ekom_q_angle_trans_y, mustard_ekom_q_angle_trans_z, mustard_ekom_q_angle_roll, mustard_ekom_q_angle_pitch, mustard_ekom_q_angle_yaw))


# ---------------------------------------------------------------------------  TOMATO  ---------------------------------------------------------------------------------------

filePath_dataset = '/data/tomato/'
tomato_gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, '005_tomato_translation_x_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, '005_tomato_translation_y_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, '005_tomato_translation_z_1_m_s_closer/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_gt_roll = np.genfromtxt(os.path.join(filePath_dataset, '005_tomato_roll_2_rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, '005_tomato_pitch_2_rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, '005_tomato_yaw_2_rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

tomato_gt_trans_x['t'] = (tomato_gt_trans_x['t']-tomato_gt_trans_x['t'][0])*10
tomato_gt_trans_x['x'] = tomato_gt_trans_x['x']*0.01
tomato_gt_trans_x['y'] = tomato_gt_trans_x['y']*0.01
tomato_gt_trans_x['z'] = tomato_gt_trans_x['z']*0.01

tomato_gt_trans_y['t'] = (tomato_gt_trans_y['t']-tomato_gt_trans_y['t'][0])*10
tomato_gt_trans_y['x'] = tomato_gt_trans_y['x']*0.01
tomato_gt_trans_y['y'] = tomato_gt_trans_y['y']*0.01
tomato_gt_trans_y['z'] = tomato_gt_trans_y['z']*0.01

tomato_gt_trans_z['t'] = (tomato_gt_trans_z['t']-tomato_gt_trans_z['t'][0])*10
tomato_gt_trans_z['x'] = tomato_gt_trans_z['x']*0.01
tomato_gt_trans_z['y'] = tomato_gt_trans_z['y']*0.01
tomato_gt_trans_z['z'] = tomato_gt_trans_z['z']*0.01

tomato_gt_roll['t'] = (tomato_gt_roll['t']-tomato_gt_roll['t'][0])*10
tomato_gt_roll['x'] = tomato_gt_roll['x']*0.01
tomato_gt_roll['y'] = tomato_gt_roll['y']*0.01
tomato_gt_roll['z'] = tomato_gt_roll['z']*0.01

tomato_gt_pitch['t'] = (tomato_gt_pitch['t']-tomato_gt_pitch['t'][0])*10
tomato_gt_pitch['x'] = tomato_gt_pitch['x']*0.01
tomato_gt_pitch['y'] = tomato_gt_pitch['y']*0.01
tomato_gt_pitch['z'] = tomato_gt_pitch['z']*0.01

tomato_gt_yaw['t'] = (tomato_gt_yaw['t']-tomato_gt_yaw['t'][0])*10
tomato_gt_yaw['x'] = tomato_gt_yaw['x']*0.01
tomato_gt_yaw['y'] = tomato_gt_yaw['y']*0.01
tomato_gt_yaw['z'] = tomato_gt_yaw['z']*0.01

tomato_ekom_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_ekom_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_ekom_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_ekom_roll = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_ekom_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/new/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_ekom_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/new/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

tomato_ekom_trans_x_resampled_x = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_ekom_trans_x['t'], tomato_ekom_trans_x['x'])
tomato_ekom_trans_x_resampled_y = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_ekom_trans_x['t'], tomato_ekom_trans_x['y'])
tomato_ekom_trans_x_resampled_z = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_ekom_trans_x['t'], tomato_ekom_trans_x['z'])
tomato_ekom_trans_x_resampled_qx = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_ekom_trans_x['t'], tomato_ekom_trans_x['qx'])
tomato_ekom_trans_x_resampled_qy = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_ekom_trans_x['t'], tomato_ekom_trans_x['qy'])
tomato_ekom_trans_x_resampled_qz = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_ekom_trans_x['t'], tomato_ekom_trans_x['qz'])
tomato_ekom_trans_x_resampled_qw = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_ekom_trans_x['t'], tomato_ekom_trans_x['qw'])

tomato_ekom_trans_y_resampled_x = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_ekom_trans_y['t'], tomato_ekom_trans_y['x'])
tomato_ekom_trans_y_resampled_y = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_ekom_trans_y['t'], tomato_ekom_trans_y['y'])
tomato_ekom_trans_y_resampled_z = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_ekom_trans_y['t'], tomato_ekom_trans_y['z'])
tomato_ekom_trans_y_resampled_qx = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_ekom_trans_y['t'], tomato_ekom_trans_y['qx'])
tomato_ekom_trans_y_resampled_qy = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_ekom_trans_y['t'], tomato_ekom_trans_y['qy'])
tomato_ekom_trans_y_resampled_qz = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_ekom_trans_y['t'], tomato_ekom_trans_y['qz'])
tomato_ekom_trans_y_resampled_qw = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_ekom_trans_y['t'], tomato_ekom_trans_y['qw'])

tomato_ekom_trans_z_resampled_x = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_ekom_trans_z['t'], tomato_ekom_trans_z['x'])
tomato_ekom_trans_z_resampled_y = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_ekom_trans_z['t'], tomato_ekom_trans_z['y'])
tomato_ekom_trans_z_resampled_z = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_ekom_trans_z['t'], tomato_ekom_trans_z['z'])
tomato_ekom_trans_z_resampled_qx = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_ekom_trans_z['t'], tomato_ekom_trans_z['qx'])
tomato_ekom_trans_z_resampled_qy = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_ekom_trans_z['t'], tomato_ekom_trans_z['qy'])
tomato_ekom_trans_z_resampled_qz = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_ekom_trans_z['t'], tomato_ekom_trans_z['qz'])
tomato_ekom_trans_z_resampled_qw = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_ekom_trans_z['t'], tomato_ekom_trans_z['qw'])

tomato_ekom_roll_resampled_x = resampling_by_interpolate(tomato_gt_roll['t'], tomato_ekom_roll['t'], tomato_ekom_roll['x'])
tomato_ekom_roll_resampled_y = resampling_by_interpolate(tomato_gt_roll['t'], tomato_ekom_roll['t'], tomato_ekom_roll['y'])
tomato_ekom_roll_resampled_z = resampling_by_interpolate(tomato_gt_roll['t'], tomato_ekom_roll['t'], tomato_ekom_roll['z'])
tomato_ekom_roll_resampled_qx = resampling_by_interpolate(tomato_gt_roll['t'], tomato_ekom_roll['t'], tomato_ekom_roll['qx'])
tomato_ekom_roll_resampled_qy = resampling_by_interpolate(tomato_gt_roll['t'], tomato_ekom_roll['t'], tomato_ekom_roll['qy'])
tomato_ekom_roll_resampled_qz = resampling_by_interpolate(tomato_gt_roll['t'], tomato_ekom_roll['t'], tomato_ekom_roll['qz'])
tomato_ekom_roll_resampled_qw = resampling_by_interpolate(tomato_gt_roll['t'], tomato_ekom_roll['t'], tomato_ekom_roll['qw'])

tomato_ekom_pitch_resampled_x = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_ekom_pitch['t'], tomato_ekom_pitch['x'])
tomato_ekom_pitch_resampled_y = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_ekom_pitch['t'], tomato_ekom_pitch['y'])
tomato_ekom_pitch_resampled_z = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_ekom_pitch['t'], tomato_ekom_pitch['z'])
tomato_ekom_pitch_resampled_qx = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_ekom_pitch['t'], tomato_ekom_pitch['qx'])
tomato_ekom_pitch_resampled_qy = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_ekom_pitch['t'], tomato_ekom_pitch['qy'])
tomato_ekom_pitch_resampled_qz = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_ekom_pitch['t'], tomato_ekom_pitch['qz'])
tomato_ekom_pitch_resampled_qw = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_ekom_pitch['t'], tomato_ekom_pitch['qw'])

tomato_ekom_yaw_resampled_x = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_ekom_yaw['t'], tomato_ekom_yaw['x'])
tomato_ekom_yaw_resampled_y = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_ekom_yaw['t'], tomato_ekom_yaw['y'])
tomato_ekom_yaw_resampled_z = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_ekom_yaw['t'], tomato_ekom_yaw['z'])
tomato_ekom_yaw_resampled_qx = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_ekom_yaw['t'], tomato_ekom_yaw['qx'])
tomato_ekom_yaw_resampled_qy = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_ekom_yaw['t'], tomato_ekom_yaw['qy'])
tomato_ekom_yaw_resampled_qz = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_ekom_yaw['t'], tomato_ekom_yaw['qz'])
tomato_ekom_yaw_resampled_qw = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_ekom_yaw['t'], tomato_ekom_yaw['qw'])

tomato_ekom_trans_x_alpha,tomato_ekom_trans_x_beta,tomato_ekom_trans_x_gamma = quaternion_to_euler_angle(tomato_ekom_trans_x['qw'], tomato_ekom_trans_x['qx'], tomato_ekom_trans_x['qy'], tomato_ekom_trans_x['qz'])
tomato_gt_trans_x_alpha,tomato_gt_trans_x_beta,tomato_gt_trans_x_gamma = quaternion_to_euler_angle(tomato_gt_trans_x['qw'], tomato_gt_trans_x['qx'], tomato_gt_trans_x['qy'], tomato_gt_trans_x['qz'])

tomato_ekom_trans_x_alpha_cleaned = cleanEuler(tomato_ekom_trans_x_alpha,0)
tomato_ekom_trans_x_beta_cleaned = cleanEuler(tomato_ekom_trans_x_beta,1)
tomato_ekom_trans_x_gamma_cleaned = cleanEuler(tomato_ekom_trans_x_gamma,2)

tomato_gt_trans_x_alpha_cleaned = cleanEuler(tomato_gt_trans_x_alpha,0)
tomato_gt_trans_x_beta_cleaned = cleanEuler(tomato_gt_trans_x_beta,1)
tomato_gt_trans_x_gamma_cleaned = cleanEuler(tomato_gt_trans_x_gamma,1)

tomato_ekom_trans_y_alpha,tomato_ekom_trans_y_beta,tomato_ekom_trans_y_gamma = quaternion_to_euler_angle(tomato_ekom_trans_y['qw'], tomato_ekom_trans_y['qx'], tomato_ekom_trans_y['qy'], tomato_ekom_trans_y['qz'])
tomato_gt_trans_y_alpha,tomato_gt_trans_y_beta,tomato_gt_trans_y_gamma = quaternion_to_euler_angle(tomato_gt_trans_y['qw'], tomato_gt_trans_y['qx'], tomato_gt_trans_y['qy'], tomato_gt_trans_y['qz'])

tomato_ekom_trans_y_alpha_cleaned = cleanEuler(tomato_ekom_trans_y_alpha,0)
tomato_ekom_trans_y_beta_cleaned = cleanEuler(tomato_ekom_trans_y_beta,1)
tomato_ekom_trans_y_gamma_cleaned = cleanEuler(tomato_ekom_trans_y_gamma,2)

tomato_gt_trans_y_alpha_cleaned = cleanEuler(tomato_gt_trans_y_alpha,0)
tomato_gt_trans_y_beta_cleaned = cleanEuler(tomato_gt_trans_y_beta,1)
tomato_gt_trans_y_gamma_cleaned = cleanEuler(tomato_gt_trans_y_gamma,2)

tomato_ekom_trans_z_alpha,tomato_ekom_trans_z_beta,tomato_ekom_trans_z_gamma = quaternion_to_euler_angle(tomato_ekom_trans_z['qw'], tomato_ekom_trans_z['qx'], tomato_ekom_trans_z['qy'], tomato_ekom_trans_z['qz'])
tomato_gt_trans_z_alpha,tomato_gt_trans_z_beta,tomato_gt_trans_z_gamma = quaternion_to_euler_angle(tomato_gt_trans_z['qw'], tomato_gt_trans_z['qx'], tomato_gt_trans_z['qy'], tomato_gt_trans_z['qz'])

tomato_ekom_trans_z_alpha_cleaned = cleanEuler(tomato_ekom_trans_z_alpha,0)
tomato_ekom_trans_z_beta_cleaned = cleanEuler(tomato_ekom_trans_z_beta,1)
tomato_ekom_trans_z_gamma_cleaned = cleanEuler(tomato_ekom_trans_z_gamma,2)

tomato_gt_trans_z_alpha_cleaned = cleanEuler(tomato_gt_trans_z_alpha,0)
tomato_gt_trans_z_beta_cleaned = cleanEuler(tomato_gt_trans_z_beta,1)
tomato_gt_trans_z_gamma_cleaned = cleanEuler(tomato_gt_trans_z_gamma,2)

tomato_ekom_roll_alpha,tomato_ekom_roll_beta,tomato_ekom_roll_gamma = quaternion_to_euler_angle(tomato_ekom_roll['qw'], tomato_ekom_roll['qx'], tomato_ekom_roll['qy'], tomato_ekom_roll['qz'])
tomato_gt_roll_alpha,tomato_gt_roll_beta,tomato_gt_roll_gamma = quaternion_to_euler_angle(tomato_gt_roll['qw'], tomato_gt_roll['qx'], tomato_gt_roll['qy'], tomato_gt_roll['qz'])

tomato_ekom_roll_alpha_cleaned = cleanEuler(tomato_ekom_roll_alpha,0)
tomato_ekom_roll_beta_cleaned = cleanEuler(tomato_ekom_roll_beta,1)
tomato_ekom_roll_gamma_cleaned = cleanEuler(tomato_ekom_roll_gamma,2)

tomato_gt_roll_alpha_cleaned = cleanEuler(tomato_gt_roll_alpha,0)
tomato_gt_roll_beta_cleaned = cleanEuler(tomato_gt_roll_beta,1)
tomato_gt_roll_gamma_cleaned = cleanEuler(tomato_gt_roll_gamma,2)

tomato_ekom_pitch_alpha,tomato_ekom_pitch_beta,tomato_ekom_pitch_gamma = quaternion_to_euler_angle(tomato_ekom_pitch['qw'], tomato_ekom_pitch['qx'], tomato_ekom_pitch['qy'], tomato_ekom_pitch['qz'])
tomato_gt_pitch_alpha,tomato_gt_pitch_beta,tomato_gt_pitch_gamma = quaternion_to_euler_angle(tomato_gt_pitch['qw'], tomato_gt_pitch['qx'], tomato_gt_pitch['qy'], tomato_gt_pitch['qz'])

tomato_ekom_pitch_alpha_cleaned = cleanEuler(tomato_ekom_pitch_alpha,0)
tomato_ekom_pitch_beta_cleaned = cleanEuler(tomato_ekom_pitch_beta,1)
tomato_ekom_pitch_gamma_cleaned = cleanEuler(tomato_ekom_pitch_gamma,2)

tomato_gt_pitch_alpha_cleaned = cleanEuler(tomato_gt_pitch_alpha,0)
tomato_gt_pitch_beta_cleaned = cleanEuler(tomato_gt_pitch_beta,1)
tomato_gt_pitch_gamma_cleaned = cleanEuler(tomato_gt_pitch_gamma,2)

tomato_ekom_yaw_alpha,tomato_ekom_yaw_beta,tomato_ekom_yaw_gamma = quaternion_to_euler_angle(tomato_ekom_yaw['qw'], tomato_ekom_yaw['qx'], tomato_ekom_yaw['qy'], tomato_ekom_yaw['qz'])
tomato_gt_yaw_alpha,tomato_gt_yaw_beta,tomato_gt_yaw_gamma = quaternion_to_euler_angle(tomato_gt_yaw['qw'], tomato_gt_yaw['qx'], tomato_gt_yaw['qy'], tomato_gt_yaw['qz'])

tomato_ekom_yaw_alpha_cleaned = cleanEuler(tomato_ekom_yaw_alpha,0)
tomato_ekom_yaw_beta_cleaned = cleanEuler(tomato_ekom_yaw_beta,1)
tomato_ekom_yaw_gamma_cleaned = cleanEuler(tomato_ekom_yaw_gamma,2)

tomato_gt_yaw_alpha_cleaned = cleanEuler(tomato_gt_yaw_alpha,0)
tomato_gt_yaw_beta_cleaned = cleanEuler(tomato_gt_yaw_beta,1)
tomato_gt_yaw_gamma_cleaned = cleanEuler(tomato_gt_yaw_gamma,2)

fig_summary, axs = plt.subplots(4,3)
fig_summary.set_size_inches(18, 12)
axs[0,0].plot(tomato_ekom_trans_x['t'], tomato_ekom_trans_x['x'], color=color_x, label='x')
axs[0,0].plot(tomato_ekom_trans_x['t'], tomato_ekom_trans_x['y'], color=color_y, label='y')
axs[0,0].plot(tomato_ekom_trans_x['t'], tomato_ekom_trans_x['z'], color=color_z, label='z')
axs[0,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x['x'], color=color_x, ls='--')
axs[0,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x['y'], color=color_y, ls='--')
axs[0,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x['z'], color=color_z, ls='--')
axs[0,1].plot(tomato_ekom_trans_y['t'], tomato_ekom_trans_y['x'], color=color_x, label='x')
axs[0,1].plot(tomato_ekom_trans_y['t'], tomato_ekom_trans_y['y'], color=color_y, label='y')
axs[0,1].plot(tomato_ekom_trans_y['t'], tomato_ekom_trans_y['z'], color=color_z, label='z')
axs[0,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y['x'], color=color_x, ls='--')
axs[0,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y['y'], color=color_y, ls='--')
axs[0,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y['z'], color=color_z, ls='--')
axs[0,2].plot(tomato_ekom_trans_z['t'], tomato_ekom_trans_z['x'], color=color_x, label='x')
axs[0,2].plot(tomato_ekom_trans_z['t'], tomato_ekom_trans_z['y'], color=color_y, label='y')
axs[0,2].plot(tomato_ekom_trans_z['t'], tomato_ekom_trans_z['z'], color=color_z, label='z')
axs[0,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z['x'], color=color_x, ls='--')
axs[0,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z['y'], color=color_y, ls='--')
axs[0,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z['z'], color=color_z, ls='--')
axs[2,0].plot(tomato_ekom_roll['t'], tomato_ekom_roll['x'], color=color_x, label='x')
axs[2,0].plot(tomato_ekom_roll['t'], tomato_ekom_roll['y'], color=color_y, label='y')
axs[2,0].plot(tomato_ekom_roll['t'], tomato_ekom_roll['z'], color=color_z, label='z')
axs[2,0].plot(tomato_gt_roll['t'], tomato_gt_roll['x'], color=color_x, ls='--')
axs[2,0].plot(tomato_gt_roll['t'], tomato_gt_roll['y'], color=color_y, ls='--')
axs[2,0].plot(tomato_gt_roll['t'], tomato_gt_roll['z'], color=color_z, ls='--')
axs[2,1].plot(tomato_ekom_pitch['t'], tomato_ekom_pitch['x'], color=color_x, label='x')
axs[2,1].plot(tomato_ekom_pitch['t'], tomato_ekom_pitch['y'], color=color_y, label='y')
axs[2,1].plot(tomato_ekom_pitch['t'], tomato_ekom_pitch['z'], color=color_z, label='z')
axs[2,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch['x'], color=color_x, ls='--')
axs[2,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch['y'], color=color_y, ls='--')
axs[2,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch['z'], color=color_z, ls='--')
axs[2,2].plot(tomato_ekom_yaw['t'], tomato_ekom_yaw['x'], color=color_x, label='x')
axs[2,2].plot(tomato_ekom_yaw['t'], tomato_ekom_yaw['y'], color=color_y, label='y')
axs[2,2].plot(tomato_ekom_yaw['t'], tomato_ekom_yaw['z'], color=color_z, label='z')
axs[2,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw['x'], color=color_x, ls='--')
axs[2,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw['y'], color=color_y, ls='--')
axs[2,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw['z'], color=color_z, ls='--')
axs[1,0].plot(tomato_ekom_trans_x['t'], tomato_ekom_trans_x_alpha_cleaned, color=color_x, label='qx')
axs[1,0].plot(tomato_ekom_trans_x['t'], tomato_ekom_trans_x_beta_cleaned, color=color_y, label='qy')
axs[1,0].plot(tomato_ekom_trans_x['t'], tomato_ekom_trans_x_gamma_cleaned, color=color_z, label='qz')
axs[1,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
axs[1,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x_beta_cleaned, color=color_y, ls = '--')
axs[1,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
axs[1,1].plot(tomato_ekom_trans_y['t'], tomato_ekom_trans_y_alpha_cleaned, color=color_x, label='qx')
axs[1,1].plot(tomato_ekom_trans_y['t'], tomato_ekom_trans_y_beta_cleaned, color=color_y, label='qy')
axs[1,1].plot(tomato_ekom_trans_y['t'], tomato_ekom_trans_y_gamma_cleaned, color=color_z, label='qz')
axs[1,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
axs[1,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y_beta_cleaned, color=color_y, ls = '--')
axs[1,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
axs[1,2].plot(tomato_ekom_trans_z['t'], tomato_ekom_trans_z_alpha_cleaned, color=color_x, label='qx')
axs[1,2].plot(tomato_ekom_trans_z['t'], tomato_ekom_trans_z_beta_cleaned, color=color_y, label='qy')
axs[1,2].plot(tomato_ekom_trans_z['t'], tomato_ekom_trans_z_gamma_cleaned, color=color_z, label='qz')
axs[1,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
axs[1,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z_beta_cleaned, color=color_y, ls = '--')
axs[1,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
axs[3,0].plot(tomato_ekom_roll['t'], tomato_ekom_roll_alpha_cleaned, color=color_x, label='qx')
axs[3,0].plot(tomato_ekom_roll['t'], tomato_ekom_roll_beta_cleaned, color=color_y, label='qy')
axs[3,0].plot(tomato_ekom_roll['t'], tomato_ekom_roll_gamma_cleaned, color=color_z, label='qz')
axs[3,0].plot(tomato_gt_roll['t'], tomato_gt_roll_alpha_cleaned, color=color_x, ls = '--')
axs[3,0].plot(tomato_gt_roll['t'], tomato_gt_roll_beta_cleaned, color=color_y, ls = '--')
axs[3,0].plot(tomato_gt_roll['t'], tomato_gt_roll_gamma_cleaned, color=color_z, ls = '--')
axs[3,1].plot(tomato_ekom_pitch['t'], tomato_ekom_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - tomato_ekom")
axs[3,1].plot(tomato_ekom_pitch['t'], tomato_ekom_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - tomato_ekom")
axs[3,1].plot(tomato_ekom_pitch['t'], tomato_ekom_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - tomato_ekom")
axs[3,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - tomato_gt")
axs[3,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - tomato_gt")
axs[3,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - tomato_gt")
axs[3,2].plot(tomato_ekom_yaw['t'], tomato_ekom_yaw_alpha_cleaned, color=color_x, label='x or roll')
axs[3,2].plot(tomato_ekom_yaw['t'], tomato_ekom_yaw_beta_cleaned, color=color_y, label='y or pitch')
axs[3,2].plot(tomato_ekom_yaw['t'], tomato_ekom_yaw_gamma_cleaned, color=color_z, label='z or yaw')
axs[3,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw_alpha_cleaned, color=color_x, ls = '--')
axs[3,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw_beta_cleaned, color=color_y, ls = '--')
axs[3,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw_gamma_cleaned, color=color_z, ls = '--')
for i in range(0, 4):
    for j in range(0, 3):
        axs[i,j].set_xlim([-2, 50])
axs[0,0].set_ylim(-0.3,  1)
axs[0,1].set_ylim(-0.3,  1)
axs[0,2].set_ylim(-0.3,  1)
axs[2,0].set_ylim(-0.1,  0.9)
axs[2,1].set_ylim(-0.1,  0.9)
axs[2,2].set_ylim(-0.1,  0.9)
axs[1,0].set_ylim(-120,  200)
axs[1,1].set_ylim(-120,  200)
axs[1,2].set_ylim(-120,  200)
axs[3,0].set_ylim(-200,  300)
axs[3,1].set_ylim(-200,  300)
axs[3,2].set_ylim(-200,  300)
axs[0,0].set_xticks([])
axs[1,0].set_xticks([])
axs[2,0].set_xticks([])
axs[0,1].set_xticks([])
axs[1,1].set_xticks([])
axs[2,1].set_xticks([])
axs[0,2].set_xticks([])
axs[1,2].set_xticks([])
axs[2,2].set_xticks([])
axs[0,0].set_xticklabels([])
axs[1,0].set_xticklabels([])
axs[2,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[1,1].set_xticklabels([])
axs[2,1].set_xticklabels([])
axs[0,2].set_xticklabels([])
axs[1,2].set_xticklabels([])
axs[2,2].set_xticklabels([])
axs[0,1].set_yticklabels([])
axs[0,2].set_yticklabels([])
axs[1,1].set_yticklabels([])
axs[1,2].set_yticklabels([])
axs[2,1].set_yticklabels([])
axs[2,2].set_yticklabels([])
axs[3,1].set_yticklabels([])
axs[3,2].set_yticklabels([])
axs[0,0].set(ylabel='Position [m]')
axs[1,0].set(ylabel='Euler angles [deg]')
axs[2,0].set(ylabel='Position [m]')
axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
axs[3,1].set(xlabel='Time [s]')
axs[3,2].set(xlabel='Time [s]')
axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=3)
fig_summary.align_ylabels(axs[:, 0])
fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
plt.show()

tomato_ekom_error_trans_x = computeEuclideanDistance(tomato_gt_trans_x['x'], tomato_gt_trans_x['y'], tomato_gt_trans_x['z'], tomato_ekom_trans_x_resampled_x, tomato_ekom_trans_x_resampled_y, tomato_ekom_trans_x_resampled_z)
tomato_ekom_error_trans_y = computeEuclideanDistance(tomato_gt_trans_y['x'], tomato_gt_trans_y['y'], tomato_gt_trans_y['z'], tomato_ekom_trans_y_resampled_x, tomato_ekom_trans_y_resampled_y, tomato_ekom_trans_y_resampled_z)
tomato_ekom_error_trans_z = computeEuclideanDistance(tomato_gt_trans_z['x'], tomato_gt_trans_z['y'], tomato_gt_trans_z['z'], tomato_ekom_trans_z_resampled_x, tomato_ekom_trans_z_resampled_y, tomato_ekom_trans_z_resampled_z)
tomato_ekom_error_roll = computeEuclideanDistance(tomato_gt_roll['x'], tomato_gt_roll['y'], tomato_gt_roll['z'], tomato_ekom_roll_resampled_x, tomato_ekom_roll_resampled_y, tomato_ekom_roll_resampled_z)
tomato_ekom_error_pitch = computeEuclideanDistance(tomato_gt_pitch['x'], tomato_gt_pitch['y'], tomato_gt_pitch['z'], tomato_ekom_pitch_resampled_x, tomato_ekom_pitch_resampled_y, tomato_ekom_pitch_resampled_z)
tomato_ekom_error_yaw = computeEuclideanDistance(tomato_gt_yaw['x'], tomato_gt_yaw['y'], tomato_gt_yaw['z'], tomato_ekom_yaw_resampled_x, tomato_ekom_yaw_resampled_y, tomato_ekom_yaw_resampled_z)

tomato_ekom_q_angle_trans_x = computeQuaternionError(tomato_ekom_trans_x_resampled_qx, tomato_ekom_trans_x_resampled_qy, tomato_ekom_trans_x_resampled_qz, tomato_ekom_trans_x_resampled_qw, tomato_gt_trans_x['qx'], tomato_gt_trans_x['qy'], tomato_gt_trans_x['qz'], tomato_gt_trans_x['qw'])
tomato_ekom_q_angle_trans_y = computeQuaternionError(tomato_ekom_trans_y_resampled_qx, tomato_ekom_trans_y_resampled_qy, tomato_ekom_trans_y_resampled_qz, tomato_ekom_trans_y_resampled_qw, tomato_gt_trans_y['qx'], tomato_gt_trans_y['qy'], tomato_gt_trans_y['qz'], tomato_gt_trans_y['qw'])
tomato_ekom_q_angle_trans_z = computeQuaternionError(tomato_ekom_trans_z_resampled_qx, tomato_ekom_trans_z_resampled_qy, tomato_ekom_trans_z_resampled_qz, tomato_ekom_trans_z_resampled_qw, tomato_gt_trans_z['qx'], tomato_gt_trans_z['qy'], tomato_gt_trans_z['qz'], tomato_gt_trans_z['qw'])
tomato_ekom_q_angle_roll = computeQuaternionError(tomato_ekom_roll_resampled_qx, tomato_ekom_roll_resampled_qy, tomato_ekom_roll_resampled_qz, tomato_ekom_roll_resampled_qw, tomato_gt_roll['qx'], tomato_gt_roll['qy'], tomato_gt_roll['qz'], tomato_gt_roll['qw'])
tomato_ekom_q_angle_pitch = computeQuaternionError(tomato_ekom_pitch_resampled_qx, tomato_ekom_pitch_resampled_qy, tomato_ekom_pitch_resampled_qz, tomato_ekom_pitch_resampled_qw, tomato_gt_pitch['qx'], tomato_gt_pitch['qy'], tomato_gt_pitch['qz'], tomato_gt_pitch['qw'])
tomato_ekom_q_angle_yaw = computeQuaternionError(tomato_ekom_yaw_resampled_qx, tomato_ekom_yaw_resampled_qy, tomato_ekom_yaw_resampled_qz, tomato_ekom_yaw_resampled_qw, tomato_gt_yaw['qx'], tomato_gt_yaw['qy'], tomato_gt_yaw['qz'], tomato_gt_yaw['qw'])

tomato_ekom_position_errors = np.concatenate((tomato_ekom_error_trans_x, tomato_ekom_error_trans_y, tomato_ekom_error_trans_z, tomato_ekom_error_roll, tomato_ekom_error_pitch, tomato_ekom_error_yaw))
tomato_ekom_rotation_errors = np.concatenate((tomato_ekom_q_angle_trans_x, tomato_ekom_q_angle_trans_y, tomato_ekom_q_angle_trans_z, tomato_ekom_q_angle_roll, tomato_ekom_q_angle_pitch, tomato_ekom_q_angle_yaw))

# ---------------------------------------------------------------------------  potted  ---------------------------------------------------------------------------------------

filePath_dataset = '/data/potted/'
potted_gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, '010_potted_translation_x_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, '010_potted_translation_y_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, '010_potted_translation_z_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_gt_roll = np.genfromtxt(os.path.join(filePath_dataset, '010_potted_roll_2_rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, '010_potted_pitch_2_rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, '010_potted_yaw_2_rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

potted_gt_trans_x['t'] = (potted_gt_trans_x['t']-potted_gt_trans_x['t'][0])*10
potted_gt_trans_x['x'] = potted_gt_trans_x['x']*0.01
potted_gt_trans_x['y'] = potted_gt_trans_x['y']*0.01
potted_gt_trans_x['z'] = potted_gt_trans_x['z']*0.01

potted_gt_trans_y['t'] = (potted_gt_trans_y['t']-potted_gt_trans_y['t'][0])*10
potted_gt_trans_y['x'] = potted_gt_trans_y['x']*0.01
potted_gt_trans_y['y'] = potted_gt_trans_y['y']*0.01
potted_gt_trans_y['z'] = potted_gt_trans_y['z']*0.01

potted_gt_trans_z['t'] = (potted_gt_trans_z['t']-potted_gt_trans_z['t'][0])*10
potted_gt_trans_z['x'] = potted_gt_trans_z['x']*0.01
potted_gt_trans_z['y'] = potted_gt_trans_z['y']*0.01
potted_gt_trans_z['z'] = potted_gt_trans_z['z']*0.01

potted_gt_roll['t'] = (potted_gt_roll['t']-potted_gt_roll['t'][0])*10
potted_gt_roll['x'] = potted_gt_roll['x']*0.01
potted_gt_roll['y'] = potted_gt_roll['y']*0.01
potted_gt_roll['z'] = potted_gt_roll['z']*0.01

potted_gt_pitch['t'] = (potted_gt_pitch['t']-potted_gt_pitch['t'][0])*10
potted_gt_pitch['x'] = potted_gt_pitch['x']*0.01
potted_gt_pitch['y'] = potted_gt_pitch['y']*0.01
potted_gt_pitch['z'] = potted_gt_pitch['z']*0.01

potted_gt_yaw['t'] = (potted_gt_yaw['t']-potted_gt_yaw['t'][0])*10
potted_gt_yaw['x'] = potted_gt_yaw['x']*0.01
potted_gt_yaw['y'] = potted_gt_yaw['y']*0.01
potted_gt_yaw['z'] = potted_gt_yaw['z']*0.01

potted_ekom_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_ekom_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/new/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_ekom_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_ekom_roll = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_ekom_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/new/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_ekom_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/new/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

potted_ekom_trans_x_resampled_x = resampling_by_interpolate(potted_gt_trans_x['t'], potted_ekom_trans_x['t'], potted_ekom_trans_x['x'])
potted_ekom_trans_x_resampled_y = resampling_by_interpolate(potted_gt_trans_x['t'], potted_ekom_trans_x['t'], potted_ekom_trans_x['y'])
potted_ekom_trans_x_resampled_z = resampling_by_interpolate(potted_gt_trans_x['t'], potted_ekom_trans_x['t'], potted_ekom_trans_x['z'])
potted_ekom_trans_x_resampled_qx = resampling_by_interpolate(potted_gt_trans_x['t'], potted_ekom_trans_x['t'], potted_ekom_trans_x['qx'])
potted_ekom_trans_x_resampled_qy = resampling_by_interpolate(potted_gt_trans_x['t'], potted_ekom_trans_x['t'], potted_ekom_trans_x['qy'])
potted_ekom_trans_x_resampled_qz = resampling_by_interpolate(potted_gt_trans_x['t'], potted_ekom_trans_x['t'], potted_ekom_trans_x['qz'])
potted_ekom_trans_x_resampled_qw = resampling_by_interpolate(potted_gt_trans_x['t'], potted_ekom_trans_x['t'], potted_ekom_trans_x['qw'])

potted_ekom_trans_y_resampled_x = resampling_by_interpolate(potted_gt_trans_y['t'], potted_ekom_trans_y['t'], potted_ekom_trans_y['x'])
potted_ekom_trans_y_resampled_y = resampling_by_interpolate(potted_gt_trans_y['t'], potted_ekom_trans_y['t'], potted_ekom_trans_y['y'])
potted_ekom_trans_y_resampled_z = resampling_by_interpolate(potted_gt_trans_y['t'], potted_ekom_trans_y['t'], potted_ekom_trans_y['z'])
potted_ekom_trans_y_resampled_qx = resampling_by_interpolate(potted_gt_trans_y['t'], potted_ekom_trans_y['t'], potted_ekom_trans_y['qx'])
potted_ekom_trans_y_resampled_qy = resampling_by_interpolate(potted_gt_trans_y['t'], potted_ekom_trans_y['t'], potted_ekom_trans_y['qy'])
potted_ekom_trans_y_resampled_qz = resampling_by_interpolate(potted_gt_trans_y['t'], potted_ekom_trans_y['t'], potted_ekom_trans_y['qz'])
potted_ekom_trans_y_resampled_qw = resampling_by_interpolate(potted_gt_trans_y['t'], potted_ekom_trans_y['t'], potted_ekom_trans_y['qw'])

potted_ekom_trans_z_resampled_x = resampling_by_interpolate(potted_gt_trans_z['t'], potted_ekom_trans_z['t'], potted_ekom_trans_z['x'])
potted_ekom_trans_z_resampled_y = resampling_by_interpolate(potted_gt_trans_z['t'], potted_ekom_trans_z['t'], potted_ekom_trans_z['y'])
potted_ekom_trans_z_resampled_z = resampling_by_interpolate(potted_gt_trans_z['t'], potted_ekom_trans_z['t'], potted_ekom_trans_z['z'])
potted_ekom_trans_z_resampled_qx = resampling_by_interpolate(potted_gt_trans_z['t'], potted_ekom_trans_z['t'], potted_ekom_trans_z['qx'])
potted_ekom_trans_z_resampled_qy = resampling_by_interpolate(potted_gt_trans_z['t'], potted_ekom_trans_z['t'], potted_ekom_trans_z['qy'])
potted_ekom_trans_z_resampled_qz = resampling_by_interpolate(potted_gt_trans_z['t'], potted_ekom_trans_z['t'], potted_ekom_trans_z['qz'])
potted_ekom_trans_z_resampled_qw = resampling_by_interpolate(potted_gt_trans_z['t'], potted_ekom_trans_z['t'], potted_ekom_trans_z['qw'])

potted_ekom_roll_resampled_x = resampling_by_interpolate(potted_gt_roll['t'], potted_ekom_roll['t'], potted_ekom_roll['x'])
potted_ekom_roll_resampled_y = resampling_by_interpolate(potted_gt_roll['t'], potted_ekom_roll['t'], potted_ekom_roll['y'])
potted_ekom_roll_resampled_z = resampling_by_interpolate(potted_gt_roll['t'], potted_ekom_roll['t'], potted_ekom_roll['z'])
potted_ekom_roll_resampled_qx = resampling_by_interpolate(potted_gt_roll['t'], potted_ekom_roll['t'], potted_ekom_roll['qx'])
potted_ekom_roll_resampled_qy = resampling_by_interpolate(potted_gt_roll['t'], potted_ekom_roll['t'], potted_ekom_roll['qy'])
potted_ekom_roll_resampled_qz = resampling_by_interpolate(potted_gt_roll['t'], potted_ekom_roll['t'], potted_ekom_roll['qz'])
potted_ekom_roll_resampled_qw = resampling_by_interpolate(potted_gt_roll['t'], potted_ekom_roll['t'], potted_ekom_roll['qw'])

potted_ekom_pitch_resampled_x = resampling_by_interpolate(potted_gt_pitch['t'], potted_ekom_pitch['t'], potted_ekom_pitch['x'])
potted_ekom_pitch_resampled_y = resampling_by_interpolate(potted_gt_pitch['t'], potted_ekom_pitch['t'], potted_ekom_pitch['y'])
potted_ekom_pitch_resampled_z = resampling_by_interpolate(potted_gt_pitch['t'], potted_ekom_pitch['t'], potted_ekom_pitch['z'])
potted_ekom_pitch_resampled_qx = resampling_by_interpolate(potted_gt_pitch['t'], potted_ekom_pitch['t'], potted_ekom_pitch['qx'])
potted_ekom_pitch_resampled_qy = resampling_by_interpolate(potted_gt_pitch['t'], potted_ekom_pitch['t'], potted_ekom_pitch['qy'])
potted_ekom_pitch_resampled_qz = resampling_by_interpolate(potted_gt_pitch['t'], potted_ekom_pitch['t'], potted_ekom_pitch['qz'])
potted_ekom_pitch_resampled_qw = resampling_by_interpolate(potted_gt_pitch['t'], potted_ekom_pitch['t'], potted_ekom_pitch['qw'])

potted_ekom_yaw_resampled_x = resampling_by_interpolate(potted_gt_yaw['t'], potted_ekom_yaw['t'], potted_ekom_yaw['x'])
potted_ekom_yaw_resampled_y = resampling_by_interpolate(potted_gt_yaw['t'], potted_ekom_yaw['t'], potted_ekom_yaw['y'])
potted_ekom_yaw_resampled_z = resampling_by_interpolate(potted_gt_yaw['t'], potted_ekom_yaw['t'], potted_ekom_yaw['z'])
potted_ekom_yaw_resampled_qx = resampling_by_interpolate(potted_gt_yaw['t'], potted_ekom_yaw['t'], potted_ekom_yaw['qx'])
potted_ekom_yaw_resampled_qy = resampling_by_interpolate(potted_gt_yaw['t'], potted_ekom_yaw['t'], potted_ekom_yaw['qy'])
potted_ekom_yaw_resampled_qz = resampling_by_interpolate(potted_gt_yaw['t'], potted_ekom_yaw['t'], potted_ekom_yaw['qz'])
potted_ekom_yaw_resampled_qw = resampling_by_interpolate(potted_gt_yaw['t'], potted_ekom_yaw['t'], potted_ekom_yaw['qw'])

potted_ekom_trans_x_alpha,potted_ekom_trans_x_beta,potted_ekom_trans_x_gamma = quaternion_to_euler_angle(potted_ekom_trans_x['qw'], potted_ekom_trans_x['qx'], potted_ekom_trans_x['qy'], potted_ekom_trans_x['qz'])
potted_gt_trans_x_alpha,potted_gt_trans_x_beta,potted_gt_trans_x_gamma = quaternion_to_euler_angle(potted_gt_trans_x['qw'], potted_gt_trans_x['qx'], potted_gt_trans_x['qy'], potted_gt_trans_x['qz'])

potted_ekom_trans_x_alpha_cleaned = cleanEuler(potted_ekom_trans_x_alpha,0)
potted_ekom_trans_x_beta_cleaned = cleanEuler(potted_ekom_trans_x_beta,1)
potted_ekom_trans_x_gamma_cleaned = cleanEuler(potted_ekom_trans_x_gamma,2)

potted_gt_trans_x_alpha_cleaned = cleanEuler(potted_gt_trans_x_alpha,0)
potted_gt_trans_x_beta_cleaned = cleanEuler(potted_gt_trans_x_beta,1)
potted_gt_trans_x_gamma_cleaned = cleanEuler(potted_gt_trans_x_gamma,1)

potted_ekom_trans_y_alpha,potted_ekom_trans_y_beta,potted_ekom_trans_y_gamma = quaternion_to_euler_angle(potted_ekom_trans_y['qw'], potted_ekom_trans_y['qx'], potted_ekom_trans_y['qy'], potted_ekom_trans_y['qz'])
potted_gt_trans_y_alpha,potted_gt_trans_y_beta,potted_gt_trans_y_gamma = quaternion_to_euler_angle(potted_gt_trans_y['qw'], potted_gt_trans_y['qx'], potted_gt_trans_y['qy'], potted_gt_trans_y['qz'])

potted_ekom_trans_y_alpha_cleaned = cleanEuler(potted_ekom_trans_y_alpha,0)
potted_ekom_trans_y_beta_cleaned = cleanEuler(potted_ekom_trans_y_beta,1)
potted_ekom_trans_y_gamma_cleaned = cleanEuler(potted_ekom_trans_y_gamma,2)

potted_gt_trans_y_alpha_cleaned = cleanEuler(potted_gt_trans_y_alpha,0)
potted_gt_trans_y_beta_cleaned = cleanEuler(potted_gt_trans_y_beta,1)
potted_gt_trans_y_gamma_cleaned = cleanEuler(potted_gt_trans_y_gamma,2)

potted_ekom_trans_z_alpha,potted_ekom_trans_z_beta,potted_ekom_trans_z_gamma = quaternion_to_euler_angle(potted_ekom_trans_z['qw'], potted_ekom_trans_z['qx'], potted_ekom_trans_z['qy'], potted_ekom_trans_z['qz'])
potted_gt_trans_z_alpha,potted_gt_trans_z_beta,potted_gt_trans_z_gamma = quaternion_to_euler_angle(potted_gt_trans_z['qw'], potted_gt_trans_z['qx'], potted_gt_trans_z['qy'], potted_gt_trans_z['qz'])

potted_ekom_trans_z_alpha_cleaned = cleanEuler(potted_ekom_trans_z_alpha,0)
potted_ekom_trans_z_beta_cleaned = cleanEuler(potted_ekom_trans_z_beta,1)
potted_ekom_trans_z_gamma_cleaned = cleanEuler(potted_ekom_trans_z_gamma,2)

potted_gt_trans_z_alpha_cleaned = cleanEuler(potted_gt_trans_z_alpha,0)
potted_gt_trans_z_beta_cleaned = cleanEuler(potted_gt_trans_z_beta,1)
potted_gt_trans_z_gamma_cleaned = cleanEuler(potted_gt_trans_z_gamma,2)

potted_ekom_roll_alpha,potted_ekom_roll_beta,potted_ekom_roll_gamma = quaternion_to_euler_angle(potted_ekom_roll['qw'], potted_ekom_roll['qx'], potted_ekom_roll['qy'], potted_ekom_roll['qz'])
potted_gt_roll_alpha,potted_gt_roll_beta,potted_gt_roll_gamma = quaternion_to_euler_angle(potted_gt_roll['qw'], potted_gt_roll['qx'], potted_gt_roll['qy'], potted_gt_roll['qz'])

potted_ekom_roll_alpha_cleaned = cleanEuler(potted_ekom_roll_alpha,0)
potted_ekom_roll_beta_cleaned = cleanEuler(potted_ekom_roll_beta,1)
potted_ekom_roll_gamma_cleaned = cleanEuler(potted_ekom_roll_gamma,2)

potted_gt_roll_alpha_cleaned = cleanEuler(potted_gt_roll_alpha,0)
potted_gt_roll_beta_cleaned = cleanEuler(potted_gt_roll_beta,1)
potted_gt_roll_gamma_cleaned = cleanEuler(potted_gt_roll_gamma,2)

potted_ekom_pitch_alpha,potted_ekom_pitch_beta,potted_ekom_pitch_gamma = quaternion_to_euler_angle(potted_ekom_pitch['qw'], potted_ekom_pitch['qx'], potted_ekom_pitch['qy'], potted_ekom_pitch['qz'])
potted_gt_pitch_alpha,potted_gt_pitch_beta,potted_gt_pitch_gamma = quaternion_to_euler_angle(potted_gt_pitch['qw'], potted_gt_pitch['qx'], potted_gt_pitch['qy'], potted_gt_pitch['qz'])

potted_ekom_pitch_alpha_cleaned = cleanEuler(potted_ekom_pitch_alpha,0)
potted_ekom_pitch_beta_cleaned = cleanEuler(potted_ekom_pitch_beta,1)
potted_ekom_pitch_gamma_cleaned = cleanEuler(potted_ekom_pitch_gamma,2)

potted_gt_pitch_alpha_cleaned = cleanEuler(potted_gt_pitch_alpha,0)
potted_gt_pitch_beta_cleaned = cleanEuler(potted_gt_pitch_beta,1)
potted_gt_pitch_gamma_cleaned = cleanEuler(potted_gt_pitch_gamma,2)

potted_ekom_yaw_alpha,potted_ekom_yaw_beta,potted_ekom_yaw_gamma = quaternion_to_euler_angle(potted_ekom_yaw['qw'], potted_ekom_yaw['qx'], potted_ekom_yaw['qy'], potted_ekom_yaw['qz'])
potted_gt_yaw_alpha,potted_gt_yaw_beta,potted_gt_yaw_gamma = quaternion_to_euler_angle(potted_gt_yaw['qw'], potted_gt_yaw['qx'], potted_gt_yaw['qy'], potted_gt_yaw['qz'])

potted_ekom_yaw_alpha_cleaned = cleanEuler(potted_ekom_yaw_alpha,0)
potted_ekom_yaw_beta_cleaned = cleanEuler(potted_ekom_yaw_beta,1)
potted_ekom_yaw_gamma_cleaned = cleanEuler(potted_ekom_yaw_gamma,2)

potted_gt_yaw_alpha_cleaned = cleanEuler(potted_gt_yaw_alpha,0)
potted_gt_yaw_beta_cleaned = cleanEuler(potted_gt_yaw_beta,1)
potted_gt_yaw_gamma_cleaned = cleanEuler(potted_gt_yaw_gamma,2)

fig_summary, axs = plt.subplots(4,3)
fig_summary.set_size_inches(18, 12)
axs[0,0].plot(potted_ekom_trans_x['t'], potted_ekom_trans_x['x'], color=color_x, label='x')
axs[0,0].plot(potted_ekom_trans_x['t'], potted_ekom_trans_x['y'], color=color_y, label='y')
axs[0,0].plot(potted_ekom_trans_x['t'], potted_ekom_trans_x['z'], color=color_z, label='z')
axs[0,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x['x'], color=color_x, ls='--')
axs[0,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x['y'], color=color_y, ls='--')
axs[0,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x['z'], color=color_z, ls='--')
axs[0,1].plot(potted_ekom_trans_y['t'], potted_ekom_trans_y['x'], color=color_x, label='x')
axs[0,1].plot(potted_ekom_trans_y['t'], potted_ekom_trans_y['y'], color=color_y, label='y')
axs[0,1].plot(potted_ekom_trans_y['t'], potted_ekom_trans_y['z'], color=color_z, label='z')
axs[0,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y['x'], color=color_x, ls='--')
axs[0,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y['y'], color=color_y, ls='--')
axs[0,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y['z'], color=color_z, ls='--')
axs[0,2].plot(potted_ekom_trans_z['t'], potted_ekom_trans_z['x'], color=color_x, label='x')
axs[0,2].plot(potted_ekom_trans_z['t'], potted_ekom_trans_z['y'], color=color_y, label='y')
axs[0,2].plot(potted_ekom_trans_z['t'], potted_ekom_trans_z['z'], color=color_z, label='z')
axs[0,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z['x'], color=color_x, ls='--')
axs[0,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z['y'], color=color_y, ls='--')
axs[0,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z['z'], color=color_z, ls='--')
axs[2,0].plot(potted_ekom_roll['t'], potted_ekom_roll['x'], color=color_x, label='x')
axs[2,0].plot(potted_ekom_roll['t'], potted_ekom_roll['y'], color=color_y, label='y')
axs[2,0].plot(potted_ekom_roll['t'], potted_ekom_roll['z'], color=color_z, label='z')
axs[2,0].plot(potted_gt_roll['t'], potted_gt_roll['x'], color=color_x, ls='--')
axs[2,0].plot(potted_gt_roll['t'], potted_gt_roll['y'], color=color_y, ls='--')
axs[2,0].plot(potted_gt_roll['t'], potted_gt_roll['z'], color=color_z, ls='--')
axs[2,1].plot(potted_ekom_pitch['t'], potted_ekom_pitch['x'], color=color_x, label='x')
axs[2,1].plot(potted_ekom_pitch['t'], potted_ekom_pitch['y'], color=color_y, label='y')
axs[2,1].plot(potted_ekom_pitch['t'], potted_ekom_pitch['z'], color=color_z, label='z')
axs[2,1].plot(potted_gt_pitch['t'], potted_gt_pitch['x'], color=color_x, ls='--')
axs[2,1].plot(potted_gt_pitch['t'], potted_gt_pitch['y'], color=color_y, ls='--')
axs[2,1].plot(potted_gt_pitch['t'], potted_gt_pitch['z'], color=color_z, ls='--')
axs[2,2].plot(potted_ekom_yaw['t'], potted_ekom_yaw['x'], color=color_x, label='x')
axs[2,2].plot(potted_ekom_yaw['t'], potted_ekom_yaw['y'], color=color_y, label='y')
axs[2,2].plot(potted_ekom_yaw['t'], potted_ekom_yaw['z'], color=color_z, label='z')
axs[2,2].plot(potted_gt_yaw['t'], potted_gt_yaw['x'], color=color_x, ls='--')
axs[2,2].plot(potted_gt_yaw['t'], potted_gt_yaw['y'], color=color_y, ls='--')
axs[2,2].plot(potted_gt_yaw['t'], potted_gt_yaw['z'], color=color_z, ls='--')
axs[1,0].plot(potted_ekom_trans_x['t'], potted_ekom_trans_x_alpha_cleaned, color=color_x, label='qx')
axs[1,0].plot(potted_ekom_trans_x['t'], potted_ekom_trans_x_beta_cleaned, color=color_y, label='qy')
axs[1,0].plot(potted_ekom_trans_x['t'], potted_ekom_trans_x_gamma_cleaned, color=color_z, label='qz')
axs[1,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
axs[1,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x_beta_cleaned, color=color_y, ls = '--')
axs[1,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
axs[1,1].plot(potted_ekom_trans_y['t'], potted_ekom_trans_y_alpha_cleaned, color=color_x, label='qx')
axs[1,1].plot(potted_ekom_trans_y['t'], potted_ekom_trans_y_beta_cleaned, color=color_y, label='qy')
axs[1,1].plot(potted_ekom_trans_y['t'], potted_ekom_trans_y_gamma_cleaned, color=color_z, label='qz')
axs[1,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
axs[1,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y_beta_cleaned, color=color_y, ls = '--')
axs[1,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
axs[1,2].plot(potted_ekom_trans_z['t'], potted_ekom_trans_z_alpha_cleaned, color=color_x, label='qx')
axs[1,2].plot(potted_ekom_trans_z['t'], potted_ekom_trans_z_beta_cleaned, color=color_y, label='qy')
axs[1,2].plot(potted_ekom_trans_z['t'], potted_ekom_trans_z_gamma_cleaned, color=color_z, label='qz')
axs[1,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
axs[1,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z_beta_cleaned, color=color_y, ls = '--')
axs[1,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
axs[3,0].plot(potted_ekom_roll['t'], potted_ekom_roll_alpha_cleaned, color=color_x, label='qx')
axs[3,0].plot(potted_ekom_roll['t'], potted_ekom_roll_beta_cleaned, color=color_y, label='qy')
axs[3,0].plot(potted_ekom_roll['t'], potted_ekom_roll_gamma_cleaned, color=color_z, label='qz')
axs[3,0].plot(potted_gt_roll['t'], potted_gt_roll_alpha_cleaned, color=color_x, ls = '--')
axs[3,0].plot(potted_gt_roll['t'], potted_gt_roll_beta_cleaned, color=color_y, ls = '--')
axs[3,0].plot(potted_gt_roll['t'], potted_gt_roll_gamma_cleaned, color=color_z, ls = '--')
axs[3,1].plot(potted_ekom_pitch['t'], potted_ekom_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - potted_ekom")
axs[3,1].plot(potted_ekom_pitch['t'], potted_ekom_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - potted_ekom")
axs[3,1].plot(potted_ekom_pitch['t'], potted_ekom_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - potted_ekom")
axs[3,1].plot(potted_gt_pitch['t'], potted_gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - potted_gt")
axs[3,1].plot(potted_gt_pitch['t'], potted_gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - potted_gt")
axs[3,1].plot(potted_gt_pitch['t'], potted_gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - potted_gt")
axs[3,2].plot(potted_ekom_yaw['t'], potted_ekom_yaw_alpha_cleaned, color=color_x, label='x or roll')
axs[3,2].plot(potted_ekom_yaw['t'], potted_ekom_yaw_beta_cleaned, color=color_y, label='y or pitch')
axs[3,2].plot(potted_ekom_yaw['t'], potted_ekom_yaw_gamma_cleaned, color=color_z, label='z or yaw')
axs[3,2].plot(potted_gt_yaw['t'], potted_gt_yaw_alpha_cleaned, color=color_x, ls = '--')
axs[3,2].plot(potted_gt_yaw['t'], potted_gt_yaw_beta_cleaned, color=color_y, ls = '--')
axs[3,2].plot(potted_gt_yaw['t'], potted_gt_yaw_gamma_cleaned, color=color_z, ls = '--')
for i in range(0, 4):
    for j in range(0, 3):
        axs[i,j].set_xlim([-2, 50])
axs[0,0].set_ylim(-0.3,  1)
axs[0,1].set_ylim(-0.3,  1)
axs[0,2].set_ylim(-0.3,  1)
axs[2,0].set_ylim(-0.1,  0.9)
axs[2,1].set_ylim(-0.1,  0.9)
axs[2,2].set_ylim(-0.1,  0.9)
axs[1,0].set_ylim(-120,  200)
axs[1,1].set_ylim(-120,  200)
axs[1,2].set_ylim(-120,  200)
axs[3,0].set_ylim(-200,  300)
axs[3,1].set_ylim(-200,  300)
axs[3,2].set_ylim(-200,  300)
axs[0,0].set_xticks([])
axs[1,0].set_xticks([])
axs[2,0].set_xticks([])
axs[0,1].set_xticks([])
axs[1,1].set_xticks([])
axs[2,1].set_xticks([])
axs[0,2].set_xticks([])
axs[1,2].set_xticks([])
axs[2,2].set_xticks([])
axs[0,0].set_xticklabels([])
axs[1,0].set_xticklabels([])
axs[2,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[1,1].set_xticklabels([])
axs[2,1].set_xticklabels([])
axs[0,2].set_xticklabels([])
axs[1,2].set_xticklabels([])
axs[2,2].set_xticklabels([])
axs[0,1].set_yticklabels([])
axs[0,2].set_yticklabels([])
axs[1,1].set_yticklabels([])
axs[1,2].set_yticklabels([])
axs[2,1].set_yticklabels([])
axs[2,2].set_yticklabels([])
axs[3,1].set_yticklabels([])
axs[3,2].set_yticklabels([])
axs[0,0].set(ylabel='Position [m]')
axs[1,0].set(ylabel='Euler angles [deg]')
axs[2,0].set(ylabel='Position [m]')
axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
axs[3,1].set(xlabel='Time [s]')
axs[3,2].set(xlabel='Time [s]')
axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=3)
fig_summary.align_ylabels(axs[:, 0])
fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
plt.show()

potted_ekom_error_trans_x = computeEuclideanDistance(potted_gt_trans_x['x'], potted_gt_trans_x['y'], potted_gt_trans_x['z'], potted_ekom_trans_x_resampled_x, potted_ekom_trans_x_resampled_y, potted_ekom_trans_x_resampled_z)
potted_ekom_error_trans_y = computeEuclideanDistance(potted_gt_trans_y['x'], potted_gt_trans_y['y'], potted_gt_trans_y['z'], potted_ekom_trans_y_resampled_x, potted_ekom_trans_y_resampled_y, potted_ekom_trans_y_resampled_z)
potted_ekom_error_trans_z = computeEuclideanDistance(potted_gt_trans_z['x'], potted_gt_trans_z['y'], potted_gt_trans_z['z'], potted_ekom_trans_z_resampled_x, potted_ekom_trans_z_resampled_y, potted_ekom_trans_z_resampled_z)
potted_ekom_error_roll = computeEuclideanDistance(potted_gt_roll['x'], potted_gt_roll['y'], potted_gt_roll['z'], potted_ekom_roll_resampled_x, potted_ekom_roll_resampled_y, potted_ekom_roll_resampled_z)
potted_ekom_error_pitch = computeEuclideanDistance(potted_gt_pitch['x'], potted_gt_pitch['y'], potted_gt_pitch['z'], potted_ekom_pitch_resampled_x, potted_ekom_pitch_resampled_y, potted_ekom_pitch_resampled_z)
potted_ekom_error_yaw = computeEuclideanDistance(potted_gt_yaw['x'], potted_gt_yaw['y'], potted_gt_yaw['z'], potted_ekom_yaw_resampled_x, potted_ekom_yaw_resampled_y, potted_ekom_yaw_resampled_z)

potted_ekom_q_angle_trans_x = computeQuaternionError(potted_ekom_trans_x_resampled_qx, potted_ekom_trans_x_resampled_qy, potted_ekom_trans_x_resampled_qz, potted_ekom_trans_x_resampled_qw, potted_gt_trans_x['qx'], potted_gt_trans_x['qy'], potted_gt_trans_x['qz'], potted_gt_trans_x['qw'])
potted_ekom_q_angle_trans_y = computeQuaternionError(potted_ekom_trans_y_resampled_qx, potted_ekom_trans_y_resampled_qy, potted_ekom_trans_y_resampled_qz, potted_ekom_trans_y_resampled_qw, potted_gt_trans_y['qx'], potted_gt_trans_y['qy'], potted_gt_trans_y['qz'], potted_gt_trans_y['qw'])
potted_ekom_q_angle_trans_z = computeQuaternionError(potted_ekom_trans_z_resampled_qx, potted_ekom_trans_z_resampled_qy, potted_ekom_trans_z_resampled_qz, potted_ekom_trans_z_resampled_qw, potted_gt_trans_z['qx'], potted_gt_trans_z['qy'], potted_gt_trans_z['qz'], potted_gt_trans_z['qw'])
potted_ekom_q_angle_roll = computeQuaternionError(potted_ekom_roll_resampled_qx, potted_ekom_roll_resampled_qy, potted_ekom_roll_resampled_qz, potted_ekom_roll_resampled_qw, potted_gt_roll['qx'], potted_gt_roll['qy'], potted_gt_roll['qz'], potted_gt_roll['qw'])
potted_ekom_q_angle_pitch = computeQuaternionError(potted_ekom_pitch_resampled_qx, potted_ekom_pitch_resampled_qy, potted_ekom_pitch_resampled_qz, potted_ekom_pitch_resampled_qw, potted_gt_pitch['qx'], potted_gt_pitch['qy'], potted_gt_pitch['qz'], potted_gt_pitch['qw'])
potted_ekom_q_angle_yaw = computeQuaternionError(potted_ekom_yaw_resampled_qx, potted_ekom_yaw_resampled_qy, potted_ekom_yaw_resampled_qz, potted_ekom_yaw_resampled_qw, potted_gt_yaw['qx'], potted_gt_yaw['qy'], potted_gt_yaw['qz'], potted_gt_yaw['qw'])

potted_ekom_position_errors = np.concatenate((potted_ekom_error_trans_x, potted_ekom_error_trans_y, potted_ekom_error_trans_z, potted_ekom_error_roll, potted_ekom_error_pitch, potted_ekom_error_yaw))
potted_ekom_rotation_errors = np.concatenate((potted_ekom_q_angle_trans_x, potted_ekom_q_angle_trans_y, potted_ekom_q_angle_trans_z, potted_ekom_q_angle_roll, potted_ekom_q_angle_pitch, potted_ekom_q_angle_yaw))


labels = ['dragon', 'jell-o', 'bottle', 'soup can', 'spam']
ticks=[0, 1, 2, 3, 4]
medianprops = dict(color='white')

rad_to_deg = 180/math.pi

all_objects_position_errors = [dragon_ekom_position_errors, gelatin_ekom_position_errors, mustard_ekom_position_errors, tomato_ekom_position_errors, potted_ekom_position_errors]
all_objects_rotation_errors = [dragon_ekom_rotation_errors*rad_to_deg, gelatin_ekom_rotation_errors*rad_to_deg, mustard_ekom_rotation_errors*rad_to_deg, tomato_ekom_rotation_errors*rad_to_deg, potted_ekom_rotation_errors*rad_to_deg]
# new_quart_array = np.array(quart_vec_pos).transpose

fig15, ax1 = plt.subplots(1,2)
ax1[0].set_xlabel('Position error [m]', color='k')
ax1[1].set_xlabel('Rotation error [deg]', color='k')
res1 = ax1[0].boxplot(all_objects_position_errors, labels=labels, vert=False, showfliers=False,
                    patch_artist=True, medianprops=medianprops)
res2 = ax1[1].boxplot(all_objects_rotation_errors, vert=False, showfliers=False,
                    patch_artist=True, medianprops=medianprops)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(res1[element], color='k')
    plt.setp(res2[element], color='k')
ax1[1].set_yticklabels([])
ax1[1].set_yticks([])
ax1[0].set_xlim(-0.01,  0.15)
ax1[1].set_xlim(-5,  65)
ax1[1].xaxis.set_major_locator(plt.MaxNLocator(4))
colors=[color_ekom, color_ekom, color_ekom, color_ekom, color_ekom]
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
fig15.subplots_adjust(wspace=0.1)
plt.show()