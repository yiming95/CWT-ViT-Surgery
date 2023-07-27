import numpy as np
import pandas as pd
import pywt
import scaleogram as scg 
import matplotlib.pyplot as plt
import random
import time
import os
import shutil

def generate_CWT_scaleogram(file_name):
    """
    Continuous wavelet transform (CWT)
    :param file_name: folder name, eg.train1.pkl for LOSO 1
    """
    dirpath = file_name.split('.')[0]
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(file_name.split('.')[0]) # extract name for the folder

    kinematic_var_list = ['Left_MTM_tool_tip_position_1', 'Left_MTM_tool_tip_position_2', 'Left_MTM_tool_tip_position_3', 
                        'Left_MTM_tool_tip_rotation_matrix_1', 'Left_MTM_tool_tip_rotation_matrix_2', 'Left_MTM_tool_tip_rotation_matrix_3',
                        'Left_MTM_tool_tip_rotation_matrix_4', 'Left_MTM_tool_tip_rotation_matrix_5', 'Left_MTM_tool_tip_rotation_matrix_6',
                        'Left_MTM_tool_tip_rotation_matrix_7', 'Left_MTM_tool_tip_rotation_matrix_8', 'Left_MTM_tool_tip_rotation_matrix_9',
                        'Left_MTM_tool_tip_linear_velocity_1', 'Left_MTM_tool_tip_linear_velocity_2', 'Left_MTM_tool_tip_linear_velocity_3',
                        'Left_MTM_tool_tip_rotational_velocity_1', 'Left_MTM_tool_tip_rotational_velocity_2', 'Left_MTM_tool_tip_rotational_velocity_3',
                        'Left_MTM_gripper_angle_velocity', 
                               
                        'Right_MTM_tool_tip_position_1', 'Right_MTM_tool_tip_position_2',
                        'Right_MTM_tool_tip_position_3', 'Right_MTM_tool_tip_rotation_matrix_1', 'Right_MTM_tool_tip_rotation_matrix_2',
                        'Right_MTM_tool_tip_rotation_matrix_3', 'Right_MTM_tool_tip_rotation_matrix_4', 'Right_MTM_tool_tip_rotation_matrix_5',
                        'Right_MTM_tool_tip_rotation_matrix_6', 'Right_MTM_tool_tip_rotation_matrix_7', 'Right_MTM_tool_tip_rotation_matrix_8',
                        'Right_MTM_tool_tip_rotation_matrix_9', 'Right_MTM_tool_tip_linear_velocity_1', 'Right_MTM_tool_tip_linear_velocity_2',
                        'Right_MTM_tool_tip_linear_velocity_3', 'Right_MTM_tool_tip_rotational_velocity_1', 'Right_MTM_tool_tip_rotational_velocity_2',
                        'Right_MTM_tool_tip_rotational_velocity_3', 'Right_MTM_gripper_angle_velocity', 
                               
                        'PSM1_tool_tip_position_1', 'PSM1_tool_tip_position_2','PSM1_tool_tip_position_3', 
                        'PSM1_tool_tip_rotation_matrix_1', 'PSM1_tool_tip_rotation_matrix_2',
                        'PSM1_tool_tip_rotation_matrix_3', 'PSM1_tool_tip_rotation_matrix_4', 'PSM1_tool_tip_rotation_matrix_5',
                        'PSM1_tool_tip_rotation_matrix_6', 'PSM1_tool_tip_rotation_matrix_7', 'PSM1_tool_tip_rotation_matrix_8',
                        'PSM1_tool_tip_rotation_matrix_9', 'PSM1_tool_tip_linear_velocity_1', 'PSM1_tool_tip_linear_velocity_2',
                        'PSM1_tool_tip_linear_velocity_3', 'PSM1_tool_tip_rotational_velocity_1', 'PSM1_tool_tip_rotational_velocity_2',
                        'PSM1_tool_tip_rotational_velocity_3', 'PSM1_gripper_angle_velocity', 
                               
                        'PSM2_tool_tip_position_1',
                        'PSM2_tool_tip_position_2', 'PSM2_tool_tip_position_3', 'PSM2_tool_tip_rotation_matrix_1',
                        'PSM2_tool_tip_rotation_matrix_2', 'PSM2_tool_tip_rotation_matrix_3', 'PSM2_tool_tip_rotation_matrix_4',
                        'PSM2_tool_tip_rotation_matrix_5', 'PSM2_tool_tip_rotation_matrix_6', 'PSM2_tool_tip_rotation_matrix_7',
                        'PSM2_tool_tip_rotation_matrix_8', 'PSM2_tool_tip_rotation_matrix_9', 'PSM2_tool_tip_linear_velocity_1',
                        'PSM2_tool_tip_linear_velocity_2', 'PSM2_tool_tip_linear_velocity_3', 'PSM2_tool_tip_rotational_velocity_1',
                        'PSM2_tool_tip_rotational_velocity_2', 'PSM2_tool_tip_rotational_velocity_3', 'PSM2_gripper_angle_velocity']  # 76 kinematic variable string list


    df = pd.read_pickle(file_name)

    os.chdir(os.getcwd() + '/' + file_name.split('.')[0])

    trail_list = df['Trail'].tolist()
    
    for trial_name in trail_list:
        #print(trial_name)
        df1 = df.loc[df['Trail'] == trial_name]
        df_trial = df1['kinematic_data'].iloc[0]

        df_trial = pd.DataFrame(df_trial)
        df_trial.rename(columns={0: 'Left_MTM_tool_tip_position_1', 1: 'Left_MTM_tool_tip_position_2', 2: 'Left_MTM_tool_tip_position_3', 
                              3: 'Left_MTM_tool_tip_rotation_matrix_1', 4: 'Left_MTM_tool_tip_rotation_matrix_2', 5: 'Left_MTM_tool_tip_rotation_matrix_3',
                              6: 'Left_MTM_tool_tip_rotation_matrix_4', 7: 'Left_MTM_tool_tip_rotation_matrix_5', 8: 'Left_MTM_tool_tip_rotation_matrix_6',
                              9: 'Left_MTM_tool_tip_rotation_matrix_7', 10: 'Left_MTM_tool_tip_rotation_matrix_8', 11: 'Left_MTM_tool_tip_rotation_matrix_9',
                              12: 'Left_MTM_tool_tip_linear_velocity_1', 13: 'Left_MTM_tool_tip_linear_velocity_2', 14: 'Left_MTM_tool_tip_linear_velocity_3',
                              15: 'Left_MTM_tool_tip_rotational_velocity_1', 16: 'Left_MTM_tool_tip_rotational_velocity_2', 17: 'Left_MTM_tool_tip_rotational_velocity_3',
                              18: 'Left_MTM_gripper_angle_velocity', 
                               
                              19: 'Right_MTM_tool_tip_position_1', 20: 'Right_MTM_tool_tip_position_2',
                              21: 'Right_MTM_tool_tip_position_3', 22: 'Right_MTM_tool_tip_rotation_matrix_1', 23: 'Right_MTM_tool_tip_rotation_matrix_2',
                              24: 'Right_MTM_tool_tip_rotation_matrix_3', 25: 'Right_MTM_tool_tip_rotation_matrix_4', 26: 'Right_MTM_tool_tip_rotation_matrix_5',
                              27: 'Right_MTM_tool_tip_rotation_matrix_6', 28: 'Right_MTM_tool_tip_rotation_matrix_7', 29: 'Right_MTM_tool_tip_rotation_matrix_8',
                              30: 'Right_MTM_tool_tip_rotation_matrix_9', 31: 'Right_MTM_tool_tip_linear_velocity_1', 32: 'Right_MTM_tool_tip_linear_velocity_2',
                              33: 'Right_MTM_tool_tip_linear_velocity_3', 34: 'Right_MTM_tool_tip_rotational_velocity_1', 35: 'Right_MTM_tool_tip_rotational_velocity_2',
                              36: 'Right_MTM_tool_tip_rotational_velocity_3', 37: 'Right_MTM_gripper_angle_velocity', 
                               
                              38: 'PSM1_tool_tip_position_1', 39: 'PSM1_tool_tip_position_2',40: 'PSM1_tool_tip_position_3', 
                              41: 'PSM1_tool_tip_rotation_matrix_1', 42: 'PSM1_tool_tip_rotation_matrix_2',
                              43: 'PSM1_tool_tip_rotation_matrix_3', 44: 'PSM1_tool_tip_rotation_matrix_4', 45: 'PSM1_tool_tip_rotation_matrix_5',
                              46: 'PSM1_tool_tip_rotation_matrix_6', 47: 'PSM1_tool_tip_rotation_matrix_7', 48: 'PSM1_tool_tip_rotation_matrix_8',
                              49: 'PSM1_tool_tip_rotation_matrix_9', 50: 'PSM1_tool_tip_linear_velocity_1', 51: 'PSM1_tool_tip_linear_velocity_2',
                              52: 'PSM1_tool_tip_linear_velocity_3', 53: 'PSM1_tool_tip_rotational_velocity_1', 54: 'PSM1_tool_tip_rotational_velocity_2',
                              55: 'PSM1_tool_tip_rotational_velocity_3', 56: 'PSM1_gripper_angle_velocity', 
                               
                              57: 'PSM2_tool_tip_position_1',
                              58: 'PSM2_tool_tip_position_2', 59: 'PSM2_tool_tip_position_3', 60: 'PSM2_tool_tip_rotation_matrix_1',
                              61: 'PSM2_tool_tip_rotation_matrix_2', 62: 'PSM2_tool_tip_rotation_matrix_3', 63: 'PSM2_tool_tip_rotation_matrix_4',
                              64: 'PSM2_tool_tip_rotation_matrix_5', 65: 'PSM2_tool_tip_rotation_matrix_6', 66: 'PSM2_tool_tip_rotation_matrix_7',
                              67: 'PSM2_tool_tip_rotation_matrix_8', 68: 'PSM2_tool_tip_rotation_matrix_9', 69: 'PSM2_tool_tip_linear_velocity_1',
                              70: 'PSM2_tool_tip_linear_velocity_2', 71: 'PSM2_tool_tip_linear_velocity_3', 72: 'PSM2_tool_tip_rotational_velocity_1',
                              73: 'PSM2_tool_tip_rotational_velocity_2', 74: 'PSM2_tool_tip_rotational_velocity_3', 75: 'PSM2_gripper_angle_velocity'}, inplace=True)

        for kinematic_var in kinematic_var_list:
            scaleogram = scg.cws(df_trial[kinematic_var],cbar=None,wavelet='morl',title="",figsize=(20,6))
            scaleogram.axis('off')
            fig = scaleogram.figure
            extent = scaleogram.get_window_extent().transformed(fig.dpi_scale_trans.inverted())    
            fig.savefig(str(trial_name)+ '_' + str(kinematic_var) + '.jpg', bbox_inches=extent)


# Knot tying CWT
root_dir = os.getcwd() + '/Processed_dataset/'
knot_tying_dir = os.getcwd() + '/Processed_dataset/' + 'Knot_Tying/'
os.chdir(knot_tying_dir)

for file_name in os.listdir(knot_tying_dir):    
    # print(file_name)

    if file_name != ".DS_Store":
        os.chdir(knot_tying_dir)
        print(os.getcwd())
        generate_CWT_scaleogram(file_name)

# Needle Passing CWT
needle_passing_dir = os.getcwd() + '/Processed_dataset/' + 'Needle_Passing/'
os.chdir(needle_passing_dir)
for file_name in os.listdir(needle_passing_dir):    
    # print(file_name)

    if file_name != ".DS_Store":
        os.chdir(needle_passing_dir)
        print(os.getcwd())
        generate_CWT_scaleogram(file_name)

# Suturing CWT
suturing_dir = os.getcwd() + '/Processed_dataset/' + 'Suturing/'
os.chdir(suturing_dir)
for file_name in os.listdir(suturing_dir):    
    # print(file_name)

    if file_name != ".DS_Store":
        os.chdir(suturing_dir)
        print(os.getcwd())
        generate_CWT_scaleogram(file_name)


