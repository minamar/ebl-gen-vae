import numpy as np
import pandas as pd
import os
from settings import *

"""
Augment the original training examples by adding anti-symmetrical ones in terms of left/right motion. 
Doubles the quantity of examples. The new examples have reversed sign for the joint values of 
'HeadYaw', 'HipRoll', swapped values for Arms (RArm to LArm and vice versa).   
"""

# Original keyframes and destination file to write
data_x_set = 'df3_25fps.csv'
dest_x_set = 'df31_25fps.csv'

data_y_set = 'y_va_cat.csv'
dest_y_set = 'y_va_cat_aug.csv'

# If True, valence and arousal labels are also augmented
augment_labels = False

# Augment the training examples first
path = os.path.join(ROOT_PATH, RAW_DATA, data_x_set)
df = pd.read_csv(path, index_col=0)

# Reverse the sign of HeadYaw and HipRoll values
df_tr = df.copy(deep=True)
df_tr.loc[:, ['HeadYaw', 'HipRoll']] = -df_tr.loc[:, ['HeadYaw', 'HipRoll']]

# Swap sides: RArm to LArm and vice versa
swap_l = df_tr.loc[:, ['LShoulderPitch', 'LShoulderRoll', 'LElbowRoll', 'LElbowYaw', 'LWristYaw', 'LHand']]
swap_r  = df_tr.loc[:, ['RShoulderPitch', 'RShoulderRoll', 'RElbowRoll', 'RElbowYaw', 'RWristYaw', 'RHand']]
df_tr[['LShoulderPitch', 'LShoulderRoll', 'LElbowRoll', 'LElbowYaw', 'LWristYaw', 'LHand']] = swap_r
df_tr[['RShoulderPitch', 'RShoulderRoll', 'RElbowRoll', 'RElbowYaw', 'RWristYaw', 'RHand']] = swap_l

# Inverse signs for arms roll and yaw joints
inv_joints = ['LShoulderRoll', 'LElbowRoll', 'LElbowYaw', 'LWristYaw', 'RShoulderRoll', 'RElbowRoll', 'RElbowYaw', 'RWristYaw']
df_tr.loc[:, inv_joints] = -df_tr.loc[:, inv_joints]

# Add a '_tr' suffix (stands for 'transformed') to the anim id
df_tr['id'] = df_tr['id'] + '_tr'

# Add to previous dataframe
augm_df = pd.concat([df, df_tr], ignore_index=True)

dest = os.path.join(ROOT_PATH, DATA_X_PATH, dest_x_set)
augm_df.to_csv(dest)

# Augment the labels
if augment_labels:

    path = os.path.join(ROOT_PATH, DATA_Y_PATH, data_y_set)
    df_y = pd.read_csv(path, index_col=0)

    df_y_tr = df_y.copy(deep=True)
    df_y_tr['nameAnim'] = df_y_tr['nameAnim'] + '_tr'
    augm_df_y = pd.concat([df_y, df_y_tr], ignore_index=True)

    dest = os.path.join(ROOT_PATH, DATA_Y_PATH, dest_y_set)
    augm_df_y.to_csv(dest)