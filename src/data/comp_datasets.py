import pandas as pd
import os
import json
from settings import *

# #  Designed animations
# df_dir = 'data/raw/df3_25fps.csv'
# dest_dir = 'data/processed/comp_des_gen'
#
# # Save path
# df_path = os.path.join(ROOT_PATH, df_dir)
# dest_path = os.path.join(ROOT_PATH, dest_dir)
#
# df = pd.read_csv(df_path, index_col=0, skipinitialspace=True)
# df.drop(columns=['time', 'category'], inplace=True)
# des1_labels = ['SadReaction_01', 'Frustrated_1', 'Angry_4', 'Heat_1', 'Shocked_1', 'Fear_1', 'Optimistic_1', 'Innocent_1', 'Content_01']
# des2_labels = ['Lonely_1', 'Disappointed_1', 'Fearful_1', 'Alienated_1', 'Bored_01', 'Curious_01', 'Joy_1', 'Happy_4', 'Excited_01']
#
# des1_df = pd.DataFrame()
# for l in des1_labels:
#     des1_df = pd.concat([des1_df, df.loc[df['id'] == l, :]])
#
# des2_df = pd.DataFrame()
# for l in des2_labels:
#     des2_df = pd.concat([des2_df, df.loc[df['id'] == l, :]])
#
# des1_df.reset_index(drop=True, inplace=True)
# des2_df.reset_index(drop=True, inplace=True)
#
# des1_df.to_csv(os.path.join(dest_path, 'des1_df.csv'))
# des2_df.to_csv(os.path.join(dest_path, 'des2_df.csv'))
#
#  Designed animations
df_dir = 'data/generated/sampled/cond_torus_longitude/63-250'
dest_dir = 'data/processed/comp_des_gen'

# Save path
df_path = os.path.join(ROOT_PATH, df_dir)
dest_path = os.path.join(ROOT_PATH, dest_dir)


# gen1_labels = ['l3/132_dec_long4_r3_Neg.csv', 'l2/180_dec_long4_r4_Neg.csv', 'l1/228_dec_long4_r5_Neg.csv',
#                'l1/139_dec_long6_r3_Neu.csv', 'l2/172_dec_long1_r4_Neu.csv', 'l3/226_dec_long3_r5_Neu.csv',
#                'l2/122_dec_long0_r3_Pos.csv', 'l3/176_dec_long2_r4_Pos.csv', 'l1/227_dec_long3_r5_Pos.csv']
# gen2_labels = ['l3/141_dec_long7_r3_Neg.csv', 'l1/168_dec_long0_r4_Neg.csv', 'l3/237_dec_long7_r5_Neg.csv',
#                'l3/139_dec_long6_r3_Neu.csv', 'l3/184_dec_long5_r4_Neu.csv', 'l3/226_dec_long3_r5_Neu.csv',
#                'l2/134_dec_long4_r3_Pos.csv', 'l3/182_dec_long4_r4_Pos.csv', 'l1/230_dec_long4_r5_Pos.csv']
gen1_labels = ['l3/132_dec_long4_r3_Neg.csv',
               'l1/139_dec_long6_r3_Neu.csv',
               'l2/122_dec_long0_r3_Pos.csv', 'l3/176_dec_long2_r4_Pos.csv']
gen2_labels = ['l1/168_dec_long0_r4_Neg.csv', 'l3/237_dec_long7_r5_Neg.csv',
               'l3/139_dec_long6_r3_Neu.csv',
               'l2/134_dec_long4_r3_Pos.csv']

gen1_df = pd.DataFrame()
for l in gen1_labels:
    df = pd.read_csv(os.path.join(df_path, l), index_col=0, skipinitialspace=True)
    df.drop(columns=['time', 'valence'], inplace=True)
    last_idx = len(df)
    df.loc[last_idx, joints_names] = standInit
    df.loc[last_idx, leds_keys] = 1
    gen1_df = pd.concat([gen1_df, df])

gen2_df = pd.DataFrame()
for l in gen2_labels:
    df = pd.read_csv(os.path.join(df_path, l), index_col=0, skipinitialspace=True)
    df.drop(columns=['time', 'valence'], inplace=True)
    last_idx = len(df)
    df.loc[last_idx, joints_names] = standInit
    df.loc[last_idx, leds_keys] = 1
    gen2_df = pd.concat([gen2_df, df])

gen1_df.reset_index(drop=True, inplace=True)
gen2_df.reset_index(drop=True, inplace=True)

gen1_df.to_csv(os.path.join(dest_path, 'gen1_df.csv'))
gen2_df.to_csv(os.path.join(dest_path, 'gen2_df.csv'))