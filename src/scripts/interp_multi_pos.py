import pandas as pd
import os
import json
from settings import *
from src.utils.sampu import interp_multi, sel_pos_frame, normalize
import seaborn as sns
sns.set(style="darkgrid")


"""Given some keyframe numbers (normalized kf), encodes them and interpolates their latent datapoints. 
    Saves the z interpolants and the decoded animations in a df. 
"""
check_model = '42'
check_epoch = '-200'
method = 'lerp'  # slerp, lerp, bspline
nsteps = 100    # per segment
fr = 0.06
frames = [0, 465, 354, 289, 252, 0]  # Has to be 2 or 4 or higher. Add 0 for standInit
x_dataset = 'df14_KF.csv'  # 'df14_KF.csv': radians, normalized in [0,1]
latent = False  # latent=True for interp the latent space directly without encoding keyframes before

# Load keyframes dataset
df = pd.read_csv(os.path.join(ROOT_PATH, 'data/processed/keyframes/', x_dataset), index_col=0)

# Postures in radians
pos_list = []
id_anim_list = []

for frame in frames:
    if frame == 0:
        pos_list.append(standInit_norm)  # List of lists
        id_anim_list.append('standInit_0')
    else:
        pos, id_anim = sel_pos_frame(df, frame)
        pos_list.append(pos)  # List of lists
        id_anim_list.append(id_anim + '_f' + str(frame))

# Get the radians frames (dec, denorm) and the latent interpolants
df_dec_interp, df_z_interp = interp_multi(pos_list, latent, nsteps, check_model, check_epoch, method, joints_names)

# Add 'time' column based on frequency fr
end = df_dec_interp.shape[0] * fr + 0.02
df_dec_interp['time'] = list(np.arange(0.02, end, fr))

# Save path
df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'interp_multi_pos')

# Prepare the overview
json_file = os.path.join(df_path, '-overview.json')

with open(json_file, 'r') as fd:
    files_dict = json.load(fd)

file_id = len(files_dict)

files_dict[file_id] = {
    'file_id': file_id,
    'interp_method': method,
    'interp_steps': nsteps,
    'frequency': fr,
    'model': check_model + check_epoch,
    'animations': id_anim_list,
    'frames': frames
}
with open(json_file, 'w') as fd:
    fd.write(json.dumps(files_dict))

# Save
df_dec_interp.to_csv(os.path.join(df_path, str(file_id) + '_dec_' + method + '.csv'))
df_z_interp.to_csv(os.path.join(df_path, str(file_id) + '_z_' + method + '.csv'))

