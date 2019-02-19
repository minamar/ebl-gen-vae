import pandas as pd
import os
from settings import *
from src.utils.sampu import interp_2pos, sel_pos_frame
import seaborn as sns
sns.set(style="darkgrid")

check_model = '42'
check_epoch = '-200'
method = 'lerp'  # slerp, lerp, bezier
nsteps = 100
fr = 0.06
frame = 465
x_dataset = 'df14_KF.csv'  # 'df14_KF.csv': radians, normalized in [0,1]


# Load animation dataset and z_means
df = pd.read_csv(os.path.join(ROOT_PATH, 'data/processed/keyframes/', x_dataset), index_col=0)

posA = standInit_norm
posB, id_anim = sel_pos_frame(df, frame)

# Get the radians frames generated from laten interpolant
output_df, df_z_interp = interp_2pos(posA, posB, nsteps, check_model, check_epoch, method)

output_df['id'] = id_anim + '_' + str(frame)

end = output_df.shape[0] * fr + 0.02

output_df['time'] = list(np.arange(0.02, end, fr))

df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'interp_2postures')

output_df.to_csv(os.path.join(df_path, method + '_' + check_model + check_epoch + '_' + output_df.loc[0, 'id'] + '.csv'))

df_z_interp['interp'] = method
df_z_interp.to_csv(os.path.join(df_path, method + '_' + 'z' + '_' + check_model + check_epoch + '_' + output_df.loc[0, 'id'] + '.csv'))
# frame 252, 'Fearful' from df14_KF
# posB = [0.652216347589752, 0.3144962438895381, 0.5372509258275667, 0.4124938990517932, 0.4422894920178799,
#         0.23146740282176015, 0.5309949774788336, 0.5156781149799264, 0.3699707180121343, 0.053170899809992,
#         0.6205521504119926, 0.9110618130472923, 0.8354058404193228, 0.7041614444806631, 0.7235819968218229,
#         0.8405977500971131, 0.3620784719863175]
