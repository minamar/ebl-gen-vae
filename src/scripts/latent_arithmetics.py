import pandas as pd
import os
from scipy import signal
from src.data.post_processing import inverse_norm
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from settings import *
from src.utils.sampu import load_model, decode
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


""" Select two animations, encode them in the latent space and blend them. Then decode and save 
"""
check_model = '47'
check_epoch = '-500'
z_dataset = '47-500_df24_20fps_mean.csv'

func_anim_id = 'IDontKnow_1'
func_anim_cat = 'Gestures'
emo_anim_id = 'Angry_4'
emo_anim_cat = 'Emotions/Negative'

blend_mode = 'gmm'  # 'conv', 'pca', 'ica', 'fa'

dest = os.path.join(ROOT_PATH, 'data/generated/arithmetic')
# Load the latent animations
df_z_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, z_dataset), index_col=0)
df_z_anim.astype({'l1': 'float32', 'l2': 'float32', 'l3': 'float32'})

func_anim_z = df_z_anim.loc[(df_z_anim['category'] == func_anim_cat) & (df_z_anim['id'] == func_anim_id), :]
emo_anim_z = df_z_anim.loc[(df_z_anim['category'] == emo_anim_cat) & (df_z_anim['id'] == emo_anim_id), :]

func = func_anim_z.loc[:, ['l1', 'l2', 'l3']].values
emo = emo_anim_z.loc[:, ['l1', 'l2', 'l3']].values

# Padding with zeros to make animations equal length
func_n = func.shape[0]
emo_n = emo.shape[0]

if func_n > emo_n:
    diff = func_n - emo_n
    emo_new = np.zeros_like(func)
    start_pad = int(np.floor(diff/2))
    emo_new[start_pad:(start_pad+emo_n), :] = emo
    emo = emo_new

elif emo_n > func_n:
    diff = emo_n - func_n
    func_new = np.zeros_like(emo)
    start_pad = int(np.floor(diff / 2))
    func_new[start_pad:(start_pad + emo_n), :] = func
    func = func_new


conv = signal.fftconvolve(func, emo, mode='same', axes=0)
conv /= np.linalg.norm(conv, axis=0)
conv = conv * 8

#  PCA
X = np.concatenate((func, emo), axis=1)
pca = PCA(n_components=3, svd_solver='randomized')
pca.fit(X.transpose())
pca = pca.components_.transpose()
pca = pca * 10

# ICA
transformer = FastICA(n_components=3, random_state=0)
ica = transformer.fit_transform(X)
ica = ica * 8

# Fa
transformer = FactorAnalysis(n_components=3, random_state=0)
fa = transformer.fit_transform(X)

# GMM
from sklearn import mixture
gmmodel = mixture.GaussianMixture(n_components=3, covariance_type='tied', max_iter=100, random_state=10).fit(X.transpose())
gmm = gmmodel.means_.transpose()
gmm_samp, gmm_y = gmmodel.sample(118)

# Plot latent anims and blends
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.plot(func[:, 0], func[:, 1], func[:, 2], label='func')
ax.plot(emo[:, 0], emo[:, 1], emo[:, 2], label='emo')
# ax.scatter(conv[:, 0], conv[:, 1], conv[:, 2], label='conv')
# ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], label='pca')
# ax.scatter(ica[:, 0], ica[:, 1], ica[:, 2], label='ica')
# ax.scatter(ica[:, 0], ica[:, 1], ica[:, 2], label='fa')
ax.plot(gmm[:, 0], gmm[:, 1], gmm[:, 2], label='gmm')
ax.set_xlabel('l1')
ax.set_ylabel('l2')
ax.set_zlabel('l3')
ax.set_title('Animations encoded in the latent space')
ax.axis('equal')
ax.axis('square')
ax.legend()
plt.show()

# Decode and normalize
if blend_mode == 'conv':
    blend = conv
elif blend_mode == 'pca':
    blend = pca
elif blend_mode == 'ica':
    blend = ica
elif blend_mode == 'fa':
    blend = fa
elif blend_mode == 'gmm':
    blend = gmm
else:
    print("Cannot understand the mode")

model = load_model(check_model, check_epoch)
dec = decode(blend, model)
df_dec = pd.DataFrame(columns=joints_names, data=dec)
scaler = 'j_scaler_nao_lim_df13_50fps.pkl'
df_dec_norm = inverse_norm(df_dec, scaler)

# Save df
df_dec_norm.to_csv(os.path.join(dest, func_anim_id + '_' + emo_anim_id + '_' + blend_mode + '.csv'))
