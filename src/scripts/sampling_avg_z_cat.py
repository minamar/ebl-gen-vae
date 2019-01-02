import pandas as pd
import os
from settings import *
from src.utils.sampu import load_model, decode
from src.data.post_processing import inverse_norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


check_model = '5'
check_epoch = '-500'
x_dataset = 'df13_50fps.csv'
z_dataset = 'mean_5-500_df13_50fps'
save_stuff = True
category = 'Neu/Cal'

# Restore model to get the decoder
model = load_model(check_model, check_epoch)

# Load animation dataset and z_means
df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, x_dataset), index_col=0)
df_z_means = pd.read_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, z_dataset), index_col=0)

# Get all z_means of anims in the category
df_z_cat = df_z_means.loc[df_z_means['category'] == category, :]
df_z_cat = df_z_cat[~df_z_cat.id.str.contains('_tr')]
df_z_cat = df_z_cat[df_z_cat['id'].isin(['Angry_3', 'AskForAttention_3'])]

# z dimension
z_dim = df_z_cat.shape[1] - 2

# List of all the ids in the category
id_list = df_z_cat['id'].unique().tolist()

# Get maximum length among animations within category
max_step = df_z_cat.groupby('id').count().max().max()

# Array to aggregate the padded animations z_means. dim is (n_anims in cat, max_length in cat, n_z_dim)
all_anims_z = np.zeros([len(id_list), max_step, z_dim])

pad_z = np.asarray([-0.10972053, -1.0265701, 0.27543408, 0.31275314, -0.04693897])
i = 0
for id_anim in id_list:

    df_single_anim = df_z_cat.loc[df_z_cat['id'] == id_anim, :]
    single_anim_mat = df_single_anim.iloc[:, 0:z_dim].values

    n_tile = max_step - single_anim_mat.shape[0]
    tiled_pad = np.tile(pad_z, (n_tile, 1))
    all_anims_z[i, :single_anim_mat.shape[0], :] = single_anim_mat
    all_anims_z[i, single_anim_mat.shape[0]:, :] = tiled_pad
    i += 1

# Average all the latent_means across the animations in the category
avg_z = np.mean(all_anims_z, axis=0)

# Decode the averaged latent means for this category
x_reconstruction = decode(avg_z, model)

output_df = pd.DataFrame(columns=joints_names, data=x_reconstruction)

# Inverse normalization
scaler = 'j_scaler_nao_lim_df23_50fps.pkl'
output_df = inverse_norm(output_df, scaler)
# Add the time vector
output_df['time'] = np.linspace(0.02, (output_df.shape[0] + 1) * 0.02, output_df.shape[0])

# Label for the id
output_df['id'] = 'Gen_' + category + '_avg_z'


# GRAPH: Plot the encoded animations in the latent space (color-coded per category)
fig1 = plt.figure()

if z_dim <= 2:
    ax = plt.axes()
    sns.scatterplot(x='l1', y='l2', hue='category', data=df_z_cat, legend=False)
else:
    ax = plt.axes(projection='3d')
    ids = df_z_cat['id'].unique()
    for anim in ids:
        df = df_z_cat.loc[df_z_cat['id'] == anim, :]
        ax.scatter(df['l1'], df['l2'], df['l3'], label=anim)

        ax.set_xlabel('l1')
        ax.set_ylabel('l2')
        ax.set_zlabel('l3')
    ax.scatter(avg_z[:, 0], avg_z[:, 1], avg_z[:, 2])
ax.set_title('latent encoding per anim in category')
ax.axis('equal')
ax.axis('square')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='best', ncol=1)
plt.show()

cat_id = category.replace('/', '_')

if save_stuff:
    # Save generated
    output_df.to_csv(os.path.join(ROOT_PATH, 'data/generated/generated_VAE/vae_sampled_1/avg_latent_category',
                                  z_dataset + '_' + 'Gen_' + cat_id + '_avg_z'))

# import sys
# sys.exit()