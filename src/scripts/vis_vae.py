import numpy as np
import pandas as pd
import os
import time
import random
from tfmodellib.vae import VAE, VAEConfig, variational_loss, build_vae_latent_layers
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from settings import *
from src.data.post_processing import inverse_norm
import seaborn as sns
sns.set(style="darkgrid")

check_model = '3'
check_epoch = '-500'
save_stuff = False

start = time.time()

check_path = os.path.join(ROOT_PATH, 'reports', 'checkpoints', check_model)

# Restore the configuration of the model
conf_restored = VAEConfig()
conf_restored.load(os.path.join(check_path, 'config.json'))

conf_restored['summaries_root'] = None
conf_restored['checkpoints_root'] = None
conf_restored['saver_interval'] = None
conf_restored['step_summaries_interval'] = None

# Restore the model
model_restored = VAE(conf_restored)
model_path = os.path.join(check_path, check_epoch)
model_restored.restore(model_path)

# # Load animations and their categories
df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, 'df23_50fps.csv'), index_col=0)
# df_categories = pd.read_csv(os.path.join(ROOT_PATH, DATA_Y_PATH, 'y_va_cat_aug.csv'), usecols=['nameAnim', 'category'], index_col=0)
# df_anim = pd.merge(df_anim, df_categories, left_on='id', right_on='nameAnim', how='left')
df_anim = df_anim[~df_anim.id.str.contains('_tr')]
x = df_anim.drop(columns=['time', 'id', 'category'], inplace=False)
x = x.values

# Encode the animation set x
latent_mean, latent_sigma = model_restored.sess.run([model_restored.latent_mean, model_restored.latent_sigma],
                                                    feed_dict={model_restored.x_input: x,
                                                               model_restored.bn_is_training: False})

latent_sigma_mean = latent_sigma.mean(axis=0)

dim = latent_mean.shape[1]

# Print Means of latent Standard Deviations per dim
print('Means for latent std: {}'.format(latent_sigma_mean))
# Print Standard Deviations of the latent means per dim
print('Std for latent means: {}'.format(np.std(latent_mean, axis=0)))

# # Get the best dimensions of the latent space and concat on axis=1 with the df_anim
dim_inds = np.argsort(latent_sigma_mean)  # Prioritize
latent_dims = latent_mean[:, dim_inds].transpose()

l_dim_names = []
for d in range(dim):
    name = 'l'+str(d+1)
    df_anim[name] = latent_dims[d, :]
    l_dim_names.append(name)

# GRAPH: the original animations dim reduced (color-coded per animation)

new_d = 3
with_tsne = False

if with_tsne:
    # Get the joints values only from the df_anim
    joints_values = df_anim.loc[:, joints_names].values
    X = np.array(joints_values)
    # TSNE
    X_embedded = TSNE(n_components=3, perplexity=30.0, n_iter=250, init='pca', random_state=4).fit_transform(X)

    df_tsne = df_anim.loc[:, ['id', 'category']]
    df_tsne['X'] = X_embedded[:, 0]
    df_tsne['Y'] = X_embedded[:, 1]
    df_tsne['Z'] = X_embedded[:, 2]

    fig4 = plt.figure()
    ax = plt.axes(projection='3d')
    ids = df_anim['id'].unique()

    for id_anim in ids:
        df = df_tsne.loc[df_tsne['id'] == id_anim, :]
        ax.scatter(df['X'], df['Y'], df['Z'], label=id_anim)
        ax.axis('equal')

        ax.set_xlabel('d1')
        ax.set_ylabel('d2')
        ax.set_zlabel('d3')
else:
    df_anim_3d = df_anim.iloc[:,[1,2,3]]
    fig4 = plt.figure()
    ax = plt.axes(projection='3d')

    ids = df_anim['id'].unique()
    joints = random.sample(range(0, 17), 3)

    for id_anim in ids:
        df = df_anim.loc[df_anim['id'] == id_anim, :]
        ax.scatter(df.iloc[:, joints[0]], df.iloc[:, joints[1]], df.iloc[:, joints[2]], label=id_anim)
        ax.axis('equal')

        ax.set_xlabel(df.columns[joints[0]])
        ax.set_ylabel(df.columns[joints[1]])
        ax.set_zlabel(df.columns[joints[2]])

    ax.set_title('original dataset')
    ax.axis('equal')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.axis('square')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best', ncol=1)

# GRAPH: the encoded animations in the latent space (color-coded per animation)
fig = plt.figure()

if dim <= 2:
    ax = plt.axes()
    sns.scatterplot(x='l1', y='l2', hue='id', data=df_anim, legend=False)
else:
    ax = plt.axes(projection='3d')
    ids = df_anim['id'].unique()
    for id_anim in ids:
        df = df_anim.loc[df_anim['id'] == id_anim, :]
        ax.scatter(df['l1'], df['l2'], df['l3'], label=id_anim)

        ax.set_xlabel('l1')
        ax.set_ylabel('l2')
        ax.set_zlabel('l3')

ax.set_title('latent encoding per animation')
ax.axis('equal')
ax.axis('square')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='best', ncol=1)


# GRAPH: Plot the encoded animations in the latent space (color-coded per category)
fig1 = plt.figure()

if dim <= 2:
    ax = plt.axes()
    sns.scatterplot(x='l1', y='l2', hue='category', data=df_anim, legend=False)
else:
    ax = plt.axes(projection='3d')
    ids = df_anim['id'].unique()
    for cat in all_categories:
        df = df_anim.loc[df_anim['category'] == cat, :]
        ax.scatter(df['l1'], df['l2'], df['l3'], label=cat)

        ax.set_xlabel('l1')
        ax.set_ylabel('l2')
        ax.set_zlabel('l3')

ax.set_title('latent encoding per category')
ax.axis('equal')
ax.axis('square')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='best', ncol=1)


# GRAPH: Bar plot mean of latent dimensions standard deviations across postures
fig2 = plt.figure(figsize=(12, 5))

ax1 = fig2.add_subplot(1, 2, 1)
ax1.bar(range(latent_sigma_mean.size), latent_sigma_mean[dim_inds])
ax1.set_title('mean across postures\' standard deviations')
ax1.set_xlabel('latent dimension')
ax1.set_ylabel('sigma')

# Bar plot mean standard deviation of latent dimensions
ax1 = fig2.add_subplot(1, 2, 2)
ax1.bar(range(np.std(latent_mean, axis=0).size), np.std(latent_mean[:, dim_inds], axis=0))
ax1.set_title('standard deviation of the means across postures')
ax1.set_xlabel('latent dimension')
ax1.set_ylabel('mean')

# Reconstruct Plymouth animations from their trained latent_mean
x_reconstruction = model_restored.sess.run([model_restored.y_output],
                                           feed_dict={model_restored.latent_layer: latent_mean,
                                                      model_restored.bn_is_training: False})
x_anim = x_reconstruction[0]

output_df = pd.DataFrame(columns=joints_names, data=x_anim)

# # Inverse normalization
scaler = 'j_scaler_nao_lim_df13_50fps.pkl'
output_df = inverse_norm(output_df, scaler)

# Plot an animation across time and below all latent_means across time
fig3, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
df_anim.loc[df_anim['id'] == 'Happy_4'].plot(y=joints_names, kind='line', ax=axes[0])
df_anim.loc[df_anim['id'] == 'Happy_4'].plot(y=l_dim_names, kind='line', ax=axes[1])
output_df.loc[output_df['id'] == 'Happy_4'].plot(y=joints_names, kind='line', ax=axes[2])
axes[0].get_legend().remove()
axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

# Generate the reconstructed df
for id_anim in ids:
    print(id_anim)
    df_idx = list(df_anim.loc[df_anim['id'] == id_anim].index)
    first = df_idx[0]
    last = df_idx[-1]
    df = df_anim.loc[df_anim['id'] == id_anim, :]
    anim_length = df.shape[0]
    print("First:" + str(first), 'Last:' + str(last), 'anim_length:' + str(anim_length))
    output_df.loc[first:last, 'time'] = np.linspace(0.02, (anim_length + 1) * 0.02, anim_length)
output_df['id'] = df_anim['id']

if save_stuff:
    # Save generated
    output_df.to_csv(os.path.join(ROOT_PATH, 'data/generated/generated_VAE/reconstruction', check_model + check_epoch + '.csv'))

# Display the latent space
plt.show()

if save_stuff:
    # Save the fig of the latent space and the bar plot of mean std
    fig.savefig(os.path.join(ROOT_PATH, 'reports/vae/visualizations', check_model + check_epoch + '_latent_per_anim'))
    fig1.savefig(os.path.join(ROOT_PATH, 'reports/vae/visualizations', check_model + check_epoch + '_latent_per_cat'))
    fig2.savefig(os.path.join(ROOT_PATH, 'reports/vae/visualizations', check_model + check_epoch + '_barplots'))
    fig3.savefig(os.path.join(ROOT_PATH, 'reports/vae/visualizations', check_model + check_epoch + '_joints_vs_latentD'))
    # fig4.savefig(os.path.join(ROOT_PATH, 'reports/vae/visualizations', check_model + check_epoch + '_original_3d_distr'))


