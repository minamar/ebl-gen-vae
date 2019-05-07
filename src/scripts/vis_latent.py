import pandas as pd
import os
from settings import *
from src.utils.visu import x_all2z, set_pub
from src.utils.sampu import load_model, encode, merge_with_VA_labels
import matplotlib.pyplot as plt
""" For a given model and dataset save the plots of the latent space per animation (latent_z dir in visualizations), 
    a plot of all the encoded animations in the latent space, color coded per anim and per category  
"""
set_pub()
check_model = '61'
check_epoch_list = ['-100']  # ['-0','-50','-100','-150','-200','-250','-300','-350','-400','-450','-500']
scaler = 'j_scaler_nao_lim_df32_25fps.pkl'
x_dataset = 'df32_25fps.csv'
ccode = 'id'  # How to colorcode the animations in latent space
save_stuff = False
all_epochs = True
grid_show = False
dataset_dir = x_dataset.split('.')[0]

# Load animation dataset
df_x = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, x_dataset), index_col=0)
# df_x = pd.read_csv(os.path.join(ROOT_PATH, 'data/processed/keyframes/', x_dataset), index_col=0)
df_x = df_x[~df_x.id.str.contains('_tr')]
df_x = df_x[~df_x.category.str.contains('Neu/Calm')]

# For CVAE
df_x = merge_with_VA_labels(df_x)
x = df_x.drop(columns=['time', 'id', 'category'], inplace=False)

# x = df_x.drop(columns=['time_diff', 'id'], inplace=False)

for check_epoch in check_epoch_list:
    # Restore model to get the decoder
    model = load_model(check_model, check_epoch)

    # Encode the animation set x
    latent_mean, latent_sigma = encode(x, model)

    # Get the best dimensions of the latent space
    latent_sigma_mean = latent_sigma.mean(axis=0)
    dim = latent_mean.shape[1]  # Number of dimensions in the latent space
    dim_inds = np.argsort(latent_sigma_mean)  # Prioritize
    latent_z_mean = latent_mean[:, dim_inds].transpose()

    # Create column names for latent dimensions and put them in a dataframe
    l_dim_names = []
    df_z_mean = pd.DataFrame()
    for d in range(dim):
        name = 'l' + str(d + 1)
        df_z_mean[name] = latent_z_mean[d, :]
        l_dim_names.append(name)

    df_z_mean['id'] = df_x['id']
    df_z_mean['category'] = df_x['category']

    fig = x_all2z(df_z_mean, ccode, leg=False)


    # GRAPH: Bar plot mean of latent dimensions standard deviations across postures

    fig2 = plt.figure(figsize=(12, 5))

    ax1 = fig2.add_subplot(2, 2, 1)
    ax1.bar(range(latent_sigma_mean.size), latent_sigma_mean[dim_inds])
    ax1.set_title('mean(latent sigmas)')

    ax1.set_ylabel('sigma')

    # Bar plot mean standard deviation of latent dimensions
    ax1 = fig2.add_subplot(2, 2, 2)
    ax1.bar(range(np.std(latent_mean, axis=0).size), np.std(latent_mean[:, dim_inds], axis=0))
    ax1.set_title('std(latent means)')

    ax1.set_ylabel('mean')

    # Boxplot standard deviation of latent dimensions
    ax1 = fig2.add_subplot(2, 2, 3)
    ax1.boxplot(latent_sigma[:, dim_inds])
    ax1.set_title('latent sigmas')
    ax1.set_xlabel('latent dimension')

    # Boxplot means of latent dimensions
    ax1 = fig2.add_subplot(2, 2, 4)
    ax1.boxplot(latent_mean[:, dim_inds])
    ax1.set_title('latent means')
    ax1.set_xlabel('latent dimension')

    if grid_show:
        method = 'slerp'
        nsteps = 10  # per segment
        fr = 0.06
        radiuses = [3]  # Sampling radius [3]  #[
        circles = 6  # Parallel to z axis. Equal to x2 longitudes
        lat = 41  # Points on a circe (first and last are the same)
        southp = int(lat / 2) + 1  # idx to split lat points to form half circle (longitude)
        top_down = True  # Save postures on the longitude from north to south pole if True

        # Save path
        df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'interp_grid_longitude/52-200/l1')

        for radius in radiuses:
            phi = np.linspace(0, np.pi, circles)  # 10 times the perimeter, parallel to z axis (longitude)
            theta = np.linspace(0, 2 * np.pi, lat)  # (latidude) in parallel to x,y plane
            x = radius * np.outer(np.sin(theta), np.cos(phi))  # the plane
            y = radius * np.outer(np.sin(theta), np.sin(phi))  # the plane
            z = radius * np.outer(np.cos(theta), np.ones_like(phi))  # the axis

            plt.plot(x.flatten('F'), y.flatten('F'), z.flatten('F'))

    plt.show()

    if save_stuff:
        plot_path = os.path.join(ROOT_PATH, DATA_VISU, 'latent_z', dataset_dir)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plot_path = os.path.join(plot_path, check_model)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(os.path.join(plot_path, check_model + check_epoch + '_' + ccode + '.eps'),
                    format='eps', dpi=1000)
        fig2.savefig(os.path.join(plot_path, check_model + check_epoch + '_' + 'boxplot' + '.eps'), bbox_inches='tight',
                    format='eps', dpi=1000)
