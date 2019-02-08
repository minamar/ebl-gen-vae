import pandas as pd
import os
from settings import *
from src.utils.visu import x_all2z
from src.utils.sampu import load_model, encode

""" For a given model and dataset save the plots of the latent space per animation (latent_z dir in visualizations), 
    a plot of all the encoded animations in the latent space, color coded per anim and per category  
"""

check_model = '31'
check_epoch_list = ['-500']  # ['-0','-50','-100','-150','-200','-250','-300','-350','-400','-450','-500']
scaler = 'j_scaler_nao_lim_df13_50fps.pkl'
x_dataset = 'df25_10fps.csv'
ccode = 'id'  # How to colorcode the animations in latent space
save_stuff = False
all_epochs = True
dataset_dir = x_dataset.split('.')[0]

# Load animation dataset
df_x = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, x_dataset), index_col=0)

df_x = df_x[~df_x.id.str.contains('_tr')]
x = df_x.drop(columns=['time', 'id', 'category'], inplace=False)

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

    fig = x_all2z(df_z_mean, ccode)

    # GRAPH: Bar plot mean of latent dimensions standard deviations across postures
    import matplotlib.pyplot as plt
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

    plt.show()

    if save_stuff:
        plot_path = os.path.join(ROOT_PATH, DATA_VISU, 'latent_z', dataset_dir)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plot_path = os.path.join(plot_path, check_model)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(os.path.join(plot_path, check_model + check_epoch + '_' + ccode + '.eps'), bbox_inches='tight',
                    format='eps', dpi=1000)
