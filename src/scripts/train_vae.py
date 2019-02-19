import pandas as pd
import tensorflow as tf
from tfmodellib.vae import VAE, VAEConfig, variational_loss, build_vae_latent_layers
from settings import *
import os
import time
import seaborn as sns
import numpy
sns.set(style="darkgrid")

dataset = 'df14_20fps.csv'

lr = 0.0001
latent_range = [3]
batch = 32
encoder = [128, 512, 512, 128]
decoder = [128, 128, 128]
n_epoch = 201
wu = False  # Warm-up
beta = 0.001
beta_range = np.linspace(0.0001, 0.01, n_epoch)
kern_init = 'xavier_uniform'
save_overview = True

df_over = pd.read_csv(os.path.join(ROOT_PATH, 'reports', 'overview.csv'), index_col=0, skipinitialspace=True)

# Load anims
df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, dataset), index_col=0)
df_postures = df_anim.drop(columns=['time', 'id', 'category'], inplace=False)

x = df_postures.values

# Shuffle data here. Permutation(x)
np.random.shuffle(x)

x_train = x[:int(x.shape[0]*0.8)]
x_valid = x[x_train.shape[0]:]

for latent_size in latent_range:
    prefetch = batch
    start = time.time()
    # create the model
    conf = VAEConfig(
            in_size=17,
            latent_size=latent_size,
            encoder_size=encoder,
            decoder_size=decoder,
            hidden_activation=tf.nn.relu,
            output_activation=None,
            reconstruction_loss=tf.losses.mean_squared_error,
            use_bn=False,
            use_dropout=True,
            shuffle_buffer=True,
            prefetch=prefetch,
            summaries_root=os.path.join(ROOT_PATH, 'reports', 'summaries'),
            checkpoints_root=os.path.join(ROOT_PATH, 'reports', 'checkpoints'),
            step_summaries_interval=1,
            saver_interval=50,  # How often to save a checkpoint
    )
    model = VAE(conf)

    # run the training
    for t in range(n_epoch):

        # Warm-up
        if wu:
            beta = beta_range[t]

        print('Epoch: '+str(t))
        np.random.shuffle(x_train)
        tr_loss, val_loss, rec_loss, var_loss = model.train(
                train_inputs=x_train,
                train_targets=x_train,
                validation_inputs=x_valid,
                validation_targets=x_valid,
                learning_rate=lr,
                batch_size=batch,
                beta=beta)

    if save_overview:
        # overview = [checkpoint, dataset, in_size, latent_size, encoder_size, batch_size, prefetch, learning_rate, beta, output_activation, reconstruction_loss]
        conf = model.config
        df_over.loc[len(df_over)] = [model.summaries_ind, dataset, conf['in_size'], conf['latent_size'],
                                     conf['encoder_size'][::-1], batch, prefetch, lr, beta, conf['use_dropout'],
                                     conf['use_bn'], wu, kern_init, "{0:.4f}".format(tr_loss), "{0:.4f}".format(val_loss), "{0:.4f}".format(rec_loss), "{0:.4f}".format(var_loss)]
        df_over.to_csv(os.path.join(ROOT_PATH, 'reports', 'overview.csv'))

        numpy.savetxt(os.path.join(ROOT_PATH, 'data/tr_val_sets/') + str(model.summaries_ind) + "_x_train.csv", x_train, delimiter=",")
        numpy.savetxt(os.path.join(ROOT_PATH, 'data/tr_val_sets/') + str(model.summaries_ind) + "_x_valid.csv", x_valid, delimiter=",")

    print("Total time: " + str(time.time()-start))
