import pandas as pd
import tensorflow as tf
from tfmodellib.vae import VAE, VAEConfig, variational_loss, build_vae_latent_layers
from settings import *
from src.utils.sampu import merge_with_VA_labels, v2cat
import os
import random
import time
import seaborn as sns
import numpy
sns.set(style="darkgrid")

dataset = 'df32_25fps.csv'
split = None #'anim'
subset = None  # from settings pos_neu_ids
# 'Emotions/Positive'

lr = 0.0001
latent_range = [3]
batch = 64
encoder = [128, 512, 512, 128]
decoder = [128, 512, 128]
n_epoch = 401
wu = False  # Warm-up
beta = 0.001
beta_range = np.linspace(0.0001, 0.01, n_epoch)
kern_init = 'xavier_uniform'
save_overview = True
v_only = True
v2cat_flag = True


df_over = pd.read_csv(os.path.join(ROOT_PATH, 'reports', 'overview.csv'), index_col=0, skipinitialspace=True)

# Load anims df: motion + leds
df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, dataset), index_col=0)
# Merge with valence and arousal scores
df_anim = merge_with_VA_labels(df_anim, 'y_va_cat_aug.csv')

if v_only:
    df_anim.drop(columns='arousal', inplace=True)
    if v2cat_flag:
        df_anim = v2cat(df_anim)

# If not None you get to train just pos valence VAE
if subset is not None:
    df_anim = df_anim.loc[df_anim['id'].isin(subset), :]
    df_anim.reset_index(drop=True, inplace=True)
    print("Number of animations: ", len(df_anim['id'].unique()))
    print("Number of postures in total: ", df_anim.shape[0])
    print(df_anim['id'].unique())

# Split training and validation sets with anims, not just postures
if split == 'anim':
    ids = df_anim['id'].unique().tolist()
    train_idx = random.sample(range(0, 72), 58)  # 72 animations in total, 58 for train and the rest for validation
    id_train = list(ids[i] for i in train_idx)
    df_train = df_anim.loc[df_anim['id'].isin(id_train), :]
    df_valid = df_anim.loc[~df_anim['id'].isin(id_train), :]
    id_valid = df_valid['id'].unique().tolist()
    id_train.sort()
    id_valid.sort()

    x_train = df_train.drop(columns=['time', 'id', 'category'], inplace=False).values
    x_valid = df_valid.drop(columns=['time', 'id', 'category'], inplace=False).values

    np.random.shuffle(x_train)
    np.random.shuffle(x_valid)

else:
    df_postures = df_anim.drop(columns=['time', 'id', 'category'], inplace=False)
    x = df_postures.values
    # Shuffle data here. Permutation(x)
    np.random.shuffle(x)

    x_train = x[:int(x.shape[0]*0.95)]
    x_valid = x[x_train.shape[0]:]


for latent_size in latent_range:
    prefetch = batch
    start = time.time()
    # The model
    conf = VAEConfig(
            in_size=66,
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

    # Training
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
