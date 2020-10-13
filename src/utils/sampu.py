import pandas as pd
import os
from settings import *
from tfmodellib.vae import VAE, VAEConfig
from scipy.interpolate import CubicSpline
import numpy as np
from sklearn.externals import joblib
from src.data.post_processing import inverse_norm
from sklearn.manifold import TSNE
from splipy import *

# Dimensional transform functions ----------------------------------------------------------------
def tsne(df_anim, ndim):
    """ Takes a df of animation, reduces the 17d to ndim dimensions """

    joints_values = df_anim.loc[:, joints_names].values
    X = np.array(joints_values)

    X_embedded = TSNE(n_components=ndim, perplexity=100.0, learning_rate=30, n_iter=50000, init='random', random_state=4).fit_transform(X)

    return pd.Series(X_embedded.reshape([100, ]))


#  Interpolation functions ---------------------------------------------------------
def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1.
        From dribnet/plat
    """
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val


def interpolate(z_p1, z_p2, n_points, method):
    """ Interpolate between the latent means of two postures.
        n_points indicates how many points will be sampled from the latent space
        The interpolation parameters are evenly spaced.
        Returns an array n_points x 3
    """
    if method == 'slerp':
        interp = np.array([slerp(t, z_p1, z_p2) for t in list(np.linspace(0, 1, n_points))])
    elif method == 'lerp':
        interp = np.array([lerp(t, z_p1, z_p2) for t in list(np.linspace(0, 1, n_points))])
    else:
        print('No valid interpolation method was given')
        interp = None

    return interp


def bspline(z_pos, steps=100):
    """ Input is a array of latent postures (each z posture as a list of 3D point - (l1,l2,l3))
        Returns a spline interpolation
    """

    n_pos = z_pos.shape[0]

    if n_pos >= 4:
        curve = curve_factory.cubic_curve(z_pos, boundary=2)
        t = np.linspace(curve.start(), curve.end(), steps * (n_pos-1))
        interp = curve(t)

        return interp

    else:
        print("Not enough frames. Use 2 frames for Bezier curve or >= 4 for interpolating B-spline")


def interp_multi(pos_list, latent, steps, check_model, check_epoch, method, feats_names, cond_AV=None):
    """ Given a list of two or more normalized postures, interpolate between them
        The latent interpolant is then decoded, and inverse normalized back to radians
    """
    # Restore model to get the decoder
    model = load_model(check_model, check_epoch)
    scaler = 'j_scaler_nao_lim_df32_25fps.pkl'

    # latent=True for interp the latent space directly without encoding keyframes before
    if latent:
        latent_mean = np.array(pos_list)
    else:
        df = pd.DataFrame(np.array(pos_list))
        latent_mean, latent_sigma = encode(df, model)

    interp_all = []

    if method == 'spline':
        interp = bspline(latent_mean, steps)
    else:
        for i in range(latent_mean.shape[0] -1):
            interp_i = interpolate(latent_mean[i, :], latent_mean[i + 1, :], steps, method)
            interp_all.append(interp_i)

        interp = np.concatenate(interp_all)

    # Get the def of the z interpolant
    df_z_interp = pd.DataFrame(interp)
    cols_list = []

    for c in range(df_z_interp.shape[1]):
        cols_list.append('l' + str(c+1))
    df_z_interp.columns = cols_list

    if cond_AV is None:
        # Get decoded denormalized latent interpolant
        dec_interp = decode(interp, model)
        df_dec_interp = pd.DataFrame(columns=feats_names, data=dec_interp)
        df_dec_interp_norm = inverse_norm(df_dec_interp, scaler)
    else:
        dec_interp = decode_cond(interp, np.tile(cond_AV, (interp.shape[0], 1)), model)
        if len(cond_AV) == 1:
            df_dec_interp = pd.DataFrame(columns=feats_names + ['valence'], data=dec_interp)
        else:
            df_dec_interp = pd.DataFrame(columns=feats_names + ['arousal', 'valence'], data=dec_interp)

        df_dec_interp_norm = df_dec_interp
        df_dec_interp_norm[joints_names] = inverse_norm(df_dec_interp.loc[:, joints_names], scaler)

    return df_dec_interp_norm, df_z_interp


# Animation/posture selection functions ----------------------------------------------
def sel_anim_id(df):
    """ Given a dataframe that contains an 'id' column, select and return an animation id.
        A prompt appears on the screen to select a category, then subcategory and subsubcategory.
    """
    df = df[~df.id.str.contains('_tr')]
    cat_list = df['category'].unique().tolist()
    cat = input("Select category: " + str(cat_list) + '\n')
    df_cat = df[df['category'] == cat]
    id_list = df_cat['id'].unique().tolist()
    id_list.sort()
    print("Select id:")
    anim_id = input('\n'.join(id_list))

    return anim_id, cat


def sel_rand_posture(df, n, label):
    """ Given a dataframe of animations it returns n entries of it, just joints values.
        The label can be 'random' (selects n postures from any animation), a VA_category of emotion
        if df is the Plymouth library (selects n postures from animations in this category),
        or "select" to display categories and animation tags for manually selecting one animation.
    """
    df_pos = pd.DataFrame()

    if label == 'random':
        length = len(df)
        idx = np.random.randint(0, length, size=n)
        df_pos = df.loc[idx, joints_names]

    elif label in all_categories:
        df_cat = df.loc[df['category'] == label, :]
        # df_cat.reset_index(drop=True, inplace=True)
        idx_cat = df_cat.index.values.tolist()
        length = len(idx_cat)
        idx_rand = np.random.randint(0, length, size=n)
        idx_list = []
        for i in idx_rand:
            idx_list.append(idx_cat[i])
        df_pos = df.loc[idx_list, joints_names]

    elif label == 'select':
        anim_id = sel_anim_id(df)
        df_cat = df.loc[df['id'] == anim_id, :]
        length = len(df_cat)
        idx = np.random.randint(df_cat['id'].index[0], df_cat['id'].index[0] + length - 1, size=n)
        df_pos = df.loc[idx, joints_names]

    else:
        print("The label was not found.")

    if ~df_pos.empty:
        return df_pos.sort_index()


def sel_pos_frame(df, frame):
    """ Given a dataframe of animations it returns a frame with joints as a list, and the animation id.
    """
    joints_list = df.loc[frame, joints_names].tolist()
    id_anim = df.loc[frame, 'id']

    return joints_list, id_anim


# tfmodel functions ------------------------------------------------------------------------
def load_model(check_model, check_epoch):
    """ Load the given checkpoint """
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
    k = None
    return model_restored


def encode(df, model):
    """ Given a df with frames (i.e. postures) and a model, encode them in the latent space.
        It returns their latent mean, latent sigma vectors (arrays)
    """
    if isinstance(df, pd.DataFrame):
        x = df.values
    elif isinstance(df, list):
        x = np.asarray(df).reshape([1, 17])
    latent_mean, latent_sigma = model.sess.run([model.latent_mean, model.latent_sigma],
                                               feed_dict={model.x_input: x, model.bn_is_training: False})
    return latent_mean, latent_sigma


def decode(latent_mean, model):
    """ Decodes latent z codes and returns reconstruction """
    x_reconstruction = model.sess.run([model.y_output],
                                      feed_dict={model.latent_layer: latent_mean, model.bn_is_training: False})
    return x_reconstruction[0]


def decode_cond(latent_mean, cond, model):
    """ Decodes latent z codes and returns reconstruction """
    x_reconstruction = model.sess.run([model.y_output],
                                      feed_dict={model.latent_layer: latent_mean, model.y_labels: cond, model.bn_is_training: False})
    return x_reconstruction[0]


def get_latent_z(check_model, check_epoch, dataset, cond=False):
    """ Load a model and a dataset of animations and save df with their latent means and sigmas"""
    model = load_model(check_model, check_epoch)
    # Load animation dataset
    df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, dataset), index_col=0)

    if cond:
        df_anim = merge_with_VA_labels(df_anim)

    # Get latent z vectors
    x = df_anim.drop(columns=['time', 'id', 'category'], inplace=False)
    latent_mean, latent_sigma = encode(x, model)

    # Get the best dimensions of the latent space
    latent_sigma_mean = latent_sigma.mean(axis=0)
    dim = latent_mean.shape[1]  # Number of dimensions in the latent space
    # dim_inds = np.argsort(latent_sigma_mean)  # Prioritize
    # latent_z_mean = latent_mean[:, dim_inds].transpose()
    # latent_z_sigma = latent_sigma[:, dim_inds].transpose()

    latent_z_mean = latent_mean.transpose()
    latent_z_sigma = latent_sigma.transpose()

    # Create column names for latent dimensions and put them in a dataframe
    l_dim_names = []
    df_z_mean = pd.DataFrame()
    df_z_sigma = pd.DataFrame()
    for d in range(dim):
        name = 'l' + str(d + 1)
        df_z_mean[name] = latent_z_mean[d, :]
        df_z_sigma[name] = latent_z_sigma[d, :]
        l_dim_names.append(name)

    df_z_mean['id'] = df_anim['id']
    df_z_mean['category'] = df_anim['category']
    df_z_sigma['id'] = df_anim['id']
    df_z_sigma['category'] = df_anim['category']

    dataset = dataset.split('.')[0]
    # Store the z_mean and z_sigma dfs
    df_z_mean.to_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, check_model + check_epoch + '_' + dataset + '_mean.csv'))
    df_z_sigma.to_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, check_model + check_epoch + '_' + dataset + '_sigma.csv'))


# Used once to correct discrepancies in datasets structures ------------------------------------------------------
def save_category(dataset):
    """ Retrospectively add a category column to the datasets """

    df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, dataset), index_col=0)

    # Depending on animation library, different handling of category feature
    if 'df2' in dataset:  # Naoqi animation library
        df_anim = add_category(df_anim)
    else:  # Plymouth animation library
        df_anim = merge_with_labels(df_anim)

    df_anim.to_csv(os.path.join(ROOT_PATH, DATA_X_PATH, dataset))


def differ_duplicate(dataset):
    """ In the Naoqi animation library some ids are the same. Add the category name as part of the id name
        to all categories except Emotions.
    """
    df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, dataset), index_col=0)
    # Unique ids
    ids = df_anim['id'].unique().tolist()

    dup_ids = []
    dup_cat = []
    for anim_id in ids:

        df_name = df_anim.loc[df_anim['id'] == anim_id, ['id', 'category']]
        df_name = df_name.drop_duplicates()
        if df_name.shape[0] > 1:
            dup_ids.append(df_name['id'].tolist())
            dup_cat.append(df_name['category'].tolist())

    # Flat lists
    dup_ids = [item for sublist in dup_ids for item in sublist]
    dup_cat = [item for sublist in dup_cat for item in sublist]

    for i in range(len(dup_cat)):
        if "Emotions" not in dup_cat[i]:
            new_id = dup_cat[i] + '/' + dup_ids[i]
            df_anim.loc[(df_anim['id'] == dup_ids[i]) & (df_anim['category'] == dup_cat[i]), 'id'] = new_id

    df_anim.to_csv(os.path.join(ROOT_PATH, DATA_X_PATH, dataset))


# Dataframe functions --------------------------------------------------------------
def merge_with_labels(df, lab_file='y_va_cat_aug.csv'):
    """Merge animations dataset with categorical labels"""
    df_y = pd.read_csv(os.path.join(ROOT_PATH, DATA_Y_PATH, lab_file), index_col=0)
    df_merged = pd.merge(df, df_y, left_on=['id'], right_on=['nameAnim'], how='left')
    df_merged.drop(columns=['nameAnim', 'valence', 'arousal'], inplace=True)
    return df_merged


def merge_with_VA_labels(df, lab_file='y_va_cat_aug.csv'):
    """Merge animations dataset with valence/arousal labels"""
    df_y = pd.read_csv(os.path.join(ROOT_PATH, DATA_Y_PATH, lab_file), index_col=0)
    df_merged = pd.merge(df, df_y, left_on=['id'], right_on=['nameAnim'], how='left')
    df_merged.drop(columns=['nameAnim', 'category_y'], inplace=True)
    df_merged.rename(columns={'category_x': 'category'}, inplace=True)
    return df_merged


def add_category(df):
    """ Only for the Naoqi animation library. Splits name into category and anim id columns"""
    id_df = df['id'].str.split('/', expand=True)
    id_df.drop(columns=[0, 1], inplace=True)  # Drop animations/Stand

    idx_no_emo = id_df[4].index[id_df[4].isnull()].values.tolist()
    df.loc[idx_no_emo, 'category'] = id_df.loc[idx_no_emo, 2]
    df.loc[idx_no_emo, 'id'] = id_df.loc[idx_no_emo, 3]

    idx_emo = id_df[4].index[~id_df[4].isnull()].values.tolist()
    df.loc[idx_emo, 'category'] = id_df.loc[idx_emo, 2] + '/' + id_df.loc[idx_emo, 3]
    df.loc[idx_emo, 'id'] = id_df.loc[idx_emo, 4]

    return df


def v2cat_df(df):
    """
    Takes the y labels, which contain valence continuous values per animation, and
    assigns them with 0, 0.5, 1 which is neg, neu, pos respectively
    """

    # Add category based on the v/a score
    df.loc[(df['valence'] > 0.66), 'category'] = 1
    df.loc[(df['valence'] > 0.33) & (df['valence'] <= 0.66), 'category'] = 0.5
    df.loc[(df['valence'] <= 0.33), 'category'] = 0

    df['valence'] = df['category']
    return df


def v2cat_value(v):
    if v > 0.66:
        cat = 'Pos'
    elif v <= 0.33:
        cat = 'Neg'
    else:
        cat = 'Neu'
    return cat


def downsample_anim(df, fps):
    df_new = pd.DataFrame(columns=df.columns)
    df_new = pd.concat([df_new, df[::fps]], ignore_index=True)

    return df_new


# TODO: Normalize Input can be list array or df
def normalize(list_pos):
    scaler_pkl = 'j_scaler_nao_lim_df13_50fps.pkl'
    path = os.path.join(ROOT_PATH, SCALERS_PATH, scaler_pkl)
    scaler = joblib.load(path)
    if isinstance(list_pos, list):
        pos_norm_list = scaler.transform(np.array(list_pos).reshape([1, n_joints]))
    else:
        pos_norm_list = scaler.transform(list_pos.loc[:, joints_names])
    pos_norm_df = pd.DataFrame(columns=joints_names, data=pos_norm_list)

    return pos_norm_list.tolist()[0]


if __name__ == '__main__':
    dataset = 'df32_25fps.csv'
    get_latent_z('54', '-500', dataset, cond=True)
