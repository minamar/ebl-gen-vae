import pandas as pd
import os
from settings import *
from tfmodellib.vae import VAE, VAEConfig


# Dataframe functions --------------------------------------------------------------
def merge_with_labels(df, lab_file='y_va_cat_aug.csv'):
    """Merge animations dataset with categorical labels"""
    df_y = pd.read_csv(os.path.join(ROOT_PATH, DATA_Y_PATH, lab_file), index_col=0)
    df_merged = pd.merge(df, df_y, left_on=['id'], right_on=['nameAnim'], how='left')
    df_merged.drop(columns=['nameAnim', 'valence_mean', 'arousal_mean'], inplace=True)
    return df_merged


def add_category(df):
    """ Only for the big animation library. Splits name into category and anim id columns"""
    id_df = df['id'].str.split('/', expand=True)
    id_df.drop(columns=[0, 1], inplace=True)  # Drop animations/Stand

    idx_no_emo = id_df[4].index[id_df[4].isnull()].values.tolist()
    df.loc[idx_no_emo, 'category'] = id_df.loc[idx_no_emo, 2]
    df.loc[idx_no_emo, 'id'] = id_df.loc[idx_no_emo, 3]

    idx_emo = id_df[4].index[~id_df[4].isnull()].values.tolist()
    df.loc[idx_emo, 'category'] = id_df.loc[idx_emo, 2] + '/' + id_df.loc[idx_emo, 3]
    df.loc[idx_emo, 'id'] = id_df.loc[idx_emo, 4]

    return df


def downsample_anim(df, fps):
    df_new = pd.DataFrame(columns=df.columns)
    df_new = pd.concat([df_new, df[::fps]], ignore_index=True)

    return df_new


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


def interpolate(z_p1, z_p2, n_points):
    """ Interpolate between the latent means of two postures.
        n_points indicates how many points will be sampled from the latent space
    """
    int_val = 1 / n_points
    interp = np.array([slerp(t, z_p1, z_p2) for t in np.arange(0.0, 1.0, int_val)])
    return interp


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

    # import os
    # import sys
    # import re
    # sys.path.append(os.path.realpath('.'))
    # from pprint import pprint
    #
    # import inquirer
    #
    # df = df[~df.id.str.contains('_tr')]
    # cat_list = df['category'].unique().tolist()
    # questions1 = [
    #     inquirer.List('size',
    #                   message="Select category: ",
    #                   choices=cat_list,
    #                   ),
    # ]
    # cat = inquirer.prompt(questions1)
    #
    # df_cat = df[df['category'] == cat]
    # id_list = df_cat['id'].unique().tolist()
    # id_list.sort()
    # print("Select id:")
    # anim_id = input('\n'.join(id_list))
    #
    # return anim_id, cat


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

# def sel_z_posture(df, steps, anim_id):
#     # df = df.loc[df['id'] == anim_id, :]
#     idx_anim = df.index.tolist()
#     idx_pos = idx_anim[0::steps]
#     # idx = np.round(np.linspace(idx_anim[0], idx_anim[-1], steps))
#     # idx_pos = idx.astype(int).tolist()
#     df_pos = df.loc[idx_pos, :]
#     df_pos = df_pos.iloc[:, :-2]
#     return df_pos


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
    return model_restored


def encode(df, model):
    """ Given a df with frames (i.e. postures) and a model, encode them in the latent space.
        It returns their latent mean, latent sigma vectors
    """
    x = df.values
    latent_mean, latent_sigma = model.sess.run([model.latent_mean, model.latent_sigma],
                                               feed_dict={model.x_input: x, model.bn_is_training: False})
    return latent_mean, latent_sigma


def decode(latent_mean, model):
    """ Decodes latent z codes and returns reconstruction """
    x_reconstruction = model.sess.run([model.y_output],
                                      feed_dict={model.latent_layer: latent_mean, model.bn_is_training: False})
    return x_reconstruction[0]


def get_latent_z(check_model, check_epoch, dataset):
    """ Load a model and a dataset of animations and save df with their latent means and sigmas"""
    model = load_model(check_model, check_epoch)
    # Load animation dataset
    df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, dataset), index_col=0)

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
    # Load animation dataset
    df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, dataset), index_col=0)

    # Depending on animation library, different handling of category feature
    if 'df2' in dataset:  # Naoqi animation library
        df_anim = add_category(df_anim)
    else:  # Plymouth animation library
        df_anim = merge_with_labels(df_anim)

    df_anim.to_csv(os.path.join(ROOT_PATH, DATA_X_PATH, dataset))


def differ_duplicate(dataset):
    """ For Big Library. Some ids are the same. Add the category name as part of the id name to all categories
        except Emotions
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

#
# if __name__ == '__main__':
#     dataset = 'df13_50fps.csv'
# # #     differ_duplicate(dataset)
# #     for r in ['3', '4', '5']:
#     get_latent_z('6', '-300', dataset)
