import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from settings import *
import pandas as pd
import os
import seaborn as sns
sns.set(style="whitegrid")



def set_pub():
    plt.rc('font', family='sans-serif')
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def z_anim2dec(df_z_mean, interp, output_df, anim_id):
    """ For an animation, plot the encoded (dotted) + interpolated latent space and the decoded output"""

    df_z_anim = df_z_mean.loc[df_z_mean['id'] == anim_id, :]
    df_z_anim.drop(columns=['id', 'category'], inplace=True)

    # GRAPH: the interpolation in the latent space
    set_pub()
    fig = plt.figure(figsize=(30, 18))

    ax1 = fig.add_subplot(2, 1, 1)

    ax1.plot(interp, label=[])
    ax1.set_prop_cycle(None)
    ax1.plot(df_z_anim.values, linestyle=':', label=[])
    # df_z_anim.plot(y=df_z_anim.columns.tolist()[:-2], linestyle='-.', ax=ax1)
    ax1.set_title('Latent space animation: original and interpolation')
    ax1.set_ylabel('Joints values')
    handles, labels = ax1.get_legend_handles_labels()
    labels = df_z_anim.columns.tolist()
    lgd = ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(0.96, 0.5))

    ax2 = fig.add_subplot(2, 1, 2)
    output_df.plot(kind='line', y=joints_names, ax=ax2)
    ax2.set_title('Decoded animation based on interpolation')
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Joints values')


    ax2.get_legend().remove()
    ax2.legend(loc='center left', bbox_to_anchor=(0.96, 0.5))
    # plt.show()

    return fig


def x2reco(df_x, df_reco):
    """ Plot an original animation across time frames and its reconstruction """

    set_pub()

    head_names = ['HeadPitch', 'HeadYaw', 'HipPitch', 'HipRoll', 'KneePitch']
    r_arms_names = ['RElbowRoll', 'RElbowYaw', 'RHand', 'RShoulderPitch', 'RShoulderRoll', 'RWristYaw']
    l_arms_names = ['LElbowRoll', 'LElbowYaw', 'LHand', 'LShoulderPitch', 'LShoulderRoll', 'LWristYaw']

    fig, ax = plt.subplots(3, sharex=True, figsize=(30, 18))

    # RECO: Head and Torso
    df_reco.plot(y=head_names, kind='line', linestyle='-.', ax=ax[0], legend=False)
    ax[0].set_prop_cycle(None)
    # ORIG
    df_x.plot(y=head_names, kind='line', ax=ax[0])

    handles, labels = ax[0].get_legend_handles_labels()
    lgd = ax[0].legend(handles[5:], labels[5:], loc='center left', bbox_to_anchor=(0.96, 0.5))
    ax[0].set_title('Head and torso')

    # RECO: R Arm
    df_reco.plot(y=r_arms_names, kind='line', linestyle='-.', ax=ax[1], legend=False)
    ax[1].set_prop_cycle(None)
    # ORIG
    df_x.plot(y=r_arms_names, kind='line', ax=ax[1])

    handles, labels = ax[1].get_legend_handles_labels()
    lgd = ax[1].legend(handles[6:], labels[6:], loc='center left', bbox_to_anchor=(0.96, 0.5))
    ax[1].set_title('Right arm')
    ax[1].set_ylabel('Joints values')

    # RECO: L Arm
    df_reco.plot(y=l_arms_names, kind='line', linestyle='-.', ax=ax[2], legend=False)
    ax[2].set_prop_cycle(None)
    # ORIG
    df_x.plot(y=l_arms_names, kind='line', ax=ax[2])

    handles, labels = ax[2].get_legend_handles_labels()
    lgd = ax[2].legend(handles[6:], labels[6:], loc='center left', bbox_to_anchor=(0.96, 0.5))
    ax[2].set_title('Left arm')
    ax[2].set_xlabel('Frames')
    # plt.show()

    return fig


def x_all2z(df_z, colorcode='id', leg=True):
    """ All the encoded animations in the latent space (color-coded per animation)"""
    set_pub()
    fig = plt.figure()

    dim = df_z.shape[1] - 2  # This is -2 for df with both category and id, just id then becomes -1. horrible, i know
    if dim <= 2:
        ax = plt.axes()
        sns.scatterplot(x='l1', y='l2', hue=colorcode, data=df_z, legend=False)
    else:
        ax = plt.axes(projection='3d')

        if colorcode == 'id':
            ids = df_z['id'].unique().tolist()
            ids = [x for x in ids if '_tr' not in x]
        else:
            ids = all_categories

        for colc in ids:
            df = df_z.loc[df_z[colorcode] == colc, :]
            ax.scatter(df['l1'], df['l2'], df['l3'], label=colc,  s=10)

            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
            ax.set_zlabel('LD3')

    # ax.set_title('Animations encoded in the latent space')
    ax.axis('equal')
    ax.axis('square')
    if leg:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(0.96, 0.5))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    return fig


def plot_3D_curve(curve, show_controlpoints=False):
    t = np.linspace(curve.start(), curve.end(), 150)
    x = curve(t)
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:,0], x[:,1], x[:,2])
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    if show_controlpoints:
        plt.plot(curve[:, 0], curve[:, 1], curve[:,2],'rs-')
    return fig


def plot_z_VA():
    z = '42-200_df14_20fps_mean.csv'
    df_z = pd.read_csv(os.path.join(ROOT_PATH, DATA_LATE, z), index_col=0)
    labels = pd.read_csv(os.path.join(ROOT_PATH, DATA_Y_PATH, 'y_va_cat_aug.csv'), index_col=0)

    df_all = pd.merge(df_z, labels, left_on="id", right_on="nameAnim", how='left')
    df_all = df_all[~df_all.id.str.contains('_tr')]

    df_z_V = df_all.loc[:, ['l1', 'l2', 'l3', 'valence_mean']]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(df_z_V.loc[:, 'l1'], df_z_V.loc[:, 'l2'], df_z_V.loc[:, 'l3'], c=df_z_V.loc[:, 'valence_mean'], s=df_z_V.loc[:, 'valence_mean']*25, marker='o')
    fig.colorbar(p)


    df_z_A = df_all.loc[:, ['l1', 'l2', 'l3', 'arousal_mean']]
    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    p2 = ax2.scatter(df_z_A.loc[:, 'l1'], df_z_A.loc[:, 'l2'], df_z_A.loc[:, 'l3'], c=df_z_A.loc[:, 'arousal_mean'], s=df_z_A.loc[:, 'arousal_mean']*25, marker='o')
    fig2.colorbar(p2)
    plt.show()

if __name__ == '__main__':

    plot_z_VA()