import matplotlib.pyplot as plt
from settings import *
import seaborn as sns
sns.set(style="darkgrid")
# sns.set_palette(sns.color_palette("Paired", 12))


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


def z_anim2dec(df_z_mean, interp, output_df, anim_id, category):
    """ For an animation, plot the encoded + interpolated (dotted) latent space and the decoded output"""

    df_z_anim = df_z_mean.loc[df_z_mean['id'] == anim_id, :]
    df_z_anim.drop(columns=['id', 'category'], inplace=True)

    # GRAPH: the interpolation in the latent space
    set_pub()
    fig = plt.figure(figsize=(30, 18))

    ax1 = fig.add_subplot(2, 1, 1)

    ax1.plot(interp, linestyle=':', label='interp')
    ax1.plot(df_z_anim.values, label='original')
    # df_z_anim.plot(kind='line', ax=ax1)
    ax1.set_title('latent_space animation', fontsize=20)

    ax2 = fig.add_subplot(2, 1, 2)
    # ax2.plot(output_df)
    output_df.plot(kind='line', y=joints_names, ax=ax2)
    ax2.set_title('decoded animation based on interpolation')
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Joints values')
    fig.suptitle('Animation: ' + anim_id + ', Category: ' + category, fontsize=20)

    ax2.get_legend().remove()
    ax2.legend(loc='center left', bbox_to_anchor=(0.96, 0.5))
    # plt.tight_layout(pad=5)
    # plt.show()

    return fig


def x2reco(df_x, df_z, df_reco):
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
