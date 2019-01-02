import matplotlib.pyplot as plt
from settings import *
import seaborn as sns
sns.set(style="darkgrid")


def z_anim2dec(df_z_mean, interp, output_df, anim_id, category):
    """ For an animation, plot the encoded + interpolated (dotted) latent space and the decoded output"""
    df_z_anim = df_z_mean.loc[df_z_mean['id'] == anim_id, :]
    df_z_anim.drop(columns=['id', 'category'], inplace=True)

    # GRAPH: the interpolation in the la
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(2, 1, 1)

    ax1.plot(interp, linestyle=':', label='interp')
    ax1.plot(df_z_anim.values, label='original')
    # df_z_anim.plot(kind='line', ax=ax1)
    ax1.set_title('latent_space animation')

    ax2 = fig.add_subplot(2, 1, 2)
    # ax2.plot(output_df)
    output_df.plot(kind='line', y=joints_names, ax=ax2)
    ax2.set_title('decoded animation based on interpolation')

    fig.suptitle('Animation: ' + anim_id + ' in ' + category, fontsize=16)

    ax2.get_legend().remove()
    ax2.legend(loc='center left', bbox_to_anchor=(0.96, 0.5))
    plt.show()

    return fig
