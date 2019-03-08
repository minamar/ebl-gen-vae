import numpy as np
from globvar import ROOT_PATH

DATA_Y_PATH = 'data/processed/labels/'
DATA_X_PATH = 'data/processed/sampling/'
DATA_Z_PATH = 'data/generated/latent_z'

DATA_SAMP = 'data/generated/sampled'
DATA_RECO = 'data/generated/reconstruction'
DATA_VISU = 'data/visualizations'
DATA_LATE = 'data/generated/latent_z'

ANIM_DIR = 'data/external/plymouth-animations_3'
RAW_DATA = 'data/raw'
SCALERS_PATH = 'data/interim/scalers'


# A list with the 8 categories (no Pos/Tir category):
all_categories = ['Pos/Exc', 'Neu/Cal', 'Neu/Tir', 'Neu/Exc', 'Neg/Cal', 'Neg/Tir', 'Neg/Exc', 'Pos/Cal']

# Constants from the dataset
n_joints = 17  # 17 joints
n_categories = 8

# StandInit posture without the time dimension (radians)
standInit = [-0.2, 0.0, -0.0399996, -0.0, -0.0099995, -0.5225337, -1.2282506, 0.6, 1.5596846, 0.1427103, -0.0004972,
             0.5225337, 1.2282506, 0.6, 1.5596846, -0.1427103, 0.0004972]

# Normalized standInit posture with *naoqi* limits (radians)
standInit_norm = [0.37715381, 0.5, 0.48074111, 0.5, 0.49028934, 0.66922499, 0.20554991, 0.60416665, 0.87390519,
                  0.08625503, 0.4998637, 0.33077501, 0.79445009, 0.60416665, 0.87390519, 0.91374497, 0.5001363]

joints_names = ['HeadPitch', 'HeadYaw', 'HipPitch', 'HipRoll', 'KneePitch', 'LElbowRoll', 'LElbowYaw', 'LHand',
                'LShoulderPitch', 'LShoulderRoll', 'LWristYaw', 'RElbowRoll', 'RElbowYaw', 'RHand', 'RShoulderPitch',
                'RShoulderRoll', 'RWristYaw']

# Dataframe columns for the output of the sampling
out_cols = joints_names + ['time']


# Min-max ranges of joints (radians)
joints_minmax = np.array([[-0.7068583369255066, 0.6370452046394348],  # 'HeadPitch' 0
                          [-2.0856685638427734, 2.0856685638427734],  # 'HeadYaw'  0
                          [-1.0384708642959595, 1.0384708642959595],  # 'HipPitch'  0
                          [-0.514872133731842, 0.514872133731842],  # 'HipRoll'  0
                          [-0.514872133731842, 0.514872133731842],  # 'KneePitch'  0
                          [-1.5620696544647217, -0.008726646192371845],  # 'LElbowRoll'  -1
                          [-2.0856685638427734, 2.0856685638427734],  # 'LElbowYaw' 0
                          [0.019999999552965164, 0.9800000190734863],  # 'LHand    0.1
                          [-2.0856685638427734, 2.0856685638427734],  # 'LShoulderPitch' 0
                          [0.008726646192371845, 1.5620696544647217],  # 'LShoulderRoll'  1
                          [-1.8238691091537476, 1.8238691091537476],  # 'LWristYaw'     0
                          [0.008726646192371845, 1.5620696544647217],  # 'RElbowRoll'   1
                          [-2.0856685638427734, 2.0856685638427734],  # 'RElbowYaw'   0
                          [0.019999999552965164, 0.9800000190734863],  # 'RHand'    0.5
                          [-2.0856685638427734, 2.0856685638427734],  # 'RShoulderPitch'   0
                          [-1.5620696544647217, -0.008726646192371845],  # 'RShoulderRoll'  -1
                          [-1.8238691091537476, 1.8238691091537476]])  # 'RWristYaw'   0

# Timestamps diffferences cannot be smaller than these limits:
# Calculated as abs(max-min)/maxVelocity per joint (radians)
maxVelocity_lims = {'LShoulderPitch': 0.5683034521145831, 'LShoulderRoll': 0.1683374199825035,
                    'LHand': 0.07643312001973787, 'LElbowYaw': 0.5683034521145831, 'LWristYaw': 0.20983935340878543,
                    'KneePitch': 0.3511173243852647, 'HipRoll': 0.45356705168743694,
                    'RShoulderRoll': 0.1683374199825035, 'RWristYaw': 0.20983935340878543,
                    'HeadYaw': 0.5683034521145831, 'RShoulderPitch': 0.5683034521145831,
                    'RElbowYaw': 0.5683034521145831, 'RElbowRoll': 0.1683374199825035, 'HeadPitch': 0.14564024409779705,
                    'RHand': 0.07643312001973787, 'LElbowRoll': 0.1683374199825035, 'HipPitch': 0.708185756103779}

maxVelocity = maxVelocity_lims['HipPitch']

# Used for the generated animation in sampling.py as the first timestamp: the max maxVelocity_lims
start_time = [maxVelocity + 0.05]


# # Get maxVelocity limits
# for joint in joints_names:
#     min_angle = motion._getFullLimits(joint)[0][0]
#     max_angle = motion._getFullLimits(joint)[0][1]
#     maxVelocity = motion._getFullLimits(joint)[0][2]
#
#     maxVelocity_lims[joint] = abs(max_angle - min_angle) / maxVelocity
