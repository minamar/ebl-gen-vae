from settings import *
from sklearn.externals import joblib
import os


# Only joints input. Returns True if there are values exceeding the limits
def check_limits(df):
    flag = False
    minmax = np.array(joints_minmax)
    min_j = minmax[:, 0]
    max_j = minmax[:, 1]
    if any(np.array(df.min().values < min_j)) | any(np.array(df.max().values > max_j)):
        flag = True
        print("***** There are values exceeding the naoqi limits")
    return flag


# Full df input. Replaces the values that exceed the limits with the min/max value
def correct_limits(df):
    j_count = 0
    for joint in joints_names:

        idx_min = df[joint].loc[df[joint] < 0].index
        df.loc[idx_min, joint] = 0

        idx_max = df[joint].loc[df[joint] > 1].index
        df.loc[idx_max, joint] = 1
        j_count += 1

    return df


# Full df input. Replaces time_diff with time
def inverse_time(df):
    time_s = abs(df.loc[:, 'time_diff'])
    # Time dimension transformation: from timestamp to time lags between subsequent timestamps
    time_l = []

    for i, v in time_s[1:].iteritems():
        t = time_s[:i+1].sum()
        time_l.append(t)

    df['time_diff'] = [0] + time_l

    return df


# Invert the normalized values using the rescaler
def inverse_norm(df, scaler_pkl):
    path = os.path.join(ROOT_PATH, SCALERS_PATH, scaler_pkl)
    scaler = joblib.load(path)
    inverse = scaler.inverse_transform(df.loc[:, joints_names])
    df.loc[:, joints_names] = inverse

    return df
