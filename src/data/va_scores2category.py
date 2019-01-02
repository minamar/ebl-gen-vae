import pandas as pd
import os
from settings import ROOT_PATH, DATA_Y_PATH

"""
Takes the y labels, which contains valence and arousal continuous values per animation, and 
and assigns them in 9 different categories 
"""

y_set = 'y_va.csv'
dest_y_set = 'y_va_cat.csv'

# Read the animation valence and arousal labels
path = os.path.join(ROOT_PATH, DATA_Y_PATH, y_set)
y_va = pd.read_csv(path)

# Add category based on the v/a score
y_va.loc[y_va.index[(y_va['valence_mean']> 0.66) & (y_va['arousal_mean']> 0.66)].tolist(),'Category'] = 'Pos/Exc'
y_va.loc[y_va.index[(y_va['valence_mean']> 0.66) & (y_va['arousal_mean']> 0.33) & (y_va['arousal_mean']<=0.66)].tolist(),'Category'] = 'Pos/Cal'
y_va.loc[y_va.index[(y_va['valence_mean']> 0.66) & (y_va['arousal_mean']<= 0.33)].tolist(),'Category'] = 'Pos/Tir'
y_va.loc[y_va.index[(y_va['valence_mean']> 0.33) & (y_va['valence_mean']<= 0.66) & (y_va['arousal_mean']> 0.66)].tolist(),'Category'] = 'Neu/Exc'
y_va.loc[y_va.index[(y_va['valence_mean']> 0.33) & (y_va['valence_mean']<= 0.66) & (y_va['arousal_mean']> 0.33) & (y_va['arousal_mean']<=0.66)].tolist(),'Category'] = 'Neu/Cal'
y_va.loc[y_va.index[(y_va['valence_mean']> 0.33) & (y_va['valence_mean']<= 0.66) & (y_va['arousal_mean']<= 0.33)].tolist(),'Category'] = 'Neu/Tir'
y_va.loc[y_va.index[(y_va['valence_mean']<= 0.33) & (y_va['arousal_mean']> 0.66)].tolist(),'Category'] = 'Neg/Exc'
y_va.loc[y_va.index[(y_va['valence_mean']<= 0.33) & (y_va['arousal_mean']> 0.33) & (y_va['arousal_mean']<=0.66)].tolist(),'Category'] = 'Neg/Cal'
y_va.loc[y_va.index[(y_va['valence_mean']<= 0.33) & (y_va['arousal_mean']<= 0.33)].tolist(),'Category'] = 'Neg/Tir'

y_va_cat = y_va

# Save the y set with categorical labels of valence and arousal
dest = os.path.join(ROOT_PATH, DATA_Y_PATH, y_set)
y_va_cat.to_csv(dest)

