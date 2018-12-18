# %%
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import numpy as np
import pandas as pd
from plotnine import *
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from scipy.stats import pearsonr
from minepy import MINE
# % matplotlib tk
pd.options.mode.chained_assignment = None

# %%

def mic(x,y):
  m = MINE()
  m.compute_score(x,y)
  print(m.mic())
  return (m.mic(), 0.5)

def pear_pa(x, y):
  ret_temp = map(lambda x: pearsonr(x, y), x.T)
  return [i[0] for i in ret_temp], [i[1] for i in ret_temp]


def mics(x, y):
  temp = map(lambda x: mic(x,y), x.T)
  return [i[0] for i in temp], [i[1] for i in temp]

# %%
raw_dtrain = pd.read_csv('data/train.csv')
raw_dtest = pd.read_csv('data/test.csv')

raw_dtrain.info()
# %% fill missing value

data = raw_dtrain.loc[np.invert(pd.isnull(raw_dtrain['LotFrontage']))]
data['MSSubClass'] = data['MSSubClass'].apply(str)
# %%

(ggplot(data)
  # + geom_point(aes(x='MasVnrType', y='LotArea'))
  + geom_bar(aes(x='MasVnrType'))
)
# %%

(ggplot(data)
  # + geom_point(aes(x='MasVnrType', y='LotArea'))
  + geom_bar(aes(x='BsmtQual'))
)

# %%
# TODO: Alley

# data['LotFrontage'].groupby(data['Neighborhood']).median()
raw_dtrain['LotFrontage'].groupby(raw_dtrain['Neighborhood']).transform(lambda x: x.fillna(x.median(), inplace=True) )
raw_dtrain['MasVnrType'].fillna('None', inplace=True)
raw_dtrain['MasVnrArea'].fillna(0, inplace=True)
raw_dtrain['BsmtQual'].fillna('NA', inplace=True)

raw_dtrain.info()

# %%

raw_dtrain[pd.isnull(raw_dtrain['BsmtQual'])]

# %%


# %%

np.count_nonzero(raw_dtrain['LotFrontage'] > 0)
np.count_nonzero( pd.isnull(raw_dtrain['LotFrontage']))




