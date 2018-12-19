# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# import collections
from plotnine import *
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from scipy.stats import pearsonr
from minepy import MINE
# % matplotlib tk
pd.options.mode.chained_assignment = None
np.set_printoptions(precision=5, suppress=True)

# %%
import socket; socket.gethostname()

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

raw_dtrain = pd.read_csv('data/train.csv')
raw_dtest = pd.read_csv('data/test.csv')

# %% fill missing values

# recover 'NA' type
none_type_word = "None"

raw_dtrain['Alley'].fillna(none_type_word, inplace=True)
raw_dtrain['FireplaceQu'].fillna(none_type_word, inplace=True)
raw_dtrain['PoolQC'].fillna(none_type_word, inplace=True)
raw_dtrain['BsmtQual'].fillna(none_type_word, inplace=True)
raw_dtrain['MiscFeature'].fillna(none_type_word, inplace=True)
raw_dtrain['Fence'].fillna(none_type_word, inplace=True)
raw_dtrain['MiscFeature'].fillna(none_type_word, inplace=True)

raw_dtrain['GarageFinish'].fillna(none_type_word, inplace=True)
raw_dtrain['GarageType'].fillna(none_type_word, inplace=True)
raw_dtrain['GarageQual'].fillna(none_type_word, inplace=True)
raw_dtrain['GarageCond'].fillna(none_type_word, inplace=True)
raw_dtrain['GarageType'].fillna(none_type_word, inplace=True)
raw_dtrain['GarageYrBlt'].fillna(-1, inplace=True)

raw_dtrain['BsmtCond'].fillna(none_type_word, inplace=True)
raw_dtrain['BsmtQual'].fillna(none_type_word, inplace=True)
raw_dtrain['BsmtExposure'].fillna('No', inplace=True)
raw_dtrain['BsmtFinType1'].fillna('Unf', inplace=True)
raw_dtrain['BsmtFinType2'].fillna('Unf', inplace=True)

raw_dtrain['MasVnrType'].fillna('None', inplace=True)
raw_dtrain['MasVnrArea'].fillna(0, inplace=True)
raw_dtrain['Electrical'].fillna('Sbrkr', inplace=True)

avg_data = raw_dtrain['LotFrontage'].groupby(raw_dtrain['Neighborhood']).median()
null_lot = raw_dtrain['LotFrontage'].isnull()
raw_dtrain['LotFrontage'][null_lot] = raw_dtrain['Neighborhood'][null_lot].map(lambda x: avg_data[x])


# %% new features

# how many years after remod when sold
RemodAge = raw_dtrain['YrSold'] - raw_dtrain['YearRemodAdd']
# years after built when sold
HouseAge = raw_dtrain['YrSold'] - raw_dtrain['YearBuilt']
# how old house is
Oldness = HouseAge*0.5 + RemodAge
# how many years garage built
def getGarageAge():
  res = []
  for i in range(raw_dtrain.shape[0]):
    sold = raw_dtrain['YrSold'][i]
    garage = raw_dtrain['GarageYrBlt'][i]
    remod = raw_dtrain['YearRemodAdd'][i]
    if garage == -1:
      res.append(-1)
    else:
      res.append(sold - max(garage, remod))
  return np.array(res)

GarageAge = getGarageAge()


raw_dtrain['RemodAge'] = pd.Series(RemodAge)
raw_dtrain['HouseAge'] = pd.Series(HouseAge)
raw_dtrain['Oldness'] = pd.Series(Oldness)
raw_dtrain['GarageAge'] = pd.Series(GarageAge)
# %% drop features
raw_dtrain.drop(['CentralAir', 'PoolQC', 'Condition2', 'RoofMatl', 'Street', 'Utilities', 'MiscFeature'], inplace=True)


# %% label -> number

# abc = pd.DataFrame({'x': ['a','b','c','c','a']})


mapping = {
# num -> label
# 'MSSubClass': {20:'20',30:'30',40:'40',45:'45',50:'50',60:'60',70:'70',75:'75',80:'80',85:'85',90:'90',120:'120',150:'150',160:'160',180:'180',190:'190'},

# label -> num
'LotShape': {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3},
'GarageQual': {'Ex': }

# multi label -> binary
'LandSlope': {'Gtl':0, 'Mod':1, 'Sev':1},
'Heating': {'GasA':0, 'Floor':1, 'GasW':1, 'Grav':1, 'OthW':1, 'Wall':1},
'Alley': {'Grvl':1, 'Pave':1, 'None':0},
'Electrical': {'SBrkr':0, 'FuseA':1, 'FuseF':1, 'FuseP':1, 'Mix':1},

# grouping
'OverallQual': {1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:2, 8:2, 9:2, 10:2},
'RoofStyle': {'Gable': 0, 'Flat':2, 'Gambrel':2, 'Hip': 1, 'Mansard':2, 'Shed':2},
'LotShape': {'Reg':0, 'IR1':1, 'IR2':2, 'IR3':2},
'MSZoning': {'RL':0, 'RM':1, 'FV':2, 'A':3, 'C':3, 'I':3, 'RH':3, 'RP':3},
'Condition1': {'Norm': 0, 'Feedr':1, 'Artery':2, 'PosA':3, 'PosN':3, 'RRAe':3, 'RRAn':3, 'RRNe':3, 'RRNn':3},

}

# abc['x'] = abc['x'].transform(lambda x: mapping['b'][x])


# %% One hot encoding
onehot_fields = ['MSSubClass','MSZoning','LotShape','Neighborhood','Condition1','BldgType','HouseType','']

pd.get_dummies(raw_dtrain[['SaleCondition', 'SaleType']])


# %%

test = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
test2 = pd.DataFrame({'c': ['a7','a8','a9'] })

xx = pd.get_dummies(test['a'])
xx[1]

# %%
for i in xx.columns:
  test[i] = xx[i]
# %%  
test

# %%









