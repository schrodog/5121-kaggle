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

# def mic(x,y):
#   m = MINE()
#   m.compute_score(x,y)
#   print(m.mic())
#   return (m.mic(), 0.5)

# def pear_pa(x, y):
#   ret_temp = map(lambda x: pearsonr(x, y), x.T)
#   return [i[0] for i in ret_temp], [i[1] for i in ret_temp]


# def mics(x, y):
#   temp = map(lambda x: mic(x,y), x.T)
#   return [i[0] for i in temp], [i[1] for i in temp]

raw_dtrain = pd.read_csv('data/train.csv')
raw_dtest = pd.read_csv('data/test.csv')

# fill missing values

# recover 'NA' type
none_type_word = "NA"

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

# drop features
raw_dtrain.drop(['CentralAir', 'PoolQC', 'Condition2', 'RoofMatl', 'Street', 'Utilities', 'MiscFeature'], inplace=True, axis=1)

# label mapping

transformation_matrix = {
# label -> num (ordered)
'ExterQual': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
'ExterCond': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
'BsmtQual': {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
'BsmtCond': {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
'BsmtExposure': {'NA':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4},
'BsmtFinType1': {'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},
'BsmtFinType2': {'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},
'HeatingQC': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
'KitchenQual': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
'Functional': {'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7},
'FireplaceQu': {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
'GarageFinish': {'NA':0, 'Unf':1, 'RFn':2, 'Fin':3},
'GarageQual': {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
'GarageCond': {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
'PavedDrive': {'N':0, 'P':1, 'Y':2},
'Fence': {'NA':0, 'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4},

# multi label -> binary
'LandSlope': {'Gtl':0, 'Mod':1, 'Sev':1},
'Heating': {'GasA':0, 'Floor':1, 'GasW':1, 'Grav':1, 'OthW':1, 'Wall':1},
'Alley': {'Grvl':1, 'Pave':1, 'NA':0},
'Electrical': {'SBrkr':0, 'FuseA':1, 'FuseF':1, 'FuseP':1, 'Mix':1}, #Sbrkr error

# grouping
'LotShape': {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 2},
'RoofStyle': {'Gable': 0, 'Flat':2, 'Gambrel':2, 'Hip': 1, 'Mansard':2, 'Shed':2},
'LotShape': {'Reg':0, 'IR1':1, 'IR2':2, 'IR3':2},
'MSZoning': {'RL':0, 'RM':1, 'FV':2, 'A':3, 'C':3, 'I':3, 'RH':3, 'RP':3},
'Condition1': {'Norm': 0, 'Feedr':1, 'Artery':2, 'PosA':3, 'PosN':3, 'RRAe':3, 'RRAn':3, 'RRNe':3, 'RRNn':3},
'Neighborhood': {'Blmngtn': 2,'Blueste': 3,'BrDale': 2,'BrkSide': 0,'ClearCr': 1,'CollgCr': 1,'Crawfor': 0,'Edwards': 1,'Gilbert': 0,'GrnHill': 3,'IDOTRR': 0,'Landmrk': 2,'MeadowV': 4,'Mitchel': 0,'NAmes': 0,'NPkVill': 2,'NWAmes': 2,'NoRidge': 2,'NridgHt': 2,'OldTown': 0,'SWISU': 3,'Sawyer': 1,'SawyerW': 1,'Somerst': 2,'StoneBr': 2,'Timber': 3,'Veenker': 2}
}
# correct error
raw_dtrain['Electrical'] = raw_dtrain['Electrical'].transform(lambda x: 'SBrkr' if x=='Sbrkr' else x )
raw_dtrain['MSZoning'] = raw_dtrain['MSZoning'].transform(lambda x: 'C' if x=='C (all)' else x )

for field in transformation_matrix:
  raw_dtrain[field] = raw_dtrain[field].transform(lambda x: transformation_matrix[field][x] )

raw_dtrain['MiscVal'] = raw_dtrain['MiscVal'].transform(lambda x: 1 if x>0 else 0)

# %% new features

# how many years after remod when sold
raw_dtrain['RemodAge'] = pd.Series(raw_dtrain['YrSold'] - raw_dtrain['YearRemodAdd'])
# years after built when sold
raw_dtrain['HouseAge'] = pd.Series(raw_dtrain['YrSold'] - raw_dtrain['YearBuilt'])
# %%

# how old house is
raw_dtrain['Oldness'] = pd.Series(raw_dtrain['HouseAge']*0.5 + raw_dtrain['RemodAge'])
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

# grade
raw_dtrain['GarageAge'] = pd.Series(getGarageAge())
raw_dtrain['OverallValue'] = pd.Series(raw_dtrain['OverallQual'] + raw_dtrain['OverallCond'])
raw_dtrain['GarageGrade'] = pd.Series(raw_dtrain['GarageQual'] + raw_dtrain['GarageCond'])
raw_dtrain['ExterValue'] = pd.Series(raw_dtrain['ExterQual'] + raw_dtrain['ExterCond'])
raw_dtrain['KitchenValue'] = pd.Series(raw_dtrain['KitchenAbvGr'] * raw_dtrain['KitchenQual'])
raw_dtrain['FireplaceValue'] = pd.Series(raw_dtrain['Fireplaces'] * raw_dtrain['FireplaceQu'])
raw_dtrain['GarageValue'] = pd.Series(raw_dtrain['GarageArea'] * raw_dtrain['GarageQual'] * raw_dtrain['GarageFinish'])
raw_dtrain['BsmtValue'] = pd.Series(raw_dtrain['BsmtFinType1']*raw_dtrain['BsmtFinSF1'] + raw_dtrain['BsmtFinType2']*raw_dtrain['BsmtFinSF2'] + 0.2*raw_dtrain['BsmtUnfSF'] + raw_dtrain['BsmtCond']*raw_dtrain['BsmtQual']*raw_dtrain['TotalBsmtSF']*0.3)
raw_dtrain['BathValue'] = raw_dtrain['BsmtFullBath'] + 0.5*raw_dtrain['BsmtHalfBath'] + 2*raw_dtrain['FullBath'] + 1.5*raw_dtrain['HalfBath']

# total area
raw_dtrain['TotalPorchSF'] = pd.Series(raw_dtrain['OpenPorchSF'] + raw_dtrain['EnclosedPorch'] + raw_dtrain['3SsnPorch'] + raw_dtrain['ScreenPorch'])
raw_dtrain['TotalSF'] = pd.Series(raw_dtrain['1stFlrSF'] + raw_dtrain['2ndFlrSF'] + raw_dtrain['GrLivArea'] + 0.4*(raw_dtrain['LowQualFinSF'] + raw_dtrain['TotalBsmtSF']) + 0.1*(raw_dtrain['WoodDeckSF'] + raw_dtrain['TotalPorchSF'] + raw_dtrain['PoolArea'] + raw_dtrain['LotArea'] + raw_dtrain['GarageArea']) + raw_dtrain['LotArea'])


# One hot encoding
onehot_fields = ['MSSubClass','MSZoning','LotShape','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','Foundation','GarageType','MoSold','YrSold','SaleType','SaleCondition','LotConfig','LandContour']

for field in onehot_fields:
  # onehot_mat = pd.get_dummies(raw_dtrain[field])
  # print(field)
  # for name in onehot_mat.columns:
  #   raw_dtrain[field+'_'+str(name)] = onehot_mat[name]
  raw_dtrain.drop(onehot_fields , inplace=True, axis=1)

# %%





