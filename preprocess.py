# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from minepy import MINE

pd.options.mode.chained_assignment = None
np.set_printoptions(precision=5, suppress=True)

# import socket; socket.gethostname()

raw_dtrain = pd.read_csv('data/train.csv')
raw_dtest = pd.read_csv('data/test.csv')

SalePrice = raw_dtrain['SalePrice']
Test_id = raw_dtest['Id']
raw_dtrain.drop(['Id','SalePrice'] , inplace=True, axis=1)
raw_dtest.drop(['Id'] , inplace=True, axis=1)

combined_df = pd.concat([raw_dtrain, raw_dtest], keys=['train','test'])
# combined_df = raw_dtest


# fill missing values
# fill training dataset
none_type_word = "NA"

combined_df['Alley'].fillna(none_type_word, inplace=True)
combined_df['FireplaceQu'].fillna(none_type_word, inplace=True)
combined_df['PoolQC'].fillna(none_type_word, inplace=True)
combined_df['BsmtQual'].fillna(none_type_word, inplace=True)
combined_df['MiscFeature'].fillna(none_type_word, inplace=True)
combined_df['Fence'].fillna(none_type_word, inplace=True)
combined_df['MiscFeature'].fillna(none_type_word, inplace=True)

combined_df['GarageFinish'].fillna(none_type_word, inplace=True)
combined_df['GarageType'].fillna(none_type_word, inplace=True)
combined_df['GarageQual'].fillna(none_type_word, inplace=True)
combined_df['GarageCond'].fillna(none_type_word, inplace=True)
combined_df['GarageType'].fillna(none_type_word, inplace=True)
combined_df['GarageYrBlt'].fillna(-1, inplace=True)

combined_df['BsmtCond'].fillna(none_type_word, inplace=True)
combined_df['BsmtQual'].fillna(none_type_word, inplace=True)
combined_df['BsmtExposure'].fillna('No', inplace=True)
combined_df['BsmtFinType1'].fillna('Unf', inplace=True)
combined_df['BsmtFinType2'].fillna('Unf', inplace=True)

combined_df['MasVnrType'].fillna('None', inplace=True)
combined_df['MasVnrArea'].fillna(0, inplace=True)
combined_df['Electrical'].fillna('Sbrkr', inplace=True)

# fill testing dataset
combined_df['MSZoning'].fillna('RL', inplace=True)
combined_df['Exterior1st'].fillna('Plywood', inplace=True)
combined_df['Exterior2nd'].fillna('Plywood', inplace=True)
combined_df['BsmtFullBath'].fillna(0, inplace=True)
combined_df['BsmtHalfBath'].fillna(0, inplace=True)
combined_df['BsmtFinSF1'].fillna(0, inplace=True)
combined_df['BsmtFinSF2'].fillna(0, inplace=True)
combined_df['BsmtUnfSF'].fillna(0, inplace=True)
combined_df['TotalBsmtSF'].fillna(0, inplace=True)
combined_df['KitchenQual'].fillna('TA', inplace=True)
combined_df['Functional'].fillna('Typ', inplace=True)
combined_df['GarageCars'].fillna(1, inplace=True)
combined_df['GarageArea'].fillna(384, inplace=True)
combined_df['SaleType'].fillna('WD', inplace=True)


avg_data = combined_df['LotFrontage'].groupby(combined_df['Neighborhood']).median()
null_lot = combined_df['LotFrontage'].isnull()
combined_df['LotFrontage'][null_lot] = combined_df['Neighborhood'][null_lot].map(lambda x: avg_data[x])

# drop features
combined_df.drop(['CentralAir', 'PoolQC', 'Condition2', 'RoofMatl', 'Street', 'Utilities', 'MiscFeature'], inplace=True, axis=1)

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
'Neighborhood': {'Blmngtn': 2,'Blueste': 3,'BrDale': 2,'BrkSide': 0,'ClearCr': 1,'CollgCr': 1,'Crawfor': 0,'Edwards': 1,'Gilbert': 0,'GrnHill': 3,'IDOTRR': 0,'Landmrk': 2,'MeadowV': 4,'Mitchel': 0,'NAmes': 0,'NPkVill': 2,'NWAmes': 2,'NoRidge': 2,'NridgHt': 2,'OldTown': 0,'SWISU': 3,'Sawyer': 1,'SawyerW': 1,'Somerst': 2,'StoneBr': 2,'Timber': 3,'Veenker': 2},
'MSSubClass': {20:2, 30:0 , 40:2 , 45:0 , 50:1 , 60:3 , 70:2 , 75:2 ,80:2 , 85:1 ,90:1 , 120:3, 150:4, 160:2 ,180:0 , 190:1 }
}
# correct error
combined_df['Electrical'] = combined_df['Electrical'].transform(lambda x: 'SBrkr' if x=='Sbrkr' else x )
combined_df['MSZoning'] = combined_df['MSZoning'].transform(lambda x: 'C' if x=='C (all)' else x )

for field in transformation_matrix:
  combined_df[field] = combined_df[field].transform(lambda x: transformation_matrix[field][x] )

combined_df['MiscVal'] = combined_df['MiscVal'].transform(lambda x: 1 if x>0 else 0)

# new features

# how many years after remod when sold
combined_df['RemodAge'] = pd.Series(combined_df['YrSold'] - combined_df['YearRemodAdd'])
# years after built when sold
combined_df['HouseAge'] = pd.Series(combined_df['YrSold'] - combined_df['YearBuilt'])

# how old house is
combined_df['Oldness'] = pd.Series(combined_df['HouseAge']*0.5 + combined_df['RemodAge'])
# how many years garage built
def getGarageAge():
  res = []
  for i in range(combined_df.shape[0]):
    sold = combined_df['YrSold'][i]
    garage = combined_df['GarageYrBlt'][i]
    remod = combined_df['YearRemodAdd'][i]
    if garage == -1:
      res.append(-10)
    else:
      res.append(sold - max(garage, remod))
  return np.array(res)

# grade
# must supply index if df's index different from array's index
combined_df['GarageAge'] = pd.Series(getGarageAge(), index=combined_df.index)
combined_df['OverallValue'] = pd.Series(combined_df['OverallQual'] + combined_df['OverallCond'])
combined_df['GarageGrade'] = pd.Series(combined_df['GarageQual'] + combined_df['GarageCond'])
combined_df['ExterValue'] = pd.Series(combined_df['ExterQual'] + combined_df['ExterCond'])
combined_df['KitchenValue'] = pd.Series(combined_df['KitchenAbvGr'] * combined_df['KitchenQual'])
combined_df['FireplaceValue'] = pd.Series(combined_df['Fireplaces'] * combined_df['FireplaceQu'])
combined_df['GarageValue'] = pd.Series(combined_df['GarageArea'] * combined_df['GarageQual'] * combined_df['GarageFinish'])
combined_df['BsmtValue'] = pd.Series(combined_df['BsmtFinType1']*combined_df['BsmtFinSF1'] + combined_df['BsmtFinType2']*combined_df['BsmtFinSF2'] + 0.2*combined_df['BsmtUnfSF'] + combined_df['BsmtCond']*combined_df['BsmtQual']*combined_df['TotalBsmtSF']*0.3)
combined_df['BathValue'] = combined_df['BsmtFullBath'] + 0.5*combined_df['BsmtHalfBath'] + 2*combined_df['FullBath'] + 1.5*combined_df['HalfBath']

# total area
combined_df['TotalPorchSF'] = pd.Series(combined_df['OpenPorchSF'] + combined_df['EnclosedPorch'] + combined_df['3SsnPorch'] + combined_df['ScreenPorch'])
combined_df['TotalSF'] = pd.Series(combined_df['1stFlrSF'] + combined_df['2ndFlrSF'] + combined_df['GrLivArea'] + 0.4*(combined_df['LowQualFinSF'] + combined_df['TotalBsmtSF']) + 0.1*(combined_df['WoodDeckSF'] + combined_df['TotalPorchSF'] + combined_df['PoolArea'] + combined_df['LotArea'] + combined_df['GarageArea']) + combined_df['LotArea'])


# One hot encoding
onehot_fields = ['MSZoning','LotShape','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','Foundation','GarageType','SaleType','SaleCondition','LotConfig','LandContour','MSSubClass']
# 'YrSold','MoSold'

for field in onehot_fields:
  onehot_mat = pd.get_dummies(combined_df[field])
  for name in onehot_mat.columns:
    combined_df[field+'_'+str(name)] = onehot_mat[name]

combined_df.drop(onehot_fields , inplace=True, axis=1)


# Extract train, test data
train_data = combined_df[combined_df.index.labels[0] == 0].values
test_data = combined_df[combined_df.index.labels[0] == 1].values

unorm_train = pd.DataFrame(train_data, columns=combined_df.columns)
unorm_train['SalePrice'] = pd.Series(SalePrice.values)
unorm_train.to_csv("result/unorm_train.csv", index=False)

unorm_test = pd.DataFrame(test_data, columns=combined_df.columns)
unorm_test['Id'] = pd.Series(Test_id)
unorm_test.to_csv("result/unorm_test.csv", index=False)

# Adjust skewed
skewed_feature = ['LotArea','TotalSF','Heating','MiscVal','KitchenAbvGr','Alley']
for i in skewed_feature:
  unorm_train[i] = np.log1p(unorm_train[i])

# Clip outlier
clip_upper_feature = ['GrLivArea','2ndFlrSF','1stFlrSF','TotalBsmtSF','MasVnrArea','LotArea','LotFrontage', 'GarageArea', 'GarageCars', 'SalePrice','BsmtValue','TotalSF']

clip_lower_feature = ['SalePrice','1stFlrSF']
for i in clip_upper_feature:
  unorm_train[i] = unorm_train[i].clip_upper(unorm_train[i].quantile(0.99))
  unorm_train[i] = unorm_train[i].clip_lower(unorm_train[i].quantile(0.01))
# for i in clip_lower_feature:


# remove outlier
# outlier_index = unorm_train[(unorm_train['TotalSF'] > 100000) | (unorm_train['BsmtValue']>60000) | (unorm_train['TotalBsmtSF'] > 6000) | (unorm_train['LotFrontage'] > 200) | (unorm_train['LotArea'] > 100000) | (unorm_train['MasVnrArea'] > 1500)].index
# outlier_index = unorm_train[(unorm_train['GrLivArea'] > 4000)].index
# unorm_train.drop(outlier_index, inplace=True)

# print(outlier_index)
train_data2 = unorm_train.drop(['SalePrice'], axis=1)


# Scaling
trans_train = RobustScaler().fit_transform(train_data2.values)
trans_test = RobustScaler().fit_transform(test_data)

output_train = pd.DataFrame(trans_train, columns=combined_df.columns)
output_train['SalePrice'] = pd.Series(np.log1p(SalePrice.values))

output_test = pd.DataFrame(trans_test , columns=combined_df.columns)
output_test['Id'] = pd.Series(Test_id)


# selected most important features

important_features = ['OverallQual', 'ExterQual', 'GarageValue',
  'GrLivArea', 'GarageAge', 'KitchenQual', 'FullBath', 'BsmtQual',
  'BathValue', 'HouseAge', 'GarageCars', 'Oldness',
  'GarageFinish', 'GarageArea', 'KitchenValue', 'ExterValue',
  'FireplaceValue', 'FireplaceQu', 'Foundation_PConc',
  'OverallValue', 'TotalBsmtSF', 'BsmtValue',
  'Fireplaces', 'TotalSF', '1stFlrSF', 'RemodAge', 'YearRemodAdd',
  'OpenPorchSF', 'Neighborhood_2', 'HeatingQC', 'GarageType_Attchd',
  'TotRmsAbvGrd', 'LotArea', '2ndFlrSF',
  'GarageType_Detchd', 'Foundation_CBlock', 'Exterior2nd_VinylSd',
  'MasVnrType_None', 'Exterior1st_VinylSd', 'GarageGrade',
  'MasVnrArea', 'MSZoning_1', 'Neighborhood_0', 'LotFrontage',
  'HalfBath', 'BsmtFinSF1']

# important_features = [
#   'OverallValue', 'GrLivArea', 'BsmtValue', 'OverallQual', 'Oldness',
#   'TotalSF', 'TotalBsmtSF', 'KitchenQual', 'GarageCars',
#   'GarageValue', 'BathValue', 'GarageArea', 'FireplaceValue',
#   'ExterQual', 'FireplaceQu', '1stFlrSF', 'GarageFinish', 'BsmtQual',
#   'HouseAge', 'HeatingQC', 'LotArea', 'MSZoning_1', 'MSSubClass_0',
#   'YearBuilt', 'Fireplaces', 'RemodAge', 'TotRmsAbvGrd',
#   'ExterValue', 'BsmtFinType1', 'Foundation_PConc', '2ndFlrSF',
#   'YearRemodAdd', 'FullBath', 'HalfBath', 'BsmtFinSF1',
#   'Neighborhood_2', 'MSZoning_0', 'GarageGrade', 'GarageType_Attchd',
#   'LotFrontage', 'GarageQual', 'PavedDrive', 'GarageCond',
#   'MSSubClass_3', 'BsmtExposure', 'HouseStyle_2Story', 'WoodDeckSF',
#   'BsmtFullBath', 'GarageYrBlt', 'BedroomAbvGr'
# ]


final_train = output_train[important_features+['SalePrice']]
final_test = output_test[important_features+['Id']]

# final_train = output_train
# final_test = output_test


# pca
# from sklearn.decomposition import PCA

# pca = PCA(60)
# pca_train = pca.fit_transform(final_train.values)
# pca_test = pca.fit_transform(final_test.values)

# x1 = dict([['x'+str(i), pca_train[:,i]] for i in range(pca_train.shape[1])])
# x2 = dict([['x'+str(i), pca_test[:,i]] for i in range(pca_test.shape[1])])
# x1['SalePrice'] = pd.Series(np.log1p(SalePrice.values))
# x2['Id'] = pd.Series(Test_id)


# pca_train_df = pd.DataFrame(x1)
# pca_test_df = pd.DataFrame(x2)


# output
final_train.to_csv("result/new_train8.csv", index=False)
final_test.to_csv("result/new_test8.csv", index=False)

# %%














