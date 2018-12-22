# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from scipy.stats import pearsonr
from minepy import MINE

pd.options.mode.chained_assignment = None
np.set_printoptions(precision=5, suppress=True)


raw_dtrain = pd.read_csv('data/train.csv')
raw_dtest = pd.read_csv('data/test.csv')

# raw_dtrain = pd.read_csv('result/unorm_train.csv')
# raw_dtest = pd.read_csv('result/unorm_test.csv')
# raw_dtrain = pd.read_csv('result/new_train7.csv')
# raw_dtest = pd.read_csv('result/new_test7.csv')

data = raw_dtrain.copy()

# %%

skewed = raw_dtrain.skew()

skew_df = pd.DataFrame({'x': skewed.index, 'y': skewed.values})
skew_df.sort_values(by='y', ascending=False).iloc[30:80]

# %%
# raw_dtest[raw_dtest['Exterior1st'].isnull()]
raw_dtest[['Exterior1st','RoofMatl','Id']].groupby(['Exterior1st','RoofMatl']).agg('count')

# %%


# %%

# data['LotArea-root5'] = data['LotArea'].transform(lambda x: x**(1/5) )
# trans = {1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1, 8:2, 9:2, 10:2}
# data['g_OverallQual'] = data['OverallQual'].transform(lambda x: trans[x])

data['GrLivArea'] = data['GrLivArea'].clip_upper(data['GrLivArea'].quantile(0.99))
# data['GrLivArea'].describe()


# %%
data.groupby('MSSubClass')['SalePrice'].agg(['mean','median','count']).sort_values(by='median')

# %%
types = "SalePrice"
gg = (ggplot(data, aes('GrLivArea'))
  # + geom_point()
  # + geom_col()
  # + geom_bar()
  # + stat_count(aes(label='stat(count)'), geom='text', position=position_stack(vjust=1.05))
  + geom_histogram(binwidth=60)
  # + facet_wrap('MSSubClass')
  # + scale_y_continuous(breaks=range(1850, 2020, 10) )
  # + scale_x_continuous(name="train")
  # + coord_cartesian(ylim=(1900,2010))
  # + theme(axis_text_x=element_text(rotation=60, ha="right"))
)
print(gg)
# %%

gg.save('result/grlivarea.png')
# %%


# %%

# gg.save('outputs/month_price.pdf')

gh = (ggplot(raw_dtest, aes('2nd'))
  # + geom_point()
  + geom_histogram(binwidth=0.1)
  + scale_x_continuous(name="test")
  # + geom_histogram(binwidth=50)
)
print(gh)
# %%
data['YrSold']

# %%
types = "BsmtValue"
value = 60000
print(np.count_nonzero(data[types] > value))
print(np.count_nonzero(raw_dtest[types] > value))
# %%
data[(data['TotalSF'] > 100000) | (data['BsmtValue']>60000) | (data['TotalBsmtSF'] > 6000) | (data['LotFrontage'] > 200) | (data['LotArea'] > 100000) | (data['MasVnrArea'] > 1500)].index


# data[['SalePrice','ExterQual']].groupby(['ExterQual']).count()
# data['TotalSF'].values > 100000

# %%

def mic(x,y):
  m = MINE()
  m.compute_score(x,y)
  return m.mic()
  # return (m.mic(), 0.5)

def pear_pa(x, y):
  ret_temp = map(lambda x: pearsonr(x, y), x.T)
  return [i[0] for i in ret_temp], [i[1] for i in ret_temp]


def mics(x, y):
  temp = map(lambda x: mic(x,y), x.T)
  return [i[0] for i in temp], [i[1] for i in temp]

# %%
mic(data['MoSold'], data['SalePrice'])

# %%

test_feat = data.columns
# test_feat = ['LotFrontage','GrLivArea','TotalSF','BsmtValue','OverallCond','LotArea','Oldness','OverallValue','1stFlrSF','OverallQual','TotalBsmtSF','Functional','GarageArea','YearBuilt','BsmtFinSF1','GarageYrBlt','YearRemodAdd','GarageValue','2ndFlrSF','HouseAge']

corr = [mic(data['OverallQual'], data[i]) for i in test_feat]
corr_df = pd.DataFrame({'field': test_feat, 'corr': corr})

corr_df.sort_values(by='corr', ascending=False)

# %%
# corr_df[corr_df['field'] == 'LotFrontage']
corr_df.sort_values(by='corr', ascending=False).values[:60][:,0]






# %% binary variables for very low variance features
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction import DictVectorizer

var_data = raw_dtrain.copy()
def label2Num(*args):
  lis = args[0] if isinstance(args[0],list) else [args[0]]
  for field in lis:
    uniq = var_data[field].unique()
    trans = dict([(uniq[i],i) for i in range(len(uniq))])
    var_data[field] = var_data[field].transform(lambda x: trans[x])

test_list = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']
label2Num(test_list)
# v = DictVectorizer(sparse=False)
# X = v.fit_transform(raw_dtrain['Condition1'])
# print(v.get_feature_names())

# data = np.array(raw_dtrain['Condition2']).reshape(-1,1)
data = var_data[test_list]
sel = VarianceThreshold(0.01)
sel_value = sel.fit_transform(data)

sel.variances_
# %%

orders = np.argsort(sel.variances_)
np.sort(sel.variances_, order=orders)

# %%

var_data = pd.DataFrame({'x': test_list, 'y': sel.variances_})
sf = var_data.sort_values(by='y', ascending=False)['x']
var_data['x'] = pd.Categorical(var_data['x'], categories=sf.values, ordered=True)
# data = data.reset_index(drop=True)

gg = (ggplot(var_data)
  + geom_col(aes(x='x',y='y'))
  + theme(axis_text_x=element_text(rotation=70, ha="right"))
)

print(gg)
# %%
sf.values


# %% find neighborhood
data = pd.DataFrame({
  "latitude": {'Blmngtn' : 42.062806,'Blueste' :42.009408,'BrDale' : 42.052500,'BrkSide':42.033590,'ClearCr': 42.025425,'CollgCr':42.021051,'Crawfor': 42.025949,'Edwards':42.022800,'Gilbert': 42.027885,'GrnHill':42.000854,'IDOTRR' : 42.019208,'Landmrk':42.044777,'MeadowV': 41.991866,'Mitchel':42.031307,'NAmes'  : 42.042966,'NoRidge':42.050307,'NPkVill': 42.050207,'NridgHt':42.060356,'NWAmes' : 42.051321,'OldTown':42.028863,'SWISU'  : 42.017578,'Sawyer' :42.033611,'SawyerW': 42.035540,'Somerst':42.052191,'StoneBr': 42.060752,'Timber' :41.998132,'Veenker': 42.040106},

  "longitude": {'Blmngtn' : -93.639963,'Blueste' : -93.645543,'BrDale' : -93.628821,'BrkSide': -93.627552,'ClearCr': -93.675741,'CollgCr': -93.685643,'Crawfor': -93.620215,'Edwards': -93.663040,'Gilbert': -93.615692,'GrnHill': -93.643377,'IDOTRR' : -93.623401,'Landmrk': -93.646239,'MeadowV': -93.602441,'Mitchel': -93.626967,'NAmes'  : -93.613556,'NoRidge': -93.656045,'NPkVill': -93.625827,'NridgHt': -93.657107,'NWAmes' : -93.633798,'OldTown': -93.615497,'SWISU'  : -93.651283,'Sawyer' : -93.669348,'SawyerW': -93.685131,'Somerst': -93.643479,'StoneBr': -93.628955,'Timber' : -93.648335,'Veenker': -93.657032}
})

data
# %%
from sklearn.cluster import KMeans

coords = data[['latitude','longitude']].values
kmeans = KMeans(n_clusters=5, random_state=2).fit(coords)

data['classes'] = pd.Series(kmeans.labels_, index=data.index)

gg = (ggplot(data, aes(x='latitude', y='longitude', color='classes', size='classes'))
  + geom_point()
  # + scale_fill_hue(expand=range(10))
  # + scale_colour_manual(values = ["red", "blue", "green"])
  # + geom_col()
  # + geom_bar()
  # + stat_count(aes(label='stat(count)'), geom='text', position=position_stack(vjust=1.05))
  # + geom_point()
  # + geom_histogram(binwidth=10)
  # + facet_wrap('Neighborhood')
  # + scale_y_continuous(breaks=range(1850, 2020, 10) )
  # + coord_cartesian(ylim=(1900,2010))
  # + theme(axis_text_x=element_text(rotation=0, ha="right"))
)
print(gg)
# gg.save('result/neighbor_dist.png')

# %%
np.column_stack((data.index, kmeans.labels_))

# %%

combined_df.columns[combined_df.count() < 2919]
# combined_df.info()

# %% feature selection L1, L2

from sklearn.linear_model import Lasso, Ridge, RandomizedLasso
from sklearn.preprocessing import StandardScaler

raw_train_df = pd.read_csv('result/new_train8.csv')
raw_trainY = raw_train_df['SalePrice']
raw_train_df.drop(['SalePrice'], inplace=True, axis=1)

# scaler = StandardScaler()
# raw_trainX = scaler.fit_transform(raw_train_df.values)
raw_trainX = raw_train_df.values

# model = Ridge(alpha=0.3)
model = RandomizedLasso(alpha=0.001)
model.fit(raw_trainX, raw_trainY)

feat_df = pd.DataFrame({'x': raw_train_df.columns, 'y': model.scores_})
feat_df.sort_values(by='y', ascending=False)

# %%
feat_df.sort_values(by='y', ascending=False)['x'].values[:50]



# %%

gh = (ggplot(raw_train_df, aes(x=raw_train_df.index, y='GarageYrBlt'))
  + geom_point()
  # + scale_x_continuous(name="test")
  # + geom_histogram(binwidth=50)
)
print(gh)

# %% PCA

from sklearn.decomposition import PCA

pca_hp = PCA(30)
x_fit = pca_hp.fit_transform(raw_trainX)

# pca_hp.components_
# %%
xx = np.array([[1,2,3],[4,5,6]])

dict([['x'+str(i), xx[i]] for i in range(xx.shape[0])])




















