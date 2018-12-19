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
# %%

# print(raw_dtrain['BsmtFinSF1'].describe())
np.count_nonzero(raw_dtrain['3SsnPorch'] == 0)
# %%

# types = 'GarageAge'
# raw_dtrain[['ExterQual','ExterCond','Id']].groupby(['ExterQual','ExterCond']).count()

data = raw_dtrain[raw_dtrain['PoolArea'] >= 0]
# %%

# data = raw_dtrain[raw_dtrain['MiscVal'] > 0]

gg = (ggplot(data, aes(x='PavedDrive',y='SalePrice'))
  + geom_point()
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
# gg.save('outputs/month_price.pdf')

# %%
print(np.count_nonzero(raw_dtrain['GarageArea'] == 0))




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














