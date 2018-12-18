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

# %%

# %% fill missing value

# data = raw_dtrain.loc[np.invert(pd.isnull(raw_dtrain['LotFrontage']))]
# data['MSSubClass'] = data['MSSubClass'].apply(str)
# %%

# (ggplot(data)
#   # + geom_point(aes(x='MasVnrType', y='LotArea'))
#   + geom_bar(aes(x='MasVnrType'))
# )

# %%

print(raw_dtrain['MSSubClass'].describe())

# %%

types = 'GarageAge'
data = raw_dtrain
# data = raw_dtrain[raw_dtrain['GarageAge'] >= 0]

gg = (ggplot(data, aes('Condition1'))
  # + geom_point(aes(x=types, y='GarageType'))
  + geom_bar()
  + stat_count(aes(label='stat(count)'), geom='text', position=position_stack(vjust=1.05))
  # + geom_point()
  # + geom_histogram(binwidth=10000)
  # + facet_wrap('MoSold')
  # + scale_y_continuous(breaks=range(1850, 2020, 10) )
  # + coord_cartesian(ylim=(1900,2010))
  + theme(axis_text_x=element_text(rotation=0, ha="right"))
)

print(gg)
# gg.save('outputs/month_price.pdf')

# print(raw_dtrain[types].describe())

# %%
print(np.count_nonzero(raw_dtrain['GarageArea'] == 0))


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

def label2Num(*args):
  lis = args[0] if isinstance(args[0],list) else [args[0]]
  for field in lis:
    uniq = raw_dtrain[field].unique()
    trans = dict([(uniq[i],i) for i in range(len(uniq))])
    raw_dtrain[field] = raw_dtrain[field].transform(lambda x: trans[x])

# %% binary variables for very low variance features
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction import DictVectorizer

label2Num(['Condition1','Condition2'])
# v = DictVectorizer(sparse=False)
# X = v.fit_transform(raw_dtrain['Condition1'])
# print(v.get_feature_names())

# data = np.array(raw_dtrain['Condition2']).reshape(-1,1)
data = raw_dtrain[['Condition1', 'Condition2']]
sel = VarianceThreshold(0.01)
sel_value = sel.fit_transform(data)

sel.variances_


# %% label -> number

# abc = pd.DataFrame({'x': ['a','b','c','c','a']})

abc['x'] = abc['x'].transform(lambda x: mapping['b'][x])


mapping = {
# num -> label
'MSSubClass': {20:'20',30:'30',40:'40',45:'45',50:'50',60:'60',70:'70',75:'75',80:'80',85:'85',90:'90',120:'120',150:'150',160:'160',180:'180',190:'190'},

# label -> num
'LotShape': {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3},
''

# grouping
'OverallQual': {1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:2, 8:2, 9:2, 10:2},
''
}



# %%


















