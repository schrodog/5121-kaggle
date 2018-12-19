# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import math
import numpy as np
import re
import pandas as pd
from plotnine import *
% matplotlib tk


# %%
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# print(train_data.shape)
# print(train_data.head(1))
# print(test_data.shape)
print(train_data.info())
print(train_data.head())

# %%

# import math

# def num_range(start,stop,step):
#   if step == 0:
#     return [start]
#   count = int((stop-start) / step) + 1
#   return [start + i*step  for i in range(count)]

# print(num_range(1, -8.1, -1.2))
previous_num_columns = train_data.select_dtypes(exclude=['object']).columns.values.tolist()
previous_num_columns.remove('Id')
previous_num_columns.remove('SalePrice')


# %%
test_col = 'TotRmsAbvGrd'

maxs = max(train_data[test_col]), max(test_data[test_col])
xx = (range(len(test_data[test_col])), range(len(train_data[test_col]) ))

(ggplot()
  + geom_point(train_data, aes(x=xx[1],y=test_col), color='green')
  + geom_point(test_data, aes(x=xx[0],y=test_col), color='red')
  # + xlim(0,max(maxs[0], maxs[1]))
  + xlim(0, max(xx[0][-1], xx[1][-1]) )
)

# %%

(ggplot()
  + geom_density(train_data, aes(test_col), color='green')
  + geom_density(test_data, aes(test_col), color='red')
  + xlim(0,max(maxs[0], maxs[1]))
)


# %%
limit = 3500
print('<train>\n',train_data[test_col][train_data[test_col] > limit])
print('<test>\n',test_data[test_col][test_data[test_col] > limit])

# %%
test_col = 'LotArea'

density = pd.DataFrame({
  'train': train_data[test_col],
  'test': test_data[test_col]
})


df_density = pd.melt(density, value_vars=['train', 'test'])
df_density
# %%
gg = (ggplot(df_density, aes(color='variable'))
  + geom_density(aes('value'), size=1)
  + xlim(0, 500)
)
gg
# gg.save()

# %%
for i in range(100, 3000, 10):
  a = sum(train_data[test_col] > i)
  b = sum(test_data[test_col] > i)
  if a == 0 or b == 0:
    print(i,':',a,b)
    break



# %%

train_data.drop(columns=['Id', 'SalePrice'], inplace=True)
train_data.head()

# %%

combined_data = pd.concat([train_data.loc[:, :'SalePrice'], test_data])
combined_data = combined_data[test_data.columns]

combined_data.head(1)
combined_data.shape





