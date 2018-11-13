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
% matplotlib inline


# %%
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# print(train_data.shape)
# print(train_data.head(1))
# print(test_data.shape)
print(train_data.info())

# %%

import math

def num_range(start,stop,step):
  if step == 0:
    return [start]
  count = int((stop-start) / step) + 1
  return [start + i*step  for i in range(count)]

print(num_range(1, -8.1, -1.2))
# %%

columns = train_data.select_dtypes(exclude=['object']).columns.values.tolist()
columns.remove('Id')
columns.remove('SalePrice')

columns

# %%
sns.set()
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", col="time",
            hue="smoker", style="smoker", size="size",
            data=tips)


# %%
mean, cov = [0,2], [(1, 0.5), (0.5,1)]
x,y = np.random.multivariate_normal(mean, cov, size=50).T
ax = sns.kdeplot(x,y, shade=True, color="b")

# %%
dots = sns.load_dataset("dots")
sns.relplot(x="time", y="firing_rate", col="align", hue="choice", size="coherence", kind="line", legend="full", data=dots)

# %%
sns.catplot(x="day", y="total_bill", hue="smoker", kind="swarm", data=tips)






