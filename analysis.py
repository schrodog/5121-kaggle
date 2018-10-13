# %%
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import numpy as np
%matplotlib inline

data = pd.read_csv('./train.csv')
# %%
aa = data.isnull().sum()
# by using filter function in numpy array
# aa[aa > 0].sort_values(ascending=False)
# aa[aa > 0].index.tolist()
full = data

cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]

for col in cols1:
  full[col].fillna("None", inplace=True)

for col in cols:
  full[col].fillna(0, inplace = True)

full.groupby(['LotArea', 'Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()) )

# %%

# group different sub class into larger groups
full.groupby(['MSSubClass'])[['SalePrice']].agg(['mean','median','count'])

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.001)
lasso.fit(X_scaled, y_log)
LI_lasso = pd.DataFrame({"Feature importance": lasso.coef_}, index=data_pipe.columns)

FI_lasso[FI_lasso["Feature importance"] != 0].sort_values("Feature importance").plot(kind="barh", figsize=(15,25))
plt.xticks(rotation=90)
plt.show()


# %%
# sort by category

def plotCategory(types):
  src = (data[types]).tolist()

  cnt = collections.Counter()
  for nums in src:
    cnt[nums] += 1

  plt.figure(facecolor='#AAAAAA')
  plt.pie(cnt.values(), labels=cnt.keys())
  plt.title(types)
  plt.legend()
  plt.show()

# %%
# sort by value intervals
def plotIntervals(types):
  src = (data[types]).tolist()
  st = {'max': max(src), 'min': min(src)}
  num_intv = 5
  interval = (st['max'] - st['min'])*1.0/num_intv

  # print(st, interval)

  cnt = collections.Counter()
  na_count = 0
  for i in src:
    if math.isnan(i):
      na_count += 1
      continue
    cnt[math.floor((i-st['min'])/interval)] += 1


  # sort keys
  od = collections.OrderedDict(sorted(cnt.items()))
  # print(od)

  explode = [0.5 for i in od.keys()]
  labels = [str(round((i)*interval+st['min']))+'-'+str(round((i+1)*interval+st['min'])) for i in od.keys()]  

  if na_count > 0:
    labels.append('NA')
    od.update({'NA' : na_count})

  plt.figure(facecolor='#AAAAAA')
  plt.title(types)
  plt.pie(od.values(), labels=labels, labeldistance=1.1)
  plt.legend()
  plt.show()

# %%
def scatterPlot(typeA, typeB):
  src1 = (data[typeA]).tolist()
  src2 = (data[typeB]).tolist()

  plt.rcParams.update({'font.size': 15})
  plt.figure(facecolor="#AAAAAA", figsize=(10,10))
  plt.title(typeA + ' VS ' + typeB)
  plt.xlabel(typeA)
  plt.ylabel(typeB)
  plt.scatter(src1, src2)
  plt.show()

# %%

for i in ['Neighborhood']:
  plotCategory(i)


# for i in ['Neighborhood']:
#   plotIntervals(i)

# for i in data.dtypes.keys().tolist():
#   try:
#     scatterPlot('SalePrice', i)
#   except:
#     pass


# %%
