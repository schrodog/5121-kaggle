# %%
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import numpy as np
%matplotlib inline

data = pd.read_csv('./train.csv')

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

# for i in ['YrSold']:
#   plotCategory(i)

# plotIntervals('YrSold')

for i in data.dtypes.keys().tolist():
  try:
    scatterPlot('SalePrice', i)
  except:
    pass


# %%
