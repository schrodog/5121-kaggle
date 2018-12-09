# %%
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data)
print(iris.target)

# %%

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

# StandardScaler().fit_transform(iris.data)
# MinMaxScaler().fit_transform(iris.data)
Normalizer().fit_transform(iris.data)

# %%
from sklearn.preprocessing import Binarizer, OneHotEncoder

# Binarizer(threshold=3).fit_transform(iris.data)
print(OneHotEncoder().fit_transform(iris.target.reshape((-1,1)) ))

# %%

enc = OneHotEncoder(handle_unknown='ignore')
x = [['male',1], ['female',3], ['female',2],['male',2]]
# enc.fit(x)

# enc.categories_
# enc.transform([['female',1], ['male',4]]).toarray()

print(enc.fit_transform(x).toarray())
print(enc.get_feature_names())

# %% multilabel 
from sklearn.preprocessing import KBinsDiscretizer
from random import randint
import numpy as np

x = [[randint(0,20) for _ in range(4)] for _ in range(20)]

print('x:\n', np.array(x))
est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
print(est.fit_transform(x))

# %% 
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x = ['p','p','t','a','a','b']
le.fit_transform(x)

# %%
from sklearn.impute import SimpleImputer

enc = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
enc.fit_transform([[7,2,3], [4,np.nan, 6], [10,3,9], [2,3,6]])

# %%
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer

x= [[1,2],[4,5]]
PolynomialFeatures().fit_transform(x)

# from numpy import log1p
# # log1p = ln(1+x)
# def add(x):
#   return x+1

# FunctionTransformer(add).fit_transform(iris.data)

# %%
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from scipy.stats import pearsonr
from sklearn.datasets import load_digits

# x = [[0,2,0,3],[0,1,4,3],[0,1,1,1]]
# VarianceThreshold(threshold=0.25).fit_transform(x)

# X,y = iris.data, iris.target
# print(X.shape)
# print(X[:5])
# le = SelectKBest(chi2, k=2)
# a = le.fit_transform(X,y)
# print(a.shape)
# print(le.get_support())
# print(le.scores_, le.pvalues_)

SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)


# %%
from minepy import MINE

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


# print(iris.data.shape, iris.target.shape)
SelectKBest(mics, k=2).fit_transform(iris.data, iris.target)

# %%
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)

# %%
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)
# SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)

# %%
from sklearn.decomposition import PCA
from sklearn.lda import LDA

print(iris.data.shape)
a = PCA(n_components=2).fit_transform(iris.data)
print(a.shape)

LDA(n_components=2).fit_transform(iris.data, iris.target)









