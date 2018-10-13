# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import numpy as np
import re
%matplotlib tk


# %%

features = pd.read_csv('./train.csv')
features = pd.get_dummies(features)
# features.columns.values

# %%

labels = np.array(features['SalePrice'])
chosenFeatures = features.columns.values.tolist()
r = re.compile("(Neighborhood_|ExterQual_|BsmtQual_|BsmtCond_).*")
selected = list(filter(r.match, chosenFeatures)) + ['OverallQual','GrLivArea','GarageArea','YearBuilt','TotalBsmtSF']
# selected = list(filter(r.match, chosenFeatures)) + ['OverallQual','GrLivArea','GarageArea','YearBuilt','TotalBsmtSF','1stFlrSF','2ndFlrSF','BsmtFinSF1','FullBath','TotRmsAbvGrd','Fireplaces']
# selected = ['OverallQual','GrLivArea','GarageArea','YearBuilt',]
selectedFeatures = features[selected]
feature_list = list(selectedFeatures.columns)

features = np.array(selectedFeatures)
# selected

# %%
train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=0.25, random_state=42)

# print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)
# baseline_preds = test_features[:, feature_list.index('average')]
# baseline_errors = abs(baseline_preds - test_labels)
# print('baseline error:', round(np.mean(baseline_errors), 2))

# %%
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=12), n_estimators=400, random_state=42, learning_rate=1.0)
rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)

print('mean abs error', np.mean(errors))

# %%

plt.rcParams.update({'font.size': 15})
fig = plt.figure(facecolor="#AAAAAA", figsize=(15, 15))

plt.title("original VS predictions")
plt.plot(test_labels, label='label')
plt.plot(predictions, label='predict')
plt.legend()
plt.show()


# %%
[i for i in zip(predictions, test_labels)]


# %%
np.mean(errors)
