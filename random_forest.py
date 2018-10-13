# %%
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
from sklearn.model_selection import train_test_split
import pandas as pd

# %%
x = np.random.uniform(1,100, 1000)
y = np.log(x) + np.random.normal(0, .3, 1000)

plt.rcParams.update({'font.size': 15})
plt.figure(facecolor="#AAAAAA", figsize=(10,10))
plt.scatter(x,y)
plt.plot(np.arange(1,100), np.log(np.arange(1,100)), color='b')
plt.xlabel('x')
plt.ylabel('f(x) = log(x)')
plt.title("a basic log function")
plt.show()

# %%

features = pd.read_csv('./temps.csv')
# print(features.describe())

plt.rcParams.update({'font.size': 15})
fig = plt.figure(facecolor="#AAAAAA", figsize=(15, 15))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.set_title("1 day prior")
ax1.set_ylabel("temp")
ax1.plot(features['temp_1'])

ax2.set_title("2 days prior")
ax2.set_ylabel("temp")
ax2.plot(features['temp_2'])

ax3.set_title("friend")
ax3.set_ylabel("temp")
ax3.plot(features['friend'])

ax4.set_title("max temp actual")
ax4.set_ylabel("temp")
ax4.plot(features['actual'])


# %%
features = pd.read_csv('./temps.csv')

features = pd.get_dummies(features)
labels = np.array(features['actual'])
features = features.drop('actual', axis=1)
feature_list = list(features.columns)

features = np.array(features)

# %%
train_features, test_features, train_labels, test_labels = \
  train_test_split(features, labels, test_size = 0.25, random_state = 42)

# print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)
baseline_preds = test_features[:, feature_list.index('average')]
baseline_errors = abs(baseline_preds - test_labels)
print('baseline error:', round(np.mean(baseline_errors), 2))

# %%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 10, random_state =42)
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






























