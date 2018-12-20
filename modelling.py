# %%
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from plotnine import *
import warnings
# warnings.filterwarnings('ignore')
# %%
import socket; socket.gethostname()

# %%
raw_train_df = pd.read_csv('result/new_train.csv')
raw_test_df = pd.read_csv('result/new_test.csv')

raw_trainY = raw_train_df['SalePrice']
raw_train_df.drop(['SalePrice'], inplace=True, axis=1)
raw_trainX = raw_train_df
test_id = raw_test_df['Id']
raw_test_df.drop(['Id'], inplace=True, axis=1)

train_data = xgb.DMatrix(raw_trainX, label=raw_trainY, nthread=-1)
label_data = xgb.DMatrix(raw_trainY, nthread=-1)
test_data = raw_test_df.values

# %%
def model_training(regressor, data, tune_param=None, label=None):
  all_param = regressor.get_xgb_params()
  best_params, best_cvresult, min_rmse = {}, None, None

  early_stopping_rounds = 60
  cv_folds = 5

  if tune_param:
    for param, value in tune_param.items():
      best_param = value[0]
      for val in value:
        all_param[param] = val
        cvresult = xgb.cv(
          all_param, data, num_boost_round=all_param['n_estimators'], 
          nfold=cv_folds, metrics='rmse', 
          early_stopping_rounds=early_stopping_rounds)

        count = cvresult.shape[0]
        mean_rmse = cvresult.loc[count-11 : count-1, 'test-rmse-mean'].mean()
        std_rmse = cvresult.loc[count-11 : count-1, 'test-rmse-std'].mean()

        print('val',val, mean_rmse)
        if not min_rmse:
          min_rmse = mean_rmse+1

        if mean_rmse < min_rmse:
          best_param = val
          best_cvresult = cvresult
          min_rmse = mean_rmse

      best_params[param] = best_param
      all_param[param] = best_param
    return (best_params, min_rmse, best_cvresult)  

  # else:
  #   cvresult = xgb.cv(
  #       all_param, data, num_boost_round=all_param['n_estimators'], 
  #       nfold=cv_folds, metrics='rmse', 
  #       early_stopping_rounds=early_stopping_rounds)
  #   count = cvresult.shape[0]
  #   mean_rmse = cvresult.loc[count-11 : count-1, 'test-rmse-mean'].mean()
  #   std_rmse = cvresult.loc[count-11 : count-1, 'test-rmse-std'].mean()    
  #   best_param, min_rmse, best_cvresult = all_param, mean_rmse, cvresult

  else:
    regressor.fit(data,label)
    predict = regressor.predict(data)
    rmse = np.sqrt(mean_squared_error(label, predict) )

    return regressor.feature_importances_
  

  

# %%
from sklearn.metrics import mean_squared_error

# common_param = {'max_depth': 2, 'eta': 1}

param_tune = {'n_estimator': range(300,400,10)}

regressor = XGBRegressor(
  learning_rate=0.05, max_depth=5, n_estimator=300,
  min_child_weight=1, gamma=0, 
  subsample=0.8, colsample_bytree=0.8,
  reg_lambda=0.1, reg_alpha=0.1,
  object='reg:linear',
  seed=10, n_jobs=12, tree_method='gpu_exact'
)

# best_param, min_rmse, best_cvresult = 
# print(best_param)
# print(best_cvresult)
importance = model_training(regressor, raw_trainX, tune_param=None, label=raw_trainY )

# %%
# dat2 = dat.iloc[:15]
raw_imp = pd.Series(importance, raw_trainX.columns.values).sort_values(ascending=False).head(25)

dat = pd.DataFrame({'x': raw_imp.index, 'y': raw_imp.values}) 
dat['x'] = pd.Categorical(dat['x'], categories=raw_imp.index, ordered=True)

gg = (ggplot(dat)
  + geom_col(aes(x='x',y='y'))
  + theme(axis_text_x=element_text(rotation=70, ha="right"))
)
print(gg)

# %%
dat

# %%

param_tune = {'max_depth': np.arange(1,10,1)}

regressor = XGBRegressor(
  learning_rate=0.2,
  min_child_weight=1, gamma=0, 
  subsample=0.8, colsample_bytree=0.8,
  reg_lambda=0.1, reg_alpha=0.1,
  object='reg:linear',
  seed=10, n_jobs=12
)

best_param, min_rmse, best_cvresult = model_training(regressor, param_tune, train_data)
print(best_param)
print(best_cvresult)

# %%
param_tune = {'min_child_weight': np.arange(1,10,1)}

regressor = XGBRegressor(
  learning_rate=0.2, max_depth=5, 
  gamma=0, 
  subsample=0.8, colsample_bytree=0.8,
  reg_lambda=0.1, reg_alpha=0.1,
  object='reg:linear',
  seed=10, n_jobs=12
)

best_param, min_rmse, best_cvresult = model_training(regressor, param_tune, train_data)
print(best_param)
print(best_cvresult)


# %%
param_tune = {'gamma': np.arange(1,10,1)}

regressor = XGBRegressor(
  learning_rate=0.2, max_depth=5, 
  gamma=0, 
  subsample=0.8, colsample_bytree=0.8,
  reg_lambda=0.1, reg_alpha=0.1,
  object='reg:linear',
  seed=10, n_jobs=12
)

best_param, min_rmse, best_cvresult = model_training(regressor, param_tune, train_data)
print(best_param)
print(best_cvresult)








# %% real prediction

regressor.fit(raw_trainX, raw_trainY)
prediction = np.expm1(regressor.predict(raw_test_df))
prediction_df = pd.DataFrame({'Id': test_id, 'SalePrice': prediction})

prediction_df.to_csv("result/submission.csv", index=False)

# %%


