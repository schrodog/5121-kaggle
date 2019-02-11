# %%
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, ElasticNet, Lasso, LassoLars
from plotnine import *
import warnings
# warnings.filterwarnings('ignore')
# import socket; socket.gethostname()

# %%
raw_train_df = pd.read_csv('result/new_train8.csv')
raw_test_df = pd.read_csv('result/new_test8.csv')

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

  early_stopping_rounds = 120
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

  else:
    regressor.fit(data,label)
    predict = regressor.predict(data)
    rmse = np.sqrt(mean_squared_error(label, predict) )

    return regressor.feature_importances_
  

  

# %%

# common_param = {'max_depth': 2, 'eta': 1}

param_tune = {
  'max_depth': np.arange(1,8,1), 
  'gamma': np.arange(0.01,0.05, 0.01), 
  'min_child_weight': np.arange(3,10,1),
  'reg_lambda': np.arange(0.03,0.09, 0.01),
  'reg_alpha': np.arange(0.02,0.09, 0.01),
}

regressor = XGBRegressor(
  learning_rate=0.1, n_estimator=300,
  max_depth=7, gamma=0.01, min_child_weight=3,
  subsample=0.8, colsample_bytree=0.5,
  reg_lambda=0.03, reg_alpha=0.02,
  object='reg:linear',
  seed=10, n_jobs=12
)

# importance = model_training(regressor, raw_trainX, tune_param=None, label=raw_trainY )
model_training(regressor, train_data, tune_param=param_tune )

# %%
# dat2 = dat.iloc[:15]
raw_imp = pd.Series(importance, raw_trainX.columns.values).sort_values(ascending=False).head(50)

dat = pd.DataFrame({'x': raw_imp.index, 'y': raw_imp.values}) 
dat['x'] = pd.Categorical(dat['x'], categories=raw_imp.index, ordered=True)

gg = (ggplot(dat)
  + geom_col(aes(x='x',y='y'))
  + scale_x_discrete(name="Features")
  + scale_y_continuous(name="Feature Importance")
  # + geom_text(aes(x='x',y='y',label='y'), position=position_stack(vjust=5))
  + theme(axis_text_x=element_text(rotation=70, ha="right"))
)
print(gg)
# %%

gg.save('result/feature_importance.png')

# %%

np.array(dat['x'].values[:50])



# %% real prediction ============================================================================================
# if missing features error: cause by different feature order in test and train data
param_tune = {'n_estimator': range(300,308,10)}

regressor = XGBRegressor(
  learning_rate=0.1, n_estimator=300,
  max_depth=7, gamma=0.01, min_child_weight=3,
  subsample=0.8, colsample_bytree=0.5,
  reg_lambda=0.03, reg_alpha=0.02,
  object='reg:linear',
  seed=10, n_jobs=12
)

best_params, xgb_err, best_cvresult = model_training(regressor, train_data, tune_param=param_tune)
print(xgb_err)

regressor.fit(raw_trainX, raw_trainY)
xgb_pred = np.expm1(regressor.predict(raw_test_df))

# xgb_prediction_df = pd.DataFrame({'Id': test_id, 'SalePrice': xgb_predictions})
# prediction_df.to_csv("result/submission6.csv", index=False)

# %%

trainX, testX, trainY, testY = train_test_split(raw_trainX, raw_trainY, test_size=0.4, random_state=0)

def general_model(model, params):
  min_err, best_param = float('inf'), None
  for p in params:
    m = model(p, max_iter=50000).fit(trainX, trainY)
    pred = m.predict(testX)
    err = np.sqrt(mean_squared_error(pred, testY))
    # print("%.4f, %.6f" %(p,err))
    if err < min_err:
      best_param = p
      min_err = err    
  return (best_param, min_err)
# %%
alpha_list, err_list = [],[]

lasso_test = np.arange(0.0001, 0.02, 0.0001)
lasso_alpha, lasso_err = general_model(Lasso, lasso_test)
alpha_list.append(lasso_alpha); err_list.append(lasso_err)

lasso_alpha, lasso_err


elas_test = np.arange(0.0001, 0.02, 0.0001)
elas_alpha, elas_err = general_model(ElasticNet, elas_test)
alpha_list.append(elas_alpha); err_list.append(elas_err)

elas_alpha, elas_err


ridge_test = np.arange(30, 300, 5)
ridge_alpha, ridge_err = general_model(Ridge, ridge_test)
alpha_list.append(ridge_alpha); err_list.append(ridge_err)

ridge_alpha, ridge_err


lars_test = np.arange(0.0001, 0.002, 0.0001)
lars_alpha, lars_err = general_model(LassoLars, lars_test)
alpha_list.append(lars_alpha); err_list.append(lars_err)

lars_alpha, lars_err

## 
max_iter = 50000 

lasso_model = Lasso(alpha=alpha_list[0], max_iter=max_iter).fit(trainX, trainY)
elasticNet_model = ElasticNet(alpha=alpha_list[1], max_iter=max_iter).fit(trainX, trainY)
ridge_model = Ridge(alpha=alpha_list[2], max_iter=max_iter).fit(trainX, trainY)
lars_model = LassoLars(alpha=alpha_list[3], max_iter=max_iter).fit(trainX, trainY)

lasso_pred = np.expm1(lasso_model.predict(raw_test_df))
ridge_pred = np.expm1(ridge_model.predict(raw_test_df))
elasticNet_pred = np.expm1(elasticNet_model.predict(raw_test_df))
lars_pred = np.expm1(lars_model.predict(raw_test_df))
pred_list = np.array([lasso_pred, ridge_pred, elasticNet_pred, lars_pred, xgb_pred])

# take average of 4 models
err_list.append(xgb_err)
err_list = np.array(err_list)

w_list = 1/err_list
total_w = np.sum(w_list)
predictions = np.matmul(w_list/total_w, pred_list)

# xgb_w, lasso_w, elas_w, ridge_w, lars_w = 1/xgb_err, 1/lasso_err, 1/elas_err, 1/ridge_err, 1/lars_err
# total_w = xgb_w + lasso_w + elas_w + ridge_w + lars_w
# predictions = lasso_w/total_w*lasso_pred + ridge_w/total_w*ridge_pred + \
#   elas_w/total_w*elasticNet_pred + xgb_w/total_w*xgb_pred + 


final_sub = pd.DataFrame({
        "Id": test_id,
        "SalePrice": predictions
    })
final_sub.to_csv("result/4_models4.csv", index=False)

# %%
w_list


