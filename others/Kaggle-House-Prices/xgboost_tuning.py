
# coding: utf-8

# In[1]:


# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import xgboost as xgb  #GBM algorithm
from xgboost import XGBRegressor
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   # Perforing grid search

from IPython.display import display

# remove warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train_data = pd.read_csv('data/new_train.csv')
test_data = pd.read_csv('data/new_test.csv')

print(train_data.shape)
display(train_data.head(3))
display(train_data.info())

print(test_data.shape)
display(test_data.head(3))
# display(test_data.info())
train_length = train_data.shape[0]


# In[3]:


import math

# to create range function support any value
def common_num_range(start,stop,step):
    
    startlen = stoplen = steplen = 0
    if '.' in str(start):
        startlen = len(str(start)) - str(start).index('.') - 1
    if '.' in str(stop):
        stoplen = len(str(stop)) - str(stop).index('.') - 1
    if '.' in str(step):
        steplen = len(str(step)) - str(step).index('.') - 1
    
    maxlen = startlen
    if stoplen > maxlen:
        maxlen = stoplen
    if steplen > maxlen:
        maxlen = steplen
    
    power = math.pow(10, maxlen)
    
    if startlen == 0 and stoplen == 0 and steplen == 0:
        return range(start, stop, step)
    else:
        return [num / power for num in range(int(start*power), int(stop*power), int(step*power))]


# In[4]:


train_id = train_data['Id']
train_Y = train_data['SalePrice']
train_data.drop(['Id', 'SalePrice'], axis=1, inplace=True)
train_X = train_data

test_Id = test_data['Id']
test_data.drop('Id', axis=1, inplace=True)
test_X = test_data


# In[5]:


# formatting for xgb
# DMatrix = internal data structure used by XGBoost, optimize both memory and training speed
dtrain = xgb.DMatrix(train_X, label=train_Y, nthread=-1)
dtest = xgb.DMatrix(test_X, nthread=-1)


# # XGBoost  & Parameter Tuning
# 
# Ref: [Complete Guide to Parameter Tuning in XGBoost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

# ## Parameters Tuning Plan
# 
# The overall parameters can be divided into 3 categories:
# 
# 1. General Parameters: Guide the overall functioning
# 2. Booster Parameters: Guide the individual booster (tree/regression) at each step
# 3. Learning Task Parameters: Guide the optimization performed
# 
# In `XGBRegressor`:
# ```
# class xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
# ```

# In[6]:


# The error metric: RMSE on the log of the sale prices.
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# In[7]:


def model_cross_validate(xgb_regressor, cv_paramters, dtrain, cv_folds = 5,
              early_stopping_rounds = 50, perform_progress=False):
    """
    xgb model cross validate to choose best param from giving cv_paramters.
    
    @param cv_paramters:dict,where to choose best param. {'param':[1,2,3]}
    @param dtrain:xgboost.DMatrix, training data formatted for xgb
    @param early_stopping_rounds: Activates early stopping.Stop when perfomance 
                                  does not improve for some rounds
    """
    # get initial parameters
    xgb_param = xgb_regressor.get_xgb_params()
    
    # save best param
    best_params = {}
    best_cvresult = None
    min_mean_rmse = float("inf")

    # 'n_estimator', 300 to 400
    for param, values in cv_paramters.items():
        print ('===========Tuning paramter:',param,'===========')
        best_param = values[0]
        for value in values:
            # set the param's value
            xgb_param[param] = value
            
            # cv to tune param from values, cv=cross validation
            cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=xgb_param['n_estimators'], 
                              nfold=cv_folds, metrics='rmse', 
                              early_stopping_rounds=early_stopping_rounds,
                              predictor='gpu_predictor', tree_method='gpu_exact')
            # print('cvresult', cvresult)                              

            # calcuate the mean of last 10 rows rmses
            round_count = cvresult.shape[0]
            mean_rmse = cvresult.loc[round_count-11:round_count-1,'test-rmse-mean'].mean()
            
            if perform_progress:
                std_rmse = cvresult.loc[round_count-11:round_count-1,'test-rmse-std'].mean()

                if isinstance(value, int):
                    print( "%s=%d CV RMSE : Mean = %.7g | Std = %.7g" % (param, value, mean_rmse, std_rmse))
                else:
                    print ("%s=%f CV RMSE : Mean = %.7g | Std = %.7g" % (param, value, mean_rmse, std_rmse))
            
            if mean_rmse < min_mean_rmse:
                best_param = value
                best_cvresult = cvresult
                min_mean_rmse = mean_rmse
        
        best_params[param] = best_param
        # set best param value for xgb params, important
        xgb_param[param] = best_param
        print ("best ", param, " = ", best_params[param])
    
    return best_params, min_mean_rmse, best_cvresult


# In[8]:


def model_fit(xgb_regressor, train_x, train_y, performCV=True, 
              printFeatureImportance=True, cv_folds=5):
    
    # Perform cross-validation
    if performCV:
        xgb_param = xgb_regressor.get_xgb_params()
        cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=xgb_param['n_estimators'], 
                              nfold=cv_folds, metrics='rmse', 
                              early_stopping_rounds=50)
        round_count = cvresult.shape[0]
        mean_rmse = cvresult.loc[round_count-11:round_count-1,'test-rmse-mean'].mean()
        std_rmse = cvresult.loc[round_count-11:round_count-1,'test-rmse-std'].mean()
        
        print ("CV RMSE : Mean = %.7g | Std = %.7g" % (mean_rmse, std_rmse))
        
    # fir the train data
    xgb_regressor.fit(train_x, train_y)
    
    # Predict training set
    train_predictions = xgb_regressor.predict(train_x)
    mse = rmse(train_y, train_predictions)
    print("Train RMSE: %.7f" % mse)
    
    # Print Feature Importance
    if printFeatureImportance:
        feature_importances = pd.Series(xgb_regressor.feature_importances_, train_x.columns.values)
        feature_importances = feature_importances.sort_values(ascending=False)
        feature_importances= feature_importances.head(40)
        feature_importances.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
    
    return xgb_regressor, feature_importances


# Baseline XGBRegressor

# In[9]:


xgb_regressor = XGBRegressor(seed=10)
xgb_regressor, feature_importances = model_fit(xgb_regressor,train_X, train_Y)


# ### 1. Choose a relatively high learning_rate, optimum n_estimators

# In[10]:

param_test = {'n_estimators':range(300,400,10)}

xgb_regressor = XGBRegressor(
                learning_rate =0.05,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda = 0.1,
                reg_alpha = 0.1,
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                n_jobs=12)

best_param, min_mean_rmse, best_cvresult = model_cross_validate(xgb_regressor, param_test, dtrain, perform_progress=True)

print ('cross-validate best params:', best_param)
print ('cross-validate min_mean_rmse:', min_mean_rmse)


# In[11]:


xgb_regressor = XGBRegressor(
                learning_rate =0.05,
                n_estimators = 300,

                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda = 0.1,
                reg_alpha = 0.1,
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                n_jobs=12)

xgb_regressor, feature_importances = model_fit(xgb_regressor,train_X, train_Y)


# ### 2.Fix learning rate and number of estimators for tuning tree-based parameters

# Tune `max_depth` and `min_child_weight`

# In[12]:


param_test = {'max_depth':range(1,6,1),
               'min_child_weight':common_num_range(1,2,0.1)}

xgb_regressor = XGBRegressor(
                learning_rate =0.05,
                n_estimators = 300,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda = 0.1,
                reg_alpha = 0.1,
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                n_jobs=12)

best_param, min_mean_rmse, best_cvresult = model_cross_validate(xgb_regressor, param_test, dtrain, perform_progress=True)

print ('cross-validate best params:', best_param)
print ('cross-validate min_mean_rmse:', min_mean_rmse)


# In[13]:


xgb_regressor = XGBRegressor(
                learning_rate =0.05,
                n_estimators = 300,
                max_depth=3,
                min_child_weight=1.1,

                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda = 0.1,
                reg_alpha = 0.1,
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                n_jobs=12)

xgb_regressor, feature_importances = model_fit(xgb_regressor,train_X, train_Y)


# Tune `gamma`,Minimum loss reduction required to make a further partition on a leaf node of the tree. 

# In[14]:


param_test = {'gamma':[0, 0.1, 0.01, 0.001,0.0001, 0.00001]}

xgb_regressor = XGBRegressor(
                learning_rate =0.05,
                n_estimators = 300,
                max_depth=3,
                min_child_weight=1.1,
    
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda = 0.1,
                reg_alpha = 0.1,
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                n_jobs=12)

best_param, min_mean_rmse, best_cvresult = model_cross_validate(xgb_regressor, param_test, dtrain, perform_progress=True)

print ('cross-validate best params:', best_param)
print ('cross-validate min_mean_rmse:', min_mean_rmse)


# In[15]:


xgb_regressor = XGBRegressor(
                learning_rate =0.05,
                n_estimators = 300,
                max_depth=3,
                min_child_weight=1.1,
                gamma=0.01,
    
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda = 0.1,
                reg_alpha = 0.1,
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                n_jobs=12)

xgb_regressor, feature_importances = model_fit(xgb_regressor,train_X, train_Y)


# Tune `subsample` and `colsample_bytree`
# 
# - subsample : Subsample ratio of the training instance.
# - colsample_bytree : Subsample ratio of columns when constructing each tree

# In[16]:


param_test = {'subsample':common_num_range(0.6, 0.9, 0.01),
               'colsample_bytree':common_num_range(0.6, 0.9, 0.01)}

xgb_regressor = XGBRegressor(
                learning_rate =0.05,
                n_estimators = 300,
                max_depth=3,
                min_child_weight=1.1,
                gamma=0.01,
    
                reg_lambda = 0.1,
                reg_alpha = 0.1,
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                predictor='gpu_predictor',
                tree_method='gpu_exact')

best_param, min_mean_rmse, best_cvresult = model_cross_validate(xgb_regressor, param_test, dtrain, perform_progress=True)

print ('cross-validate best params:', best_param)
print ('cross-validate min_mean_rmse:', min_mean_rmse)


# In[17]:


xgb_regressor = XGBRegressor(
                learning_rate =0.05,
                n_estimators = 300,
                max_depth=3,
                min_child_weight=1.1,
                gamma=0.01,
                subsample=0.72,
                colsample_bytree=0.89,
    
                reg_lambda = 0.1,
                reg_alpha = 0.1,
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                predictor='gpu_predictor',
                tree_method='gpu_exact')

xgb_regressor, feature_importances = model_fit(xgb_regressor,train_X, train_Y)


# In[18]:


param_test2 = {'reg_lambda':common_num_range(0.55, 0.65, 0.01),
               'reg_alpha':common_num_range(0.45, 0.6, 0.01)}

xgb_regressor = XGBRegressor(
                learning_rate =0.05,
                n_estimators = 300,
                max_depth=3,
                min_child_weight=1.1,
                gamma=0.01,
                subsample=0.72,
                colsample_bytree=0.89,
    
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                predictor='gpu_predictor',
                tree_method='gpu_exact')

best_param, min_mean_rmse, best_cvresult = model_cross_validate(xgb_regressor, param_test2, dtrain, perform_progress=True)

print ('cross-validate best params:', best_param)
print ('cross-validate min_mean_rmse:', min_mean_rmse)


# In[19]:


xgb_regressor = XGBRegressor(
                learning_rate =0.05,
                n_estimators = 300,
                max_depth=3,
                min_child_weight=1.1,
                gamma=0.01,
                subsample=0.72,
                colsample_bytree=0.89,
                reg_lambda = 0.61,
                reg_alpha = 0.53,
    
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                predictor='gpu_predictor',
                tree_method='gpu_exact')

xgb_regressor, feature_importances = model_fit(xgb_regressor,train_X, train_Y)                                           


# In[20]:


xgb_regressor = XGBRegressor(
                learning_rate =0.01,
                n_estimators = 4000,
                max_depth=3,
                min_child_weight=1.1,
                gamma=0.01,
                subsample=0.72,
                colsample_bytree=0.89,
                reg_lambda = 0.61,
                reg_alpha = 0.53,
    
                scale_pos_weight=1,
                objective= 'reg:linear',
                seed=10,
                predictor='gpu_predictor',
                tree_method='gpu_exact')

xgb_regressor, feature_importances = model_fit(xgb_regressor,train_X, train_Y)                                           


# Final paramters:
# 
# ```
# xgb_regressor = XGBRegressor(
#                 learning_rate =0.01,
#                 n_estimators = 4000,
#                 max_depth=3,
#                 min_child_weight=1.1,
#                 gamma=0.01,
#                 subsample=0.72,
#                 colsample_bytree=0.89,
#                 reg_lambda = 0.61,
#                 reg_alpha = 0.53,
#     
#                 scale_pos_weight=1,
#                 objective= 'reg:linear',
#                 seed=10)
# ```

# In[21]:


xgb_predictions = xgb_regressor.predict(test_X)
xgb_predictions = np.expm1(xgb_predictions)

submission = pd.DataFrame({
        "Id": test_Id,
        "SalePrice": xgb_predictions
    })

submission.to_csv("result2/xgb_param_tune_predictions_2_13.csv", index=False)

print ("Done.")


# # Model Voting
# 
# Ridge, ElasticNet, Lasso, XGBRegressor model voting.

# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, 
                                                    test_size=0.4, random_state=0)


# In[23]:


from sklearn.linear_model import Ridge, ElasticNet, Lasso

def simple_model_cross_validate(alphas, Model, model_name):
    min_rmse = float('inf')
    best_alpha = None
    for alpha in alphas:
        model = Model(alpha, max_iter=50000).fit(X_train, y_train)
        model_rmse = rmse(model.predict(X_test), y_test)
        if model_rmse < min_rmse:
            best_alpha = alpha
            min_rmse = model_rmse

    print (model_name, 'best_alpha = ', best_alpha, 'min_rmse = ', min_rmse)

alphas = common_num_range(0.0001, 0.002, 0.0001)
simple_model_cross_validate(alphas, Lasso, 'Lasso')
simple_model_cross_validate(alphas, ElasticNet, 'ElasticNet')


# In[24]:

alphas = common_num_range(25, 50, 1)
simple_model_cross_validate(alphas, Ridge, 'Ridge')


# In[25]:

lasso_model = Lasso(alpha=0.0009, max_iter=50000).fit(X_train, y_train)
elastic_net_model = ElasticNet(alpha=0.0019, max_iter=50000).fit(X_train, y_train)
ridge_model = Ridge(alpha=41, max_iter=50000).fit(X_train, y_train)


# In[26]:


lasso_predictions = lasso_model.predict(test_X)
lasso_predictions = np.expm1(lasso_predictions)

ridge_predictions = ridge_model.predict(test_X)
ridge_predictions = np.expm1(ridge_predictions)

elastic_net_predictions = elastic_net_model.predict(test_X)
elastic_net_predictions = np.expm1(elastic_net_predictions)


# In[27]:

# take average of 4 models
predictions = (lasso_predictions + ridge_predictions + elastic_net_predictions + xgb_predictions) / 4
predictions

# In[28]:


plt.subplot(221)
plt.plot(lasso_predictions, c="blue")  # 0.12818
plt.title('lasso 0.12818')
plt.subplot(222)
plt.plot(elastic_net_predictions, c="yellow")  # 0.12908
plt.title('elastic_net 0.12908')
plt.subplot(223)
plt.plot(ridge_predictions, c="pink")  # 0.13161
plt.title('ridge 0.13161')
plt.subplot(224)
plt.plot(xgb_predictions, c="green")  # 0.12167
plt.title('xgb 0.12167')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
plt.subplot(111)
plt.plot(predictions, c="red")  # 0.12419
plt.title('4 model vote 0.12419')


# In[29]:


# outlier data
np.argwhere(xgb_predictions == xgb_predictions[xgb_predictions > 700000])


# In[30]:


# convert outlier data to xgb_predictions[1089]
lasso_predictions[1089] =  xgb_predictions[1089]
ridge_predictions[1089] =  xgb_predictions[1089]
elastic_net_predictions[1089] =  xgb_predictions[1089]


# In[31]:


lasso_score = 1-0.12818
ridge_score = 1-0.13161
elastic_net_score = 1-0.12908
xgb_score = 1-0.12167
total_score = lasso_score + ridge_score + elastic_net_score + xgb_score
predictions = (lasso_score / total_score) * lasso_predictions + \
                (ridge_score / total_score) * ridge_predictions +\
                (elastic_net_score / total_score) * elastic_net_predictions +\
                (xgb_score / total_score) * xgb_predictions


# In[32]:


plt.subplot(221)
plt.plot(lasso_predictions, c="blue")  # 0.12818
plt.title('lasso 0.12818')
plt.subplot(222)
plt.plot(elastic_net_predictions, c="yellow")  # 0.12908
plt.title('elastic_net 0.12908')
plt.subplot(223)
plt.plot(ridge_predictions, c="pink")  # 0.13161
plt.title('ridge 0.13161')
plt.subplot(224)
plt.plot(xgb_predictions, c="green")  # 0.12167
plt.title('xgb 0.12167')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
plt.subplot(111)
plt.plot(predictions, c="red")  # 0.12417
plt.title('4 model vote 0.12417')


# In[33]:


submission = pd.DataFrame({
        "Id": test_Id,
        "SalePrice": lasso_predictions
    })
submission.to_csv("result2/lasso_predictions_2_13.csv", index=False)

submission = pd.DataFrame({
        "Id": test_Id,
        "SalePrice": ridge_predictions
    })
submission.to_csv("result2/ridge_predictions_2_13.csv", index=False)

submission = pd.DataFrame({
        "Id": test_Id,
        "SalePrice": elastic_net_predictions
    })
submission.to_csv("result2/elastic_net_predictions_2_13.csv", index=False)

submission = pd.DataFrame({
        "Id": test_Id,
        "SalePrice": xgb_predictions
    })
submission.to_csv("result2/xgb_predictions_2_13.csv", index=False)

submission = pd.DataFrame({
        "Id": test_Id,
        "SalePrice": predictions
    })
submission.to_csv("result2/4_model_vote_predictions_2_13.csv", index=False)

print ("Done.")


# # Best Vote Score

# In[34]:


from sklearn.linear_model import LassoCV

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train_X, train_Y)
print( rmse(model_lasso.predict(train_X), train_Y))
lasso_predictions = model_lasso.predict(test_X)
lasso_predictions = np.expm1(lasso_predictions)


# In[35]:


predictions = (lasso_predictions + xgb_predictions) / 2
submission = pd.DataFrame({
        "Id": test_Id,
        "SalePrice": predictions
    })
submission.to_csv("result2/lasso_xgb_vote_predictions_2_13.csv", index=False)

print ("Done.")


# In[36]:


lasso_score = 1-0.12818
xgb_score = 1-0.12167
total_score = lasso_score + xgb_score
predictions = (lasso_score / total_score) * lasso_predictions +               (xgb_score / total_score) * xgb_predictions
            
submission = pd.DataFrame({
        "Id": test_Id,
        "SalePrice": predictions
    })
submission.to_csv("result2/lasso_xgb_weighted_vote_predictions_2_13.csv", index=False)

print( "Done.")

