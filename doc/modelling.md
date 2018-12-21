# Parameter tuning
1. define model_cv
specify early_stopping_round=stop if no improvement for n rounds,
   cv_folds=number of cross-validation
2. start from larger learning rate, then let GridSearchCV auto adjust param
3. first try max_depth, min_child_weight; start from larger region, then converge it
4. then try gamma, and others such as subsample, colsample_bytree, scale_pos_weight, reg_alpha, reg_lambda
5. after all these adjustment, reduce learning rate to try again


## xgb parameters
n_estimator: 300
max_depth: 4
min_child_weight: 8
reg_lambda: 0.06
reg_alpha: 0.04
objective function: linear

## other models
Lasso
Ridge
ElasticNet

## Ensemble model
1. get rmse for training dataset
2. weight = 1/error
3. weighted sum of predictions from 4 models


## features
obvious improvement when restrict fields to most related 50

200 features: 0.26
40 features: 0.2376
50 features: 0.228
60 features: 0.233091

clipped data: 0.227593
clipped upper,lower: 0.225630

0.146653












