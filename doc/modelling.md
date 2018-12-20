# Parameter tuning
1. define model_cv
specify early_stopping_round=stop if no improvement for n rounds,
   cv_folds=number of cross-validation
2. start from larger learning rate, then let GridSearchCV auto adjust param
3. first try max_depth, min_child_weight; start from larger region, then converge it
4. then try gamma, and others such as subsample, colsample_bytree, scale_pos_weight, reg_alpha, reg_lambda
5. after all these adjustment, reduce learning rate to try again


n_estimator: 300

## max_depth
best: 5
higher better

## min_child_weight
best: 2
lower better

## 

















