# %%
from sklearn.datasets import make_regression
reg_data, reg_target = make_regression(n_samples=200, n_features=500, n_informative=5, noise=25 )
reg_data

# %%
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(reg_data, reg_target)

# %%
import numpy as np
np.sum(lasso.coef_ != 0)

# %%
lasso_0 = Lasso(1)
lasso_0.fit(reg_data, reg_target)
# np.sum(lasso_0.coef_ != 0)

np.where(lasso_0.coef_ != 0)

# %%
import os

print(os.getcwd())



















