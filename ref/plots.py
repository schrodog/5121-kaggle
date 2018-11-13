# %%
import pandas as pd
import numpy as np
from plotnine import *

%matplotlib inline

# %%
n = 12
df = pd.DataFrame({'x': np.arange(n), 'y': np.arange(n),
                  'yfit': np.arange(n) + np.tile([-.2, .2], n//2),
                  'cat': ['a', 'b', 'c']*(n//3) })

(ggplot(df)
  + geom_col(aes('x', 'y', fill='cat'))
  + geom_point(aes('x', y='yfit', color='cat' ))
  + geom_path(aes('x', y='yfit', color='cat'))
  + scale_color_discrete(l=.4)
  + guides(
    fill=guide_legend(title='Data'),
    color=guide_legend(title='Model')
  )
)

# %%
df = pd.DataFrame({
  'letter': ['Alpha', 'Beta', 'Delta', 'Gamma'],
  'pos': [1,2,3,4],
  'num_of_letters': [5,4,5,5]
})

# %%
(ggplot(df)
  + geom_col(aes(x='letter', y='pos'))
  + geom_line(aes(x='letter', y='num_of_letters'), group=1)
  + ggtitle('Greek Letter Analysis')
)

# %%
df2 = pd.DataFrame({
  'letter': ['Alpha', 'Beta', 'Delta', 'Gamma']*2 , 
  'pos': [1,2,3,4]*2,
  'num_of_letters': [5,4,5,5]*2
})

df2.loc[4:, 'num_of_letters'] += 0.8
df2
# %%

(ggplot(df2)
  + geom_col(aes(x='letter', y='pos'))
  + geom_line(aes(x='letter', y='num_of_letters'))
)






