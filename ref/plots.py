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

# %%
df = pd.DataFrame({
  'group': range(3), #["Male", "Female", "Child"],
  'value': [25, 25, 50],
  'all': [1]*3
})

(ggplot(df)
  + geom_col(aes(x='group', y='value'))
  + coord_flip()
)

# %%
from random import randint

n = 100

df = pd.DataFrame({
  'x': [randint(0,100)/10 for _ in range(n)],
  'y': [randint(0,100)/10 for _ in range(n)],
})

df['bb'] = np.array(df['x']) + np.array(df['y'])
df.columns
df.values

# %%

(ggplot(df, aes(x="x", y='y', size="bb"))
  + geom_point()
)


# %%
from plotnine import * 
from plotnine.data import *
import pandas as pd
import datetime
from mizani.breaks import date_breaks
from mizani.formatters import date_format

economics.head()
# %%

df = pd.melt(economics, id_vars=['date'], value_vars=['psavert', 'uempmed']) 
df

# %%
p = ggplot(df, aes(x='date', y='value', color='variable'))
(p + geom_line()
   + scale_x_datetime(breaks=date_breaks('10 years'),labels=date_format('%Y'))
   + scale_color_manual(['r', 'b'])
)






