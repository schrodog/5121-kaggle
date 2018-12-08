# %%
import pandas as pd
import os
import numpy as np
# %%

df2 = pd.DataFrame([
  ['a1',3,5,3],
  ['a2',0,2,3],
  ['a23',8,3,7],
  ['a3',7,3,4]],
  index=range(4),
  columns=['aa','da','af','g'] 
)

df3 = pd.DataFrame({
  'cow': [5,8,3,5],
  'fox': ['a','b','d','k']
})

pd.to_numeric(df2['da'])
# df2['aa'].str.extract(r'^(a2)')
# np.where(df2['da'] >= 2)


# df.assign(Area=lambda df: df.aa*df.da)

# %%
cwd = os.getcwd()
df = pd.read_csv(cwd+'/ref/abc.csv')

# %%

to_drop = ['Edition Statement', 'Corporate Author', 'Corporate Contributors', 'Former owner', 'Engraver', 'Shelfmarks', 'Issuance type', 'Contributors']

df.drop(columns=to_drop, inplace=True)
df.head()

# %%
df.set_index('Identifier', inplace=True)
df.head()

# %%

extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
# return numpy array after .to_numeric
df['Date of Publication'] = pd.to_numeric(extr)

# %%
# lost = df['Date of Publication'].isnull().sum() / len(df)

pub = df['Place of Publication']
london = pub.str.contains('London')
oxford = pub.str.contains('Oxford')
np.where(london, 'London', np.where(oxford, 'Oxford',
          pub.str.replace('-', ' ')))

df['Place of Publication'].head()

# %%
df4 = pd.DataFrame({
  '0': ['Mock', 'Python', 'Real', 'Numpy'],
  '1': ['Dataset', 'Pandas', 'Python', 'Clean']
})

df4













