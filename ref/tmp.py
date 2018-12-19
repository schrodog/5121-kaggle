# %%
d1 = pd.DataFrame({'a': [1,2,3,4],'b':[4,5,6,7]})
d2 = pd.DataFrame({'a': [11,12,13],'b':[14,15,16]})

total = pd.concat([d1,d2], keys=['train','test'])

# %%
total['b'] = total['b'].transform(lambda x: x+1)
total

# %%

total[total.index.labels[0] == 0].values







