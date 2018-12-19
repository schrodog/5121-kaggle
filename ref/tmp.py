# %%
data = pd.DataFrame({
  "latitude": {'Blmngtn' : 42.062806,'Blueste' :42.009408,'BrDale' : 42.052500,'BrkSide':42.033590,'ClearCr': 42.025425,'CollgCr':42.021051,'Crawfor': 42.025949,'Edwards':42.022800,'Gilbert': 42.027885,'GrnHill':42.000854,'IDOTRR' : 42.019208,'Landmrk':42.044777,'MeadowV': 41.991866,'Mitchel':42.031307,'NAmes'  : 42.042966,'NoRidge':42.050307,'NPkVill': 42.050207,'NridgHt':42.060356,'NWAmes' : 42.051321,'OldTown':42.028863,'SWISU'  : 42.017578,'Sawyer' :42.033611,'SawyerW': 42.035540,'Somerst':42.052191,'StoneBr': 42.060752,'Timber' :41.998132,'Veenker': 42.040106},

  "longitude": {'Blmngtn' : -93.639963,'Blueste' : -93.645543,'BrDale' : -93.628821,'BrkSide': -93.627552,'ClearCr': -93.675741,'CollgCr': -93.685643,'Crawfor': -93.620215,'Edwards': -93.663040,'Gilbert': -93.615692,'GrnHill': -93.643377,'IDOTRR' : -93.623401,'Landmrk': -93.646239,'MeadowV': -93.602441,'Mitchel': -93.626967,'NAmes'  : -93.613556,'NoRidge': -93.656045,'NPkVill': -93.625827,'NridgHt': -93.657107,'NWAmes' : -93.633798,'OldTown': -93.615497,'SWISU'  : -93.651283,'Sawyer' : -93.669348,'SawyerW': -93.685131,'Somerst': -93.643479,'StoneBr': -93.628955,'Timber' : -93.648335,'Veenker': -93.657032}
})

data
# %%
from sklearn.cluster import KMeans

coords = data[['latitude','longitude']].values
kmeans = KMeans(n_clusters=5, random_state=2).fit(coords)

data['classes'] = pd.Series(kmeans.labels_, index=data.index)

gg = (ggplot(data, aes(x='latitude', y='longitude', color='classes', size='classes'))
  + geom_point()
  # + scale_fill_hue(expand=range(10))
  # + scale_colour_manual(values = ["red", "blue", "green"])
  # + geom_col()
  # + geom_bar()
  # + stat_count(aes(label='stat(count)'), geom='text', position=position_stack(vjust=1.05))
  # + geom_point()
  # + geom_histogram(binwidth=10)
  # + facet_wrap('Neighborhood')
  # + scale_y_continuous(breaks=range(1850, 2020, 10) )
  # + coord_cartesian(ylim=(1900,2010))
  # + theme(axis_text_x=element_text(rotation=0, ha="right"))
)
print(gg)
# gg.save('result/neighbor_dist.png')

# %%
np.column_stack((data.index, kmeans.labels_))












