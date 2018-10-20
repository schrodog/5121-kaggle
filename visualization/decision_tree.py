# %%
import sklearn.datasets as datasets
import pandas as pd

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# %%
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
model = dtree.fit(df, y)

# %%
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True,
  rounded=True, special_characters=True,
  feature_names=iris.feature_names,
  class_names=iris.target_names)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filename = "iris.png"
graph.write_png(filename)
img = mpimg.imread(filename)
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.imshow(img)
plt.show()














