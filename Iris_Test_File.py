import numpy as np
import pandas as pd
from PCA_Model import PCA
from PCA_Model import create_visualization


df=pd.read_table('Iris.xls',sep=',')
df.drop(['Id'],axis=1,inplace=True)
X=df.drop(['Species'],axis=1)
y=df['Species']
y=y.map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
y.unique()
df.head()


# In[20]:


pca=PCA(n_components=4).fit(X)
print('Explained variance ratio:\n', pca.explained_variance_ratio)
print('Cumulative explained variance from scratch:\n', pca.cum_explained_variance)

X_proj = pca.transform(X)
print('Transformed data shape from scratch:', X_proj.shape)

pca.find_components(explainability=0.80) # Explainability: convert percentage into decimal


# In[21]:


viz=create_visualization(n_components=2,isClassification=True)
viz.create_scatterplots(X,y=y)


# In[22]:


viz.create_cummulativeplot(X)

viz=create_visualization(n_components=3,isClassification=False)
viz.create_scatterplots(X)

