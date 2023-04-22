import numpy as np
import pandas as pd
from Implementation import PCA
from Implementation import create_visualization


data_cat = pd.read_csv("winequality-white.csv",sep=';')


# In[5]:


data_cat


# In[6]:


data_cat.isnull().sum()


# In[7]:


X=data_cat.drop(['quality'],axis=1)
y=data_cat['quality']

y


# In[8]:


pca=PCA(n_components=4).fit(X)
print('Explained variance ratio:\n', pca.explained_variance_ratio)
print('Cumulative explained:\n', pca.cum_explained_variance)

X_proj = pca.transform(X)
print('Transformed data shape:', X_proj.shape)

'''Functionality of find_components: If a business problem requires us to retain 80 % of variance for example, 
then this function can be used to identify how many components are required to retain & explain 80% of variance'''


pca.find_components(explainability=0.80) # Explainability: convert percentage into decimal
pca.find_components(explainability=0.90)


# In[9]:


viz=create_visualization(n_components=3,isClassification=True)
viz.create_scatterplots(X,y)


# In[10]:


viz=create_visualization(n_components=2,isClassification=True)
viz.create_scatterplots(X,y)


# In[11]:


viz.create_cummulativeplot(X)


