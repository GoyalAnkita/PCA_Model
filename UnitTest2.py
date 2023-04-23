import numpy as np
import pandas as pd
from Implementation import PCA
from Implementation import create_visualization

data_reg=pd.read_fwf( 'housing.data',sep=" ",names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MDEV'])


# In[13]:


data_reg.head()


# In[14]:


X=data_reg.drop(['MDEV'],axis=1)
y=data_reg['MDEV']


# In[15]:


pca=PCA(n_components=4).fit(X)
print('Explained variance ratio from scratch:\n', pca.explained_variance_ratio)
print('Cumulative explained variance from scratch:\n', pca.cum_explained_variance)

X_proj = pca.transform(X)
print('Transformed data shape from scratch:', X_proj.shape)

'''Functionality of find_components: If a business problem requires us to retain 80 % of variance for example, 
then this function can be used to identify how many components are required to retain & explain 80% of variance'''


pca.find_components(explainability=0.85) # Explainability: convert percentage into decimal
pca.find_components(explainability=0.95)


# In[16]:


#viz.create_cummulativeplot(X)



# In[17]:


viz=create_visualization(n_components=2,isClassification=False)
viz.create_scatterplots(X)


# In[18]:


viz=create_visualization(n_components=3,isClassification=False)
viz.create_scatterplots(X)

viz.create_cummulativeplot(X)


