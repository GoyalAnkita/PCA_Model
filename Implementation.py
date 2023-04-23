# - Implementing custom made PCA model on WineQuality Dataset, which is a muliclass classification type of dataset

# In[4]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PCA_Model import PCA
from PCA_Model import create_visualization
#from PCA import create_visualization

"""data_cat = pd.read_csv("winequality-white.csv",sep=';')


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


# - Implementing custom made PCA model on Boston Housing Dataset, which is a regression type of dataset

# In[12]:


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


viz.create_cummulativeplot(X)


# In[17]:


viz=create_visualization(n_components=2,isClassification=False)
viz.create_scatterplots(X)


# In[18]:


viz=create_visualization(n_components=3,isClassification=False)
viz.create_scatterplots(X)


# - Implementing custom made PCA model on Boston Housing Dataset, which is a binary classification type of dataset

# In[19]:


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


# In[22]:





# In[22]:





# In[22]:





# In[22]:





# In[22]:"""




