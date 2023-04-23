#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components   
        self.eigen_vec_sorted=[]
      
    def fit(self, X):
        X=X.to_numpy()
        # Standardize data 
        X = X.copy()
        self.mean = np.mean(X, axis = 0)
        self.scale = np.std(X, axis = 0)
        self.X_std = (X - self.mean) / self.scale
        
        # Eigendecomposition of covariance matrix       
        cov_mat = np.cov(self.X_std.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat) 
        
        # Adjusting the eigenvectors that are largest in absolute value to be positive    
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
        eig_vecs = eig_vecs*signs[np.newaxis,:]
        eig_vecs = eig_vecs.T
       
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i,:]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])
        self.eigen_vec_sorted=eig_vecs_sorted
        self.components = eig_vecs_sorted[:self.n_components,:]
        
        # Explained variance ratio
        self.explained_variance_ratio = [i/np.sum(eig_vals) for i in eig_vals_sorted[:]]
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

        return self

    def transform(self, X):
        X = X.copy()
        X_std = (X - self.mean) / self.scale
        X_proj = X_std.dot(self.components.T)
        
        return X_proj
    
    def find_components(self,explainability):
      #print('Cummulative explained variance: ',self.cum_explained_variance)
      explained_variance=0
      for index,val in enumerate(np.round(self.cum_explained_variance,2)):
        #explained_variance+=val
        if val>explainability:
          break
      print('To retain and explain {} % of the variance, you must consider {} principal components'.format((explainability)*100,index+1))


# In[3]:


class create_visualization(PCA):

  def __init__(self,isClassification=False,**kwargs):
    super().__init__(**kwargs)
    self.isClassification=isClassification

  def plot_2D(self,X,y=None):

    PCA_fitted=PCA.fit(self,X)
    W=self.eigen_vec_sorted[:2,:]

    #print(self.eigen_vec_sorted[:2,:])

    X_proj=self.X_std.dot(W.T)

    if self.isClassification:
      plt.scatter(X_proj[:, 0], X_proj[:, 1],c=y)
    else:
      plt.scatter(X_proj[:, 0], X_proj[:, 1])

    plt.xlabel('PC1');

    plt.xlabel('PC1'); plt.xticks([])
    plt.ylabel('PC2'); plt.yticks([])
    plt.ylabel('PC2')

    plt.title('2 Components, capture {} of total variance'.format(self.cum_explained_variance[1]))
    plt.show()


  def plot_3D(self,X,y=None):

    PCA_fitted=PCA.fit(self,X)
    W=self.eigen_vec_sorted[:3,:]

    X_proj=self.X_std.dot(W.T)

    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')

    if self.isClassification:
      ax.scatter(X_proj[:, 0], X_proj[:, 1],X_proj[:, 2],c=y)
    else:
      ax.scatter(X_proj[:, 0], X_proj[:, 1],X_proj[:, 2])
    
    ax.set_xlabel('PC1')
    ax.set_xlabel('PC2')
    ax.set_xlabel('PC3')

    plt.title('3 Components, capture {} of total variance'.format(self.cum_explained_variance[2]))
    plt.show()

  def create_scatterplots(self,X,y=None):
    if self.n_components>3:
      print("For more than 3 dimensions, plot can not be generated, however out of {} n_components , the 3D plot for first 3 components is as follows:\n".format(self.n_components))
      self.plot_3D(X,y)
    elif self.n_components==2:
      self.plot_2D(X,y)
    else:
      self.plot_3D(X,y)

  def create_cummulativeplot(self,X):
    PCA_fitted=PCA.fit(self,X)
    #plt.figure(figsize=(20,8))
    #print(X.shape[1])
    plt.plot(np.arange(1,X.shape[1]+1),self.cum_explained_variance,'-o')
    plt.xticks(np.arange(1,X.shape[1]+1))

    #plt.plot(np.arange(1,X.shape[1]+1),self.cum_explained_variance,'-o')
    #plt.xticks(np.arange(1,X.shape[1]+1))
    plt.xlabel('Number of Components')
    plt.ylabel('Cummulative Explained Variance')
    return plt.show()
     
def plt_():
    xpoints = np.array([1, 8])
    ypoints = np.array([3, 10])

    plt.plot(xpoints, ypoints)
    plt.show()




