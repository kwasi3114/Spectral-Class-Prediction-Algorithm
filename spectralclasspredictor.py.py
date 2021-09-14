#!/usr/bin/env python
# coding: utf-8

# In[4]:


#program written to predict the spectral class of a star based on temperature (K),
#luminosity (L/Lo), radius (R/Ro), absolute magnitude (Mv), and star type
#last modified: 9/13/21

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

stellar_data = pd.read_csv(r"C:\Users\kwasi\OneDrive\Desktop\Books\6 class csv.csv")
#data used: https://www.kaggle.com/deepu1109/star-dataset

X = stellar_data.values[:,0:5]
Y = stellar_data.values[:,6]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 50, max_depth = 20, min_samples_leaf = 40);
clf_gini.fit(X_train,Y_train)

DecisionTreeClassifier(class_weight=None, criterion='gini',max_depth=7,max_features=None,max_leaf_nodes=None,min_samples_leaf=20,min_samples_split=10,min_weight_fraction_leaf=0.0,presort=False,random_state=100,splitter='best')

#from left to right: temperature, luminosity, radius, absolute magnitude, star type
#change the values in the line below to test different values
pred = clf_gini.predict([[3068,0.0024,0.17,16.12,0]])
print(pred)

#array(['R'], dtype=object)

#Y_pred = clf_gini.predict(X_test)
#print(Y_pred)

#print(len(stellar_data))
#print(stellar_data.shape)


# In[ ]:





# In[ ]:





# In[ ]:




