#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[108]:


df= pd.read_excel('Data.xlsx')
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]


# **Splitting Data into Training Set Testing Set** 

# In[112]:


def splitter(X,Y,testsize=0.2, validationsize=0.25,randomstate=99):
    X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=testsize, random_state=randomstate)
    X_train, X_val, Y_train, Y_val= train_test_split(X_train,Y_train, test_size= validationsize, random_state= randomstate)
    
    scaler= StandardScaler().fit(X_train)
    x_train= scaler.transform(X_train)
    x_test= scaler.transform(X_test)
    x_val= scaler.transform (X_val)
    y_train, y_val, y_test= Y_train,Y_val, Y_test   ## for the purpose of keeping uniform letter cases of X and Y
    
    return x_train, x_test, y_train, y_test
    


# In[113]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# In[114]:


x_train, x_test, y_train, y_test=splitter(X,Y)
x_train.shape
x_test.shape


# RANDOM FOREST REGRESSOR

# In[288]:


from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor(n_estimators=100, random_state=0)


# In[289]:


model.fit(x_train, y_train)


# In[290]:


def get_score( y_test, y_pred, base=' ' ):
    r2=r2_score(y_test, y_pred)
    mse=mean_squared_error(y_test, y_pred)
    return print(f'{base} has R2:= {r2} ,  mse:= {mse}')


# In[291]:


y_pred_test=model.predict(x_test)
y_pred_train=model.predict(x_train)


# In[293]:


get_score(y_train, y_pred_train, "Training set")
get_score(y_test, y_pred_test, "Testing Set")


# In[ ]:





# In[ ]:





# RIDGE REGRESSION

# In[146]:


from sklearn.linear_model import Ridge
model= Ridge(alpha=0.01)

x_train, x_test, y_train, y_test= splitter(X,Y,randomstate=456)


# In[147]:


model.fit(x_train,y_train)


# In[148]:


y_pred_train= model.predict(x_train)
y_pred_test= model.predict(x_test)


# In[149]:


get_score(y_train, y_pred_train, "Training Set")
get_score(y_test, y_pred_test, 'Testing Set')


# ELASTIC NET

# In[150]:


from sklearn.linear_model import ElasticNet
model= ElasticNet(alpha=0.01)


# In[151]:


x_train, x_test, y_train, y_test= splitter(X,Y,randomstate=234)
model.fit(x_train,y_train)


# In[152]:


y_pred_train= model.predict(x_train)
y_pred_test= model.predict(x_test)


# In[153]:


get_score(y_train, y_pred_train, "Training Set")
get_score(y_test, y_pred_test, 'Testing Set')


# LASSO REGRESSION

# In[128]:


from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01)


# In[140]:


x_train, x_test, y_train, y_test= splitter(X,Y, randomstate=5678)
model.fit(x_train,y_train)


# In[141]:


y_pred_train= model.predict(x_train)
y_pred_test= model.predict(x_test)


# In[142]:


get_score(y_train, y_pred_train, "Training Set")
get_score(y_test, y_pred_test, 'Testing Set')


# LINEAR REGRESSION
# 

# In[177]:


from sklearn.linear_model import LinearRegression
model= LinearRegression()


# In[178]:


x_train, x_test, y_train, y_test= splitter(X,Y,randomstate=999)  #randomstate=99
model.fit(x_train,y_train)


# In[179]:


y_pred_train= model.predict(x_train)
y_pred_test= model.predict(x_test)


# In[180]:


get_score(y_train, y_pred_train, "Training Set")
get_score(y_test, y_pred_test, 'Testing Set')


# SVM ANALYSIS
# 
# 
# 

# In[298]:


from sklearn.svm import SVR
model=SVR(kernel='rbf')


# In[299]:


x_train, x_test, y_train, y_test= splitter(X,Y,randomstate=214)  #randomstate=99
model.fit(x_train,y_train)


# In[300]:


y_pred_train= model.predict(x_train)
y_pred_test= model.predict(x_test)


# In[301]:


get_score(y_train, y_pred_train, "Training Set")
get_score(y_test, y_pred_test, 'Testing Set')


# ANN REGRESSION

# In[276]:


from sklearn.neural_network import MLPRegressor
model= MLPRegressor(hidden_layer_sizes=(10,), max_iter=5000)


# In[277]:


x_train, x_test, y_train, y_test= splitter(X,Y,randomstate=999)  #randomstate=999
model.fit(x_train,y_train)


# In[278]:


y_pred_train= model.predict(x_train)
y_pred_test= model.predict(x_test)


# In[279]:


get_score(y_train, y_pred_train, "Training Set")
get_score(y_test, y_pred_test, 'Testing Set')


# In[ ]:




