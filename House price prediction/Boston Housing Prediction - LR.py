#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm


# In[6]:


from sklearn.linear_model import LinearRegression


# In[67]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[ ]:





# | **Feature** | **Description** |
# |-------------|:-----------------|
# | **CRIM**    | Per capita crime rate by town |
# | **ZN**      | Proportion of residential land zoned for lots over 25,000 sq.ft. |
# | **INDUS**   | Proportion of non-retail business acres per town. |
# | **CHAS**    | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
# | **NOX**     | Nitric oxides concentration (parts per 10 million) |
# | **RM**      | Average number of rooms per dwelling |
# | **AGE**     | Proportion of owner-occupied units built prior to 1940 |
# | **DIS**     | Weighted distances to five Boston employment centres |
# | **RAD**     | Index of accessibility to radial highways |
# | **TAX**     | Full-value property-tax rate per \$10,000 |
# | **PTRATIO** | Pupil-teacher ratio by town |
# | **B**       | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town |
# | **LSTAT**   | % lower status of the population |
# | **MEDV**    | Median value of owner-occupied homes in $1000's |
# 

# In[12]:


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[9]:


path = r'D:\projects\z datasets\boston house pricing\housing.csv'


# In[91]:


data = pd.read_csv(path, header= None, names= column_names, delimiter= r'\s+') 
# s+: regular expression: will match one or more consecutive whitespace characters.


# In[92]:


data.head()


# In[93]:


data.shape


# In[111]:


data[data.isnull().any(axis=1)]


# In[39]:


median = data.median()
mean = data.mean()


# In[40]:


median - mean


# - median - mean: -ve number indicate left ( -ve ) skew [Tax, ZN] -> tail on left
# - +ve number indicate right skew ( +ve ) skew [ B ] -> tail on right

# In[94]:


data.describe().T


# In[95]:


data.info()


# In[ ]:





# In[96]:


fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in data.items():
    sns.boxplot(y=k, data=data, ax=axs[index])
    axs[index].set_title(f'Boxplot of {k}')
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[ ]:





# In[97]:


for k, v in data.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))
    
    


# In[ ]:





# In[98]:


fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in data.items():
    sns.scatterplot(x= data.index, y=v, ax=axs[index])
    axs[index].set_title(f'Boxplot of {k}')
    index += 1
plt.tight_layout(pad=0.9, w_pad=0.7, h_pad=2.0)
plt.show()


# In[ ]:





# - remove outliers

# In[ ]:


data = data[~(data['MEDV'] >= 50.0)]
print(np.shape(data))


# In[ ]:





# In[99]:


fig, axs = plt.subplots(ncols= 7, nrows= 2, figsize=(20, 10))

index= 0
axs = axs.flatten()
for k, v in data.items():
    sns.distplot(v, ax= axs[index])
    index += 1

plt.tight_layout(pad= 0.4, w_pad= 0.8, h_pad= 20)


# In[ ]:





# In[100]:


plt.figure(figsize=(30, 30))
sns.heatmap(data.corr().abs(),  annot=True, vmax= 1, cmap= 'Blues', fmt= '.1f')


# - TAX, RAD are highly correlated
# -  LSTAT, INDUS, RM, TAX, NOX, PTRAIO has a correlation score of 0.5 with MEDV

# In[ ]:





# In[102]:


sscaler = MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = data.loc[:,column_sels]
y = data['MEDV']
x = pd.DataFrame(data=sscaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[77]:


y.shape


# In[69]:


# try remove skewness
y =  np.log1p(y)
for col in x.columns:
    if np.abs(x[col].skew()) > 0.3:
        x[col] = np.log1p(x[col])


# In[79]:


y.shape


# In[70]:


from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[90]:


x.isnull().sum()


# # Linear regresion

# In[103]:


l_regression = linear_model.LinearRegression()
kf = KFold(n_splits=10)
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scores = cross_val_score(l_regression, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

scores_map = {}
scores_map['LinearRegression'] = scores
l_ridge = linear_model.Ridge()
scores = cross_val_score(l_ridge, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['Ridge'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# Lets try polinomial regression with L2 with degree for the best fit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

#for degree in range(2, 6):
#    model = make_pipeline(PolynomialFeatures(degree=degree), linear_model.Ridge())
#    scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
#    print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge())
scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['PolyRidge'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


# In[87]:


np.isnan(x_scaled).sum()  # Check for NaNs in your original data


# # SVM

# In[105]:


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# In[106]:


svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#grid_sv = GridSearchCV(svr_rbf, cv=kf, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, scoring='neg_mean_squared_error')
#grid_sv.fit(x_scaled, y)
#print("Best classifier :", grid_sv.best_estimator_)
scores = cross_val_score(svr_rbf, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['SVR'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


# In[ ]:





# # DT

# In[107]:


from sklearn.tree import DecisionTreeRegressor

desc_tr = DecisionTreeRegressor(max_depth=5)
#grid_sv = GridSearchCV(desc_tr, cv=kf, param_grid={"max_depth" : [1, 2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
#grid_sv.fit(x_scaled, y)
#print("Best classifier :", grid_sv.best_estimator_)
scores = cross_val_score(desc_tr, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['DecisionTreeRegressor'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


# # KNN

# In[108]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=7)
scores = cross_val_score(knn, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['KNeighborsRegressor'] = scores
#grid_sv = GridSearchCV(knn, cv=kf, param_grid={"n_neighbors" : [2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
#grid_sv.fit(x_scaled, y)
#print("Best classifier :", grid_sv.best_estimator_)
print("KNN Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


# # GBoosting R

# In[109]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(alpha=0.9,learning_rate=0.05, max_depth=2, min_samples_leaf=5, min_samples_split=2, n_estimators=100, random_state=30)
#param_grid={'n_estimators':[100, 200], 'learning_rate': [0.1,0.05,0.02], 'max_depth':[2, 4,6], 'min_samples_leaf':[3,5,9]}
#grid_sv = GridSearchCV(gbr, cv=kf, param_grid=param_grid, scoring='neg_mean_squared_error')
#grid_sv.fit(x_scaled, y)
#print("Best classifier :", grid_sv.best_estimator_)
scores = cross_val_score(gbr, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['GradientBoostingRegressor'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


# In[ ]:





# In[110]:


plt.figure(figsize=(20, 10))
scores_map = pd.DataFrame(scores_map)
sns.boxplot(data=scores_map)


# In[ ]:





# In[ ]:




