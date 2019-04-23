
# coding: utf-8

# In[1]:


file_list = ["vascular_multimodal_dataset_{}.csv".format(i) for i in range(1, 9)]


# In[2]:


file_list


# In[3]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[4]:


sets = [set(pd.read_csv(file_list[i])) for i in range(8)]
result = sets[0]
for s in sets:
    result = result.intersection(s)
intersect_cols = list(result)


# In[5]:


dfs = [pd.read_csv(file_list[i]) for i in range(8)]
MCI_df = pd.concat(dfs, ignore_index=True)
MCI_df.shape


# In[6]:


MCI_df = MCI_df[intersect_cols]


# In[7]:


MCI_df.head()


# In[37]:


sns.heatmap(MCI_df.corr(), cmap='coolwarm')


# In[38]:


sns.pairplot(MCI_df)


# In[8]:


# Many features seem to be linearly related. This will create unwanted variance in our models and does not contribute
# new information. Let us get rid of them


# In[9]:


from statsmodels.stats.outliers_influence import variance_inflation_factor    

def calculate_vif_(X, thresh=100):
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]


# In[10]:


Y = MCI_df['PO_PR_RAVLT_1.5_SUM']
MCI_df = calculate_vif_(MCI_df)


# In[49]:


sns.pairplot(MCI_df)


# In[11]:


# now we can see our features are much more varied and not so srongly related


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics


X_train, X_test, y_train, y_test = train_test_split(MCI_df.index, Y, test_size=0.2)

print("Number of train datapoints: {}".format(X_train.shape[0]))
print("Number of test datapoints: {}".format(y_test.shape[0]))

lm = LinearRegression()
lm.fit(MCI_df.iloc[X_train], y_train)

predictions = lm.predict(MCI_df.iloc[X_test])

mse = np.sqrt(np.mean((y_test - predictions) ** 2))
print ("The RMSE from Linear Model with one train/test split: {}".format(mse))

mse_naive = np.sqrt(np.mean((np.mean(y_train) - y_test) ** 2))
print ("The MSE from Naive prediction: {}".format(mse_naive))

kf = KFold(n_splits=5)
kf.get_n_splits(MCI_df)

c=1
for train_index, test_index in kf.split(MCI_df):
    X_train, X_test = MCI_df.iloc[train_index], MCI_df.iloc[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)

    mse = np.mean((y_test - predictions) ** 2)
    print ("The MSE from Linear Model fold #{}: {}".format(c,mse))
    c= c+1
    


# In[13]:


import xgboost

xgb = xgboost.XGBRegressor()

c=1
for train_index, test_index in kf.split(MCI_df):
    X_train, X_test = MCI_df.iloc[train_index], MCI_df.iloc[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    xgb.fit(X_train, y_train)
    predictions = xgb.predict(X_test)

    mse = np.sqrt(np.mean((y_test - predictions) ** 2))
    print ("The MSE from XGB Model fold #{}: {}".format(c,mse))
    c= c+1

