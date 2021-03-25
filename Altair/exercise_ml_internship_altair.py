#!/usr/bin/env python
# coding: utf-8

# - """
# Here are 2 small exercises to evaluate your motivation and skills.
# The first one is simply a question, and the second one is related to
# applying a regression model to a dataset. The approach is at least or even
# more important than the result, please detail all the steps of your research.
# """
# 

# # 1 - Data preprocessing

# In[1]:


# ----------------------
# In a dataset, there is a feature named "Server/Machine Type", how would you
# transform/prepare this feature so that it can be used in a regression model
# (one only accepting float/bool as value), you don't have to code a solution,
# just write an answer on what you would do.

# Some example of values on this feature:
# Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (x86_64), 2950 MHz, 385570 MB RAM,  12079 MB swap
# Intel(R) Xeon(R) CPU E5-2670 v2 @ 2.50GHz (x86_64), 2500 MHz,  95717 MB RAM, 149012 MB swap
# Intel(R) Xeon(R) CPU E5-2697A v4 @ 2.60GHz (x86_64), 1300 MHz, 257868 MB RAM,  12027 MB swap
# Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (x86_64), 3138 MHz, 772642 MB RAM,   9042 MB swap
# Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (x86_64), 2214 MHz, 385570 MB RAM,  12090 MB swap
# Core(TM) i7-6700HQ CPU @ 2.60GHz (x86_64), 2600 MHz,  40078 MB RAM,  75183 MB swap
# Intel(R) Xeon(R) CPU E5-2697A v4 @ 2.60GHz (x86_64), 1199 MHz, 257868 MB RAM,  12247 MB swap
# Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (x86_64), 3246 MHz, 514658 MB RAM,  10770 MB swap
# Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (x86_64), 2483 MHz, 772642 MB RAM,   8266 MB swap


# # Your full text (not python) answer:
# 
# - There is no simple answer to transform this textual (non-numeric) data to float/bool value.
# - One way to solve this problem is, 
#     - We have one feature named "Server/Machine Type", we can transform this single feature into multiple features.
#     - As we can see the value of the feature is a configuration of the "Server/Machine Type", we can break it to multiple
# features like Server/Machine Name, Processor Speed (GHz), (32 or 64 bit), Processor Number(2950, 2500, 2600.. ),
# RAM(385570, 95717..), Swap Memory(12079, 149012..).
# 
#     - Most of the new columns are numeric values and doesn't need to transform again. 
#     - For Server/Machine Name other textual data we can handle in different ways like One Hot Encoding.
#     
# - Although we are increasing the dimension by transforming single feature into mulltiple features, We enable machine learning 
# algorithm to comprehend them and extract the useful information.

# # 2 - Regression

# 1. You are given a dataset (providing as additional file to this exercise) with 34 features and 1 target to predict, your task is to train a regression model on this dataset.
# 
# <br>
# 2. Code it in python3, provide as well a requirements.txt containing the version you use
#    I should be able to run directly in my linux terminal:
#    
#    - pip install -r requirements.txt && python exercise_ml_internship_altair.py
#    
# <br>
# 3. You are free to use every library you think is appropriate
# 

# # 2.1 loading dataset

import pandas as pd

df = pd.read_csv("dataset_ml_exo.csv")

# # 2.2 Data preparation

# This dataset has already been mostly prepared (it contains only float/bool features), but you may
# still have to do pre-processing (e.g. features reduction, other...).

df.head()
df['Unnamed: 0'].head()

# drop unnamed column (Index column)
df = df.drop(['Unnamed: 0'], axis=1)

# display dataframe
print(df)

# - Index column dropped

# ## 2.2.2 Checking for duplicate records

# checking whether we have any duplicated rows
df.duplicated().sum()
# - No duplicate records.

# Details of the dataset like number of columns, non=null count and the data type of each column
df.info()


# - feature_1 has mixed values (int, float, 'unknown').
# - Althought the column looks like it is a Date column, we will fill the unknown values with most frequently occurred value.

# ## 2.2.3 Handling unknown values in feature_1 column


# Filling the "unknown" values with most frequently occurred value.
df['feature_1'] = pd.to_numeric(df['feature_1'], errors='coerce')
df['feature_1'] = df['feature_1'].fillna(int(df.feature_1.mode()))


# display dataframe
print(df.head())

# Details of the dataset like number of columns and the data type of each column
df.info()

# ## 2.2.4 Filling the missing values with mean

# Checking for null values in all the column (features & columns)
df.isnull().sum()


# - feature_20, feature_26, feature_27, feature_28, feature_29, feature_30 has missing values which need to be filled.

# Filling the missing values with mean

df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())

# Checking for null values in all the column after filling the missing values with mean
df.isnull().sum()


# - No more missing values in the dataset

# ## 2.2.5 Changing boolean to 0 and 1 values

# Making sure whether all the columns are numberic values
df.info()


# - We have 2 columns (feature_12,feature_16) which are boolean and we need to convert it to numeric; 1 for True and 0 for False

# Changing boolean to 0 and 1 values

df['feature_12'] = df['feature_12'].astype(int)
df['feature_16'] = df['feature_16'].astype(int)

# Checking the dataframe again to confirm that all the columns are numberic values
df.info()

# - Now all the values in the dataframe are numeric

# ## 2.2.6 Dropping feature_1 as the column values are not consistent and we are not performing any time series analysis
df = df.drop(['feature_1'], axis=1)

df.head()

# ## 2.2.7 Feature Selection 

from sklearn import preprocessing

# Feature Scaling: Transform features by scaling each feature to a (0, 1) range

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_scaled = pd.DataFrame(x_scaled)

print(df_scaled)

X = df_scaled.iloc[:, :-1]
y = df_scaled.iloc[:, -1]

print(X.shape)
print(y.shape)

# Splitting test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 0) #fitting multiple regression model to the training set 

print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)

# ## Checking the correlation among the features 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(25,20))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

# function to find highly correlated features

def correlation(dataset, threshold):
    """
    dataset: pass the dataset to find the correlation
    threshold: features with more than threshold value will be added to set - col_corr
    
    return: returns col_corr set contains all the features with correlation more than threshold value.
            Here absolute coeff value is taken into consideration.
    """
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


corr_features = correlation(X_train, 0.89) # gets correlated features with more than .89 value
len(set(corr_features))


print(corr_features)

# dropped correlated features

X_train_dropped = X_train.drop(corr_features,axis=1)
X_test_dropped = X_test.drop(corr_features,axis=1)


# # 2.3 model training

# metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ## 2.3.1 Linear Regression

# Fitting the Linear Regression Model to the dataset

from sklearn.linear_model import LinearRegression

regressor_lr = LinearRegression()
regressor_lr.fit(X_train_dropped, y_train)

#predicting the test set results

y_pred_lr = regressor_lr.predict(X_test_dropped)

# calculating metrics

r2_score_lr =r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = (np.sqrt(mse_lr))

print('R2_score: ', r2_score_lr)
print('Mean Absolute Error: ', mae_lr)
print('Mean Squared Error: ', mse_lr)
print("RMSE: ", rmse_lr)

# ## 2.3.2 Support Vector Regression

# Fitting the Support Vector Regression Model to the dataset

from sklearn.svm import SVR

regressor_svr = SVR(kernel = 'poly', gamma = 'scale')
regressor_svr.fit(X_train_dropped, y_train)

#predicting the test set results

y_pred_svr = regressor_svr.predict(X_test_dropped)

# calculating metrics

r2_score_svr =r2_score(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = (np.sqrt(mse_svr))

print('R2_score: ', r2_score_svr)
print('Mean Absolute Error: ', mae_svr)
print('Mean Squared Error: ', mse_svr)
print("RMSE: ", rmse_svr)

# ## 2.3.3 Decision Tree

# Fitting the Decision Tree Regression Model to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor_dt = DecisionTreeRegressor(random_state = 0)
regressor_dt.fit(X_train_dropped, y_train)

#predicting the test set results

y_pred_dt = regressor_dt.predict(X_test_dropped)

# calculating metrics

r2_score_dt = r2_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = (np.sqrt(mse_dt))

print('R2_score: ', r2_score_dt)
print('Mean Absolute Error: ', mae_dt)
print('Mean Squared Error: ', mse_dt)
print("RMSE: ", rmse_dt)

# ## 2.3.4 Random Forest

# Fitting the Random Forest Regression Model to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor_rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_rf.fit(X_train_dropped, y_train)

#predicting the test set results

y_pred_rf = regressor_rf.predict(X_test_dropped)

# calculating metrics

r2_score_rf =r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = (np.sqrt(mse_rf))

print('R2_score: ', r2_score_rf)
print('Mean Absolute Error: ', mae_rf)
print('Mean Squared Error: ', mse_rf)
print("RMSE: ", rmse_rf)

# # 2.4 model evaluation (evaluate model perf and display metrics)

# All metric appended to the list

benchmark_metrics = ['Linear Regression', 'Support Vector Regression', 
                     'Decision Tree', 'Random Forest']

# All model RMSE values appended to the list
RMSE_values = [rmse_lr, rmse_svr, rmse_dt, rmse_rf]

# All model MAE values appended to the list
MAE_values = [mae_lr, mae_svr, mae_dt, mae_rf]

# All model MSE values appended to the list
MSE_values = [mse_lr, mse_svr, mse_dt, mse_rf]

# All model R2_score values appended to the list
R2_score = [r2_score_lr, r2_score_svr, r2_score_dt, r2_score_rf]


# ### Bar graph for models Vs RMSE values

fig_dims = (10, 5)
fig, ax = plt.subplots(figsize=fig_dims)
ax.bar(benchmark_metrics,RMSE_values,color=['#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'] )
ax.set_xlabel("Model", fontweight='bold')
ax.set_ylabel("RMSE values", fontweight='bold')
ax.set_title('Accuracy by model and RMSE value', fontweight='bold')
plt.show()

# ### Bar graph for models Vs MSE values

fig_dims = (10, 5)
fig, ax = plt.subplots(figsize=fig_dims)
ax.bar(benchmark_metrics, MSE_values,color=['#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'] )
ax.set_xlabel("Model", fontweight='bold')
ax.set_ylabel("MSE values", fontweight='bold')
ax.set_title('Accuracy by model and MSE value', fontweight='bold')
plt.show()

# ### Bar graph for models Vs MAE values

fig_dims = (10, 5)
fig, ax = plt.subplots(figsize=fig_dims)
ax.bar(benchmark_metrics,MAE_values,color=['#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'] )
ax.set_xlabel("Model", fontweight='bold')
ax.set_ylabel("MAE values", fontweight='bold')
ax.set_title('Accuracy by model and MAE value', fontweight='bold')
# plt.xticks(rotation=90)
plt.show()

X = np.arange(4)
barWidth = 0.2
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0,0,1,1])

ax.bar(X + 0.00, RMSE_values, color = 'b', width = 0.2, label = 'RMSE')
ax.bar(X + 0.2, MAE_values, color = 'g', width = 0.2, label = 'MAE')
ax.bar(X + 0.40, MSE_values, color = 'r', width = 0.2, label = 'MSE')
ax.set_title('Accuracy of model Vs RMSE, MAE, MSE value', fontweight='bold')

plt.xlabel('Model', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(benchmark_metrics))], benchmark_metrics)
plt.legend()
plt.show()


# ### Bar graph for models Vs R2_score values

fig_dims = (10, 5)
fig, ax = plt.subplots(figsize=fig_dims)
ax.bar(benchmark_metrics,R2_score,color=['#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'] )
ax.set_xlabel("Model", fontweight='bold')
ax.set_ylabel("R2_score values", fontweight='bold')
ax.set_title('Accuracy by model and R2_score value', fontweight='bold')
# plt.xticks(rotation=90)
plt.show()


# Thanks !