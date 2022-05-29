#%%

import jupyterthemes as jt
from jupyterthemes import get_themes
import jupyterthemes as jt
# from jupyterthemes.stylefx import set_nb_theme

#%% md

import jupyterthemes as jt
from jupyterthemes.stylefx import set_nb_theme

#%%

# set_nb_theme('onedork')
#monokai
#chesterish
#oceans16 gamed
#onedork gamed brdo
#solarizedl

#%% md

# Code start from here :

#%% md

### Importing :

#%%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import datetime

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import joblib


#%% md

### Reading data from csv

#%%

df = pd.read_csv('Samples/airline-price-prediction.csv')

#%%

df.head()

#%%

df.shape

#%% md

### Data preprocessing on 'price'

#%%
df['price'] = df['price'].str.replace(",", "")
df['price'] = pd.to_numeric(df['price'])
#%% md

### Data preprocessing on 'date'

#%%

df["date"]=pd.to_datetime(df["date"])
df["date"]=df["date"].dt.strftime("%m/%d/%Y")


#%%

print(df[df.columns[0]])

#%%

df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
df['dayofyear'] = pd.DatetimeIndex(df['date']).dayofyear

#%% md

### Data preprocessing : 'ch_code'

#%%

ch_enc = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=9)
df['ch_code'] = ch_enc.fit_transform(df[["ch_code"]])
print(df['ch_code'])

#%%

filename = "ch_enc.save"
joblib.dump(ch_enc,filename)

#%% md

### Data preprocessing : 'dep_time'

#%%

df["dep_time"]=pd.to_datetime(df["dep_time"])
df['dep_time'] = df['dep_time'].dt.strftime("%-H:%M")

#%%

df["dep_hour"]=pd.DatetimeIndex(df["dep_time"]).hour
df["dep_minute"]=pd.DatetimeIndex(df["dep_time"]).minute

#%% md

### Data preprocessing : 'time_taken'

#%%

df["hours_taken"] = df["time_taken"].str.split('h').str.get(0)
df["minutes_taken"] = df["time_taken"].str[4:6]
df["minutes_taken"] = df["minutes_taken"].str.replace('m', '')
df["minutes_taken"] = df["minutes_taken"].str.replace('h', '')
df["hours_taken"] = pd.to_numeric(df["hours_taken"])
df["minutes_taken"] = pd.to_numeric(df["minutes_taken"], errors='coerce')
df.head()

#%% md

### Data preprocessing : 'stop'

#%%

df["stop"] = df["stop"].str.split('-').str.get(0)
df["stop"] = df["stop"].replace(['non'], 0)
df.isna().sum() #  28944 null vals
df["stop"] = df["stop"].replace(['2+'], 2) # Indicates for 2 or more stops
df['stop'] = df['stop'].fillna(0)
df['stop'] = pd.to_numeric(df['stop'])
# print(df[9:14])

#%% md

### Data preprocessing : 'arr_time'

#%%

df["arr_time"]=pd.to_datetime(df["arr_time"])
df['arr_time'] = df['arr_time'].dt.strftime("%-H:%M")
df["arr_hour"]=pd.DatetimeIndex(df["arr_time"]).hour
df["arr_minute"]=pd.DatetimeIndex(df["arr_time"]).minute
df["arr_hour"] = pd.to_numeric(df["arr_hour"])
df["arr_minute"] = pd.to_numeric(df["arr_minute"])
df.head()


#%% md

### Data preprocessing : 'type'

#%%

type_enc = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=2)
df['type'] = type_enc.fit_transform(df[["type"]])
print(df['type'])

#%%

filename = "type_enc.save"
joblib.dump(type_enc,filename)

#%% md

### Data preprocessing : 'route'

#%%

df['source'] = df['route'].str.split( ', ').str.get(0).str.split(':').str.get(1)
df['destination'] = df['route'].str.split( ', ').str.get(1).str.split(':').str.get(1).str.split('}').str.get(0)
df['source'] = df['source'].str.replace('\'', "")
df['destination'] = df['destination'].str.replace('\'', "")

#%%

source_enc = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=6)
df['source'] = source_enc.fit_transform(df[["source"]])
print(df['source'])

#%%

filename = "source_enc.save"
joblib.dump(source_enc,filename)

#%%

destination_enc = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=6)
df['destination'] = destination_enc.fit_transform(df[["destination"]])
print(df['destination'])
df = df.fillna(-1)
df = df.drop(['airline', 'date', 'dep_time', "time_taken", 'arr_time', 'route',], axis=1)
#df = pd.get_dummies(df)

#%%
filename = "destination_enc.save"
joblib.dump(destination_enc,filename)

#%%
df.head()

#%%

X = df.loc[:, df.columns != 'price']
Y = df['price']

#%%

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=120)

#%%

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
scaler = ss.fit(XTrain)
trainX_scaled = scaler.transform(XTrain)
testX_scaled = scaler.transform(XTest)

#%%

scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)

#%%

lasso = linear_model.Lasso()
elastic = linear_model.ElasticNet()
#logestic =linear_model.LogisticRegression(C=1) #Classifier
dtree = DecisionTreeRegressor(random_state=42)
ridge = linear_model.Ridge()
polynomial = PolynomialFeatures(degree=3)
XPolynomial = polynomial.fit_transform(trainX_scaled)
poly = linear_model.LinearRegression()

ridge.fit(trainX_scaled,YTrain)
poly.fit(XPolynomial, YTrain)
lasso.fit(trainX_scaled, YTrain)
elastic.fit(trainX_scaled, YTrain)
#logestic.fit(trainX_scaled, YTrain)
dtree.fit(trainX_scaled, YTrain)

#%%

for i, clf in enumerate((lasso, elastic, dtree,ridge)):
    predictions = clf.predict(testX_scaled)
    print('Mean Square Error of '+ str(clf)+":"+ str(metrics.mean_squared_error(YTest, predictions)))
    print(('Accuracy of '+ str(clf)+":"+str(r2_score(YTest, predictions))))
    print('\n')

#%%
filename = '_RegressionModel.save'
for i, clf in enumerate((lasso, elastic, dtree, ridge)):
    joblib.dump(clf, str(clf)+filename)