#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import datetime

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import joblib
#%%
df = pd.read_csv('Samples/airline-test-samples.csv')
df.head()
#%%
import warnings
warnings.filterwarnings("ignore")
#%% md

### Data preprocessing on 'price'

#%%
df['price'] = df['price'].str.replace(",", "")
df['price'] = pd.to_numeric(df['price'])
#%% md
#### Data preprocessing on Date
#%%
df["date"]=pd.to_datetime(df["date"])
df["date"]=df["date"].dt.strftime("%m/%d/%Y")
df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
df['dayofyear'] = pd.DatetimeIndex(df['date']).dayofyear

#%% md
#### Loading Ch_code Encoder :
#%%
ch_enc = joblib.load("EncoderModels/ch_enc.save")

df['ch_code'] = ch_enc.transform(df[["ch_code"]])

df.head()
#%% md
#### Departure time preprocessing
#%%
df["dep_time"]=pd.to_datetime(df["dep_time"])
df['dep_time'] = df['dep_time'].dt.strftime("%-H:%M")
df["dep_hour"]=pd.DatetimeIndex(df["dep_time"]).hour
df["dep_minute"]=pd.DatetimeIndex(df["dep_time"]).minute
#%% md
#### Time_taken preprocessing
#%%
df["hours_taken"] = df["time_taken"].str.split('h').str.get(0)
df["minutes_taken"] = df["time_taken"].str[4:6]
df["minutes_taken"] = df["minutes_taken"].str.replace('m', '')
df["minutes_taken"] = df["minutes_taken"].str.replace('h', '')
df["hours_taken"] = pd.to_numeric(df["hours_taken"])
df["minutes_taken"] = pd.to_numeric(df["minutes_taken"], errors='coerce')
#%% md
#### Stop preprocessing
#%%
df["stop"] = df["stop"].str.split('-').str.get(0)
df["stop"] = df["stop"].replace(['non'], 0)
df.isna().sum() #  28944 null vals
df["stop"] = df["stop"].replace(['2+'], 2) # Indicates for 2 or more stops
df['stop'] = df['stop'].fillna(0)
df['stop'] = pd.to_numeric(df['stop'])
#%% md
#### Arrival time preprocessing
#%%
df["arr_time"]=pd.to_datetime(df["arr_time"])
df['arr_time'] = df['arr_time'].dt.strftime("%-H:%M")
df["arr_hour"]=pd.DatetimeIndex(df["arr_time"]).hour
df["arr_minute"]=pd.DatetimeIndex(df["arr_time"]).minute
df["arr_hour"] = pd.to_numeric(df["arr_hour"])
df["arr_minute"] = pd.to_numeric(df["arr_minute"])
#%% md
#### Source & Destination preprocessing
#%%
df['source'] = df['route'].str.split( ', ').str.get(0).str.split(':').str.get(1)
df['destination'] = df['route'].str.split( ', ').str.get(1).str.split(':').str.get(1).str.split('}').str.get(0)
df['source'] = df['source'].str.replace('\'', "")
df['destination'] = df['destination'].str.replace('\'', "")
#%% md
#### Loading Type Encoder :
#%%
type_enc = joblib.load("EncoderModels/type_enc.save")
df['type'] = type_enc.transform(df[["type"]])
df.head()
#%% md
#### Loading Source Encoder :
#%%
source_enc = joblib.load("EncoderModels/source_enc.save")
df['source'] = source_enc.transform(df[["source"]])
df.head()
#%% md
#### Loading Destination Encoder :
#%%
destination_enc = joblib.load("EncoderModels/destination_enc.save")
df['destination'] = source_enc.transform(df[["destination"]])
df.head()
#%% md
#### Cleaning Data
#%%
df = df.fillna(-1)
df = df.drop(['airline', 'date', 'dep_time', "time_taken", 'arr_time', 'route',], axis=1)
df.head()
#%%
X = df.loc[:, df.columns != 'price']
Y = df['price']
#%% md
#### Loading Scaler Model :
#%%
scaler = joblib.load("scaler.save")
X_scaled = scaler.transform(X)
#%%
dtr_model = joblib.load("RegressionModels/DecisionTreeRegressor(random_state=42)_RegressionModel.save")
elastic_model = joblib.load("RegressionModels/ElasticNet()_RegressionModel.save")
lasso_model = joblib.load("RegressionModels/Lasso()_RegressionModel.save")
ridge_model = joblib.load("RegressionModels/Ridge()_RegressionModel.save")
#%%
for i, clf in enumerate((dtr_model, elastic_model, lasso_model,ridge_model)):
    predictions = clf.predict(X_scaled)
    print('Mean Square Error of '+ str(clf)+":"+ str(metrics.mean_squared_error(Y, predictions)))
    print(('Accuracy of '+ str(clf)+":"+str(r2_score(Y, predictions))))
    print('\n')

