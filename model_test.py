# Code start from here :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

#%% md

### Reading data from csv

#%%

df = pd.read_csv('airline-price-prediction.csv')
df.head()
df.shape

#%% md

### Data preprocessing on 'date'

#%%

df["date"]=pd.to_datetime(df["date"])
df["date"]=df["date"].dt.strftime("%m/%d/%Y")

print(df[df.columns[0]])

df["Year"]=pd.DatetimeIndex(df["date"]).year
df["Month"]=pd.DatetimeIndex(df["date"]).month
df["Day"]=pd.DatetimeIndex(df["date"]).day

### Data preprocessing : 'ch_code'

#%%

lbl_enc = LabelEncoder()
df['ch_code'] = lbl_enc.fit_transform(df[["ch_code"]])
print(df['ch_code'])

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

df["hours_taken"] = df["time_taken"].str[:2]
df["minutes_taken"] = df["time_taken"].str[4:6]
df.head()
# Stop Column : (Problem : '+2')

#%% md

### Data preprocessing : 'stop'

#%%

df["stop"] = df["stop"].str.split('-').str.get(0)
df["stop"] = df["stop"].replace(['non'], 0)
df.isna().sum() #  28944 null vals
df["stop"] = df["stop"].replace(['2+'], 2) # Indicates for 2 or more stops
df['stop'] = df['stop'].fillna(0)
# print(df[9:14])

#%% md

### Data preprocessing : 'arr_time'

#%%

df["arr_time"]=pd.to_datetime(df["arr_time"])
df['arr_time'] = df['arr_time'].dt.strftime("%-H:%M")
df["arr_hour"]=pd.DatetimeIndex(df["arr_time"]).hour
df["arr_minute"]=pd.DatetimeIndex(df["arr_time"]).minute
df.head()


#%% md

### Data preprocessing : 'type'

#%%

df['type'] = lbl_enc.fit_transform(df[["type"]])
print(df['type'])

#%% md

### Data preprocessing : 'route'

## sum hours + mins in one col (departure_time, arrival_time)*****
## drop year (as all are duplicate)
##

#%%

df['source']=df['route'].str.split( ', ').str.get(0).str.split(':').str.get(1)
df['destination']=df['route'].str.split( ', ').str.get(1).str.split(':').str.get(1).str.split('}').str.get(0)
l = LabelEncoder()
df['source'] = l.fit_transform(df[['source']])
m = LabelEncoder()
df['destination'] = m.fit_transform(df[['destination']])

print (df['source'])
print (df['destination'])

#%%

df=df.drop(["date","airline","dep_time","time_taken","arr_time","route","Year"],axis=1) #drop initial column
X = df.loc[:, df.columns != 'price']
Y = df['price']
print(X)

#%%

df.head()
