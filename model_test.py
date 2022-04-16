import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('airline-price-prediction.csv')
df["date"]=pd.to_datetime(df["date"])
df["date"]=df["date"].dt.strftime("%m/%d/%Y")
df["Year"]=pd.DatetimeIndex(df["date"]).year
df["Month"]=pd.DatetimeIndex(df["date"]).month
df["Day"]=pd.DatetimeIndex(df["date"]).day

lbl_enc = LabelEncoder()
df['ch_code'] = lbl_enc.fit_transform(df[["ch_code"]])

#%%

df["dep_time"]=pd.to_datetime(df["dep_time"])
df['dep_time'] = df['dep_time'].dt.strftime("%-H:%M")

#%%

df["dep_hour"]=pd.DatetimeIndex(df["dep_time"]).hour
df["dep_minute"]=pd.DatetimeIndex(df["dep_time"]).minute

#%%

df=df.drop(["date","airline","dep_time"],axis=1) # drop initial columns
X = df.loc[:, df.columns != 'price']
Y = df['price']
print(X)