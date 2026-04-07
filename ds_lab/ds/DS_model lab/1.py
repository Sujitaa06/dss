import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
df=pd.read_csv("TiTanic-Dataset.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum().sort_values(ascending=False))
df.drop("Cabin",inplace=True,axis=1)
df.fillna(df["Age"].median(numeric_only=True),inplace=True)
df.fillna(df["Embarked"].mode()[0],inplace=True)
print(df.isnull().sum().sort_values(ascending=False))
x=df[["Age","Fare","SibSp","Pclass"]]
#standard scaler svm,logistic regression,linear regression
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
print(x_scaled[:5])
#MinMax Scaler  knn,k means
sc=MinMaxScaler()
x_sc=sc.fit_transform(x)
print(x_sc[:5])
x=df[["Age","Fare","Pclass"]]
y=df["Survived"]
model=LinearRegression()
model.fit(x,y)