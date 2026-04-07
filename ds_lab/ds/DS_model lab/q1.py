import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("TiTanic-Dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

# Null values
print(df.isnull().sum().sort_values(ascending=False))

# Drop column
df.drop("Cabin", axis=1, inplace=True)

# Fill missing values (FIXED)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

print(df.isnull().sum().sort_values(ascending=False))

# =========================================================
# 📊 VISUALIZATION 1: Histogram (Distribution)
# =========================================================
plt.figure()
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# =========================================================
# 📊 VISUALIZATION 2: Scatter Plot (Relationship)
# =========================================================
plt.figure()
plt.scatter(df["Age"], df["Fare"])
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# =========================================================
# 📊 VISUALIZATION 3: Correlation Heatmap (Manual)
# =========================================================
corr = df[["Age","Fare","SibSp","Pclass","Survived"]].corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()

# =========================================================
# SCALING
# =========================================================
x = df[["Age","Fare","SibSp","Pclass"]]

# StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled[:5])

# MinMaxScaler
sc = MinMaxScaler()
x_sc = sc.fit_transform(x)
print(x_sc[:5])

# =========================================================
# MODEL
# =========================================================
x = df[["Age","Fare","Pclass"]]
y = df["Survived"]

model = LinearRegression()
model.fit(x, y)

# Predictions
y_pred = model.predict(x)

# =========================================================
# 📊 VISUALIZATION 4: Actual vs Predicted
# =========================================================
plt.figure()
plt.scatter(y, y_pred)
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()