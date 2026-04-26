import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("netflix_titles.csv")
print(df.head())
print(df.describe())
print("mean",df.mean(numeric_only=True))
print(df.mean(axis=1,numeric_only=True))
print("median",df.median(numeric_only=True))
print(df.mode(numeric_only=True))
print(df.var(numeric_only=True))
print(df.std(numeric_only=True))
print(df['release_year'].min())
print(df['release_year'].max())
print(df['release_year'].kurt())
print(df['release_year'].skew())
print(df['release_year'].quantile(0.25))



df=pd.read_csv("netflix_titles.csv")
df=df.drop_duplicates()
df=df.fillna(df.mean(numeric_only=True))
df=df.ffill()
y=df.iloc[:,-1]
x=df.iloc[:,:-1]



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

x=pd.get_dummies(x)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)


x_res,y_res=x_scaled,y

df_final=pd.concat(
    [pd.DataFrame(x_res),pd.DataFrame(y_res)],
    axis=1
)



df_final.head(100).to_csv("Preprocessed data.csv",index=False)
print("Preprocessing completed successfully" )

plt.figure()
plt.hist(df['country'],bins=20)
plt.title("distribution of ppl from different countries")
plt.xlabel("country")
plt.ylabel("duration")
plt.show

plt.figure()
plt.ScatterPlot(x=df['country'],y=df['duration'])
plt.xlabel("country")
plt.ylabel("duration")
plt.show