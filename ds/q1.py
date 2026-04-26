import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt

df=pd.read_csv("covid.csv")
print("________head________")
print(df.head())
print("________mean________")
print(df.mean(numeric_only=True))
print
print(df.mean(axis=1, numeric_only=True))
print("________mode________")
print(df.mode(numeric_only=True))
print("________median________")
print(df.median(numeric_only=True))
print("________quantile________")

print(df["Confirmed"].quantile(0.25))


print(df.describe())

print(df.var(numeric_only=True))
print(df.std(numeric_only=True))
print("________min________")

print(df["Confirmed"].min())
print("________max________")
print(df["Confirmed"].max())
print("________skewness________")
print(df["Confirmed"].skew())
print("________kurtosis________")
print(df["Confirmed"].kurt())




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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("netflix_titles.csv")

# Clean data
df = df.dropna()

plt.figure()
plt.hist(df['country'],bins=20)
plt.title("distribution of ppl from different countries")
plt.xlabel("country")
plt.ylabel("duration")
plt.show()

plt.figure()
plt.scatter(x=df['country'],y=df['duration'])
plt.xlabel("country")
plt.ylabel("duration")
plt.show()

# 1. Histogram (clear)-always numbers
plt.figure()
plt.hist(df['Confirmed'], bins=20)
plt.title("Distribution of Confirmed Cases")
plt.xlabel("Confirmed")
plt.ylabel("Frequency")
plt.show()

#  2. Boxplot - numeric 
plt.figure()
sns.boxplot(x=df['Confirmed'])
plt.title("Boxplot of Confirmed Cases")
plt.show()

#  3. Scatter plot - 2 numeric colums 
plt.figure()
sns.scatterplot(x=df['Confirmed'], y=df['Deaths'])
plt.title("Confirmed vs Deaths")
plt.show()

#  4. Bar plot (use meaningful values)  categorical vs numerical 
plt.figure()
sns.barplot(x=df.head(10).index, y=df.head(10)['Confirmed'])
plt.title("Top 10 Confirmed Cases")
plt.xticks(rotation=45)
plt.show()

# 5. Heatmap numeric 
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()