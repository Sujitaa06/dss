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