import pandas as pd






df=pd.read_csv("netflix_titles.csv",encoding="latin-1")

df=df[["type","description"]]
df.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df["type"]=l.fit_transform(df["type"])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df["description"],df["type"],test_size=0.2)

from sklearn.feature_extraction.text import  TfidfVectorizer

tf=TfidfVectorizer()
x_train1=tf.fit_transform(x_train)
x_test1=tf.transform(x_test)


from imblearn.over_sampling import SMOTE
s=SMOTE()

x_train_res,y_train_res=s.fit_resample(x_train1,y_train)
from sklearn.svm import LinearSVC
model =LinearSVC(max_iter=200,C=1.0)
model.fit(x_train_res,y_train_res)
y_pred=model.predict(x_test1)

from sklearn.metrics import accuracy_score , confusion_matrix

print("Accuracy score:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))

print(df["type"].value_counts())
print(pd.Series(y_train).value_counts())
print(pd.Series(y_train_res).value_counts())
