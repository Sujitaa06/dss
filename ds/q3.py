import pandas as pd
from sklearn.preprocessing import LabelEncoder as le
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score,confusion_matrix
from imblearn.over_sampling import SMOTE as sm
from sklearn.svm import LinearSVC   


df=pd.read_csv("Corona_NLP_train.csv",encoding="latin-1")
df=df[["Sentiment","OriginalTweet"]]
df.dropna(inplace=True)

         
l=le()
df["Sentiment"]=l.fit_transform(df["Sentiment"])
X_train,X_test,y_train,y_test=train_test_split(df["OriginalTweet"],df["Sentiment"],test_size=0.2)


tf=TfidfVectorizer()
X_train1=tf.fit_transform(X_train)
X_test1=tf.transform(X_test)
s=sm()
X_train_res,y_train_res=s.fit_resample(X_train1,y_train)
model = LinearSVC(max_iter=200, C=1.0)
model.fit(X_train_res,y_train_res)
y_pred=model.predict(X_test1)


print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print(df["Sentiment"].value_counts())
print(pd.Series(y_train).value_counts())      # before SMOTE
print(pd.Series(y_train_res).value_counts())  # after SMOTE
