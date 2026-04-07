import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix
df=pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum().sort_values(ascending=False))
df.drop("Cabin",inplace=True,axis=1)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
df["Age"].fillna(df["Age"].median(),inplace=True)
print(df.isnull().sum().sort_values(ascending=False))
X=df[["Age","Fare","Pclass"]]
y=df["Survived"]
scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("logistic")
print("f1",f1_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred))
print("recall",recall_score(y_test,y_pred))
print("accuracy",accuracy_score(y_test,y_pred))
print("confusion",confusion_matrix(y_test,y_pred))



