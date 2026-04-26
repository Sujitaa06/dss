import pandas as pd
df=pd.read_csv('netflix_titles.csv')
df=df.dropna()
df=df.drop_duplicates()
y=df['type']
x=df[['rating','country','duration']]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

x=pd.get_dummies(x)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

dt= DecisionTreeClassifier()
rf=RandomForestClassifier()

lr=LogisticRegression(max_iter=1000)
knn=KNeighborsClassifier()

dt.fit(x_train,y_train)
rf.fit(x_train,y_train)
lr.fit(x_train,y_train)
knn.fit(x_train,y_train)

y_pred_dt = dt.predict(x_test)
y_pred_rf = rf.predict(x_test)
y_pred_lr = lr.predict(x_test)
y_pred_knn = knn.predict(x_test)

from sklearn.metrics import accuracy_score

acc_dt = accuracy_score(y_test,y_pred_dt)
acc_rf = accuracy_score(y_test,y_pred_rf)
acc_lr = accuracy_score(y_test,y_pred_lr)
acc_knn = accuracy_score(y_test,y_pred_knn)

print("Decision Tree:", acc_dt)
print("Random Forest:", acc_rf)
print("logistic Regression:", acc_lr)
print("kNeighbors:", acc_knn)

accuracies = {
    "Decision Tree": acc_dt,
    "Random Forest": acc_rf,
    "logistic Regression": acc_lr,
    "kNeighbors": acc_knn

}
best_model=max(accuracies, key=accuracies.get)

print("\nBest Model:",best_model)
print("Best Accuracy :",accuracies[best_model])