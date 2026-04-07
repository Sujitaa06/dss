import pandas as pd
import numpy as np

# -------------------------
# IMPORT LIBRARIES
# -------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Regression metrics
from sklearn.metrics import mean_squared_error, r2_score

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("TiTanic-Dataset.csv")

# -------------------------
# PREPROCESSING
# -------------------------
df.drop("Cabin", axis=1, inplace=True)

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

# =========================================================
# 🔵 CLASSIFICATION (TARGET = Survived)
# =========================================================

print("\n================ CLASSIFICATION ================")

X = df[["Age","Fare","SibSp","Pclass"]]
y = df["Survived"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------- Logistic Regression --------
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1 Score:", f1_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

# -------- Decision Tree --------
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("F1 Score:", f1_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))


# =========================================================
# 🟢 REGRESSION (TARGET = Fare)
# =========================================================

print("\n================ REGRESSION ================")

X_reg = df[["Age","SibSp","Pclass"]]
y_reg = df["Fare"]

# Scaling
X_reg = scaler.fit_transform(X_reg)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# -------- Linear Regression --------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\nLinear Regression")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# -------- KNN Regression --------
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("\nKNN Regression")
print("MSE:", mean_squared_error(y_test, y_pred_knn))
print("R2 Score:", r2_score(y_test, y_pred_knn))


# =========================================================
# 🟣 CLUSTERING (NO TARGET)
# =========================================================

print("\n================ CLUSTERING ================")

X_clust = df[["Age","Fare","SibSp","Pclass"]]

# Scaling
X_clust = scaler.fit_transform(X_clust)

# -------- KMeans --------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_clust)

print("\nKMeans Labels (first 10):")
print(kmeans_labels[:10])

# -------- Agglomerative --------
agg = AgglomerativeClustering(n_clusters=2)
agg_labels = agg.fit_predict(X_clust)

print("\nAgglomerative Labels (first 10):")
print(agg_labels[:10])

print("\n✅ ALL MODELS EXECUTED SUCCESSFULLY")