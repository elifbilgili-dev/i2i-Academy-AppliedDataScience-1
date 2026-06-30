import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------------
# Load the Dataset
# -----------------------------------

df = pd.read_csv("breast_cancer_bd.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# -----------------------------------
# Data Cleaning
# -----------------------------------


df.replace("?", pd.NA, inplace=True)

df["Bare Nuclei"] = pd.to_numeric(df["Bare Nuclei"])


df["Bare Nuclei"] = df["Bare Nuclei"].fillna(df["Bare Nuclei"].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDataset Information After Cleaning:")
print(df.info())

# -----------------------------------
# Feature Selection
# -----------------------------------

# Features (X) and Target (y)
X = df.drop(columns=["Sample code number", "Class"])
y = df["Class"]

# -----------------------------------
# Split the Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------
# Feature Scaling
# -----------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData has been successfully split into training and testing sets.")
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# -----------------------------------
# Logistic Regression
# -----------------------------------

logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

logistic_model.fit(X_train, y_train)

y_pred_lr = logistic_model.predict(X_test)

print("\n========== Logistic Regression ==========")
print("Accuracy Score:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# -----------------------------------
# K-Nearest Neighbors (KNN)
# -----------------------------------

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

print("\n========== K-Nearest Neighbors (KNN) ==========")
print("Accuracy Score:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# ---------------------------------------------------------
# Conclusion
#
# Two classification algorithms were compared:
# Logistic Regression and K-Nearest Neighbors (KNN).
#
# Logistic Regression achieved an accuracy of 96.43%,
# while KNN achieved 97.14%.
#
# KNN performed slightly better because it correctly
# classified more samples than Logistic Regression.
# Therefore, KNN was the best-performing model
# for this dataset.
# ---------------------------------------------------------