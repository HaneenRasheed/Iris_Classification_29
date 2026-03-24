# Iris Flower Species Classification
# Supervised Machine Learning - Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Loading The Dataset 
iris = load_iris()
X = iris.data
y = iris.target

print("Feature Names:", iris.feature_names)
print("Target Classes:", iris.target_names)
print("Dataset Shape:", X.shape)

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]
print("\nFirst 5 rows:")
print(df.head())

# 2. Spliting The Dataset (80% train, 20% test) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# 3. Preprocessing 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 4. Train Logistic Regression Model
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
print("\nModel training complete.")

# 5. Evaluating the Model 
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Iris Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion matrix saved as confusion_matrix.png")
