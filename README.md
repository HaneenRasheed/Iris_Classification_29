# 🌸 Iris Flower Species Classification  
**Supervised Machine Learning using Logistic Regression**

---

## 📌 Problem Statement
The goal of this project is to classify iris flowers into three species - *Setosa*, *Versicolor*, and *Virginica*, based on their physical features such as sepal length, sepal width, petal length, and petal width. This is a supervised machine learning classification problem where the model learns from labeled data and predicts the species of unseen samples.

---

## 🛠️ Tools & Technologies Used
- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  

---

## ⚙️ Installation Steps

### 1. Clone the repository
```bash
git clone https://github.com/your-username/iris-classification.git
cd iris-classification
```
### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
```
Activate the environment:
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### ▶️ Execution Procedure
```bash
python iris_classification.py
```
### 📊 Output
🔹 Sample Console Output
```bash
Feature Names: ['sepal length (cm)', 'sepal width (cm)', ...]
Target Classes: ['setosa' 'versicolor' 'virginica']

Accuracy: ~100%

Classification Report:
              precision    recall  f1-score   support
setosa         1.00       1.00      1.00
versicolor     1.00       1.00      1.00
virginica      1.00       1.00      1.00
```
🔹 Confusion Matrix

A confusion matrix is generated and saved as:
```bash
confusion_matrix.png
```