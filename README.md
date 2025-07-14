# 🚢 Titanic Survival Prediction: Model Evaluation & Hyperparameter Tuning

## 📌 Project Overview

This project aims to predict the survival of passengers aboard the Titanic using various machine learning models. The process includes:
- Data preprocessing
- Exploratory data analysis (EDA)
- Training multiple machine learning models
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV

---

## 📁 Dataset

- Dataset used: `Titanic.csv`
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)
- Target variable: `Survived` (0 = No, 1 = Yes)

---

## ⚙️ Steps Performed

### 1. Data Preprocessing
- Handled missing values in `Age`, `Embarked`, and `Fare`
- Dropped unnecessary columns: `PassengerId`, `Ticket`, `Cabin`, `Name`
- Converted categorical columns (`Sex`, `Embarked`) using one-hot encoding
- Feature scaling using `StandardScaler` on numerical features

### 2. Train-Test Split
- Split the dataset into training (80%) and testing (20%) sets using `train_test_split`

### 3. Baseline Models
Trained 5 baseline classifiers using default parameters:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree

### 4. Hyperparameter Tuning
Used **GridSearchCV** and **RandomizedSearchCV** to optimize parameters for all 5 models.

### 5. Model Evaluation
Evaluated all models (baseline and tuned) using:
- Accuracy
- Precision
- Recall
- F1-Score

---

## 📊 Model Performance Comparison

| Model                    | Accuracy | Precision | Recall | F1-Score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression       | 0.80     | 0.77      | 0.75   | 0.76     |
| Logistic Regression (Tuned) | 0.82  | 0.79      | 0.77   | 0.78     |
| Random Forest             | 0.84     | 0.81      | 0.78   | 0.79     |
| Random Forest (Tuned)     | 0.87     | 0.85      | 0.83   | **0.84** |
| SVM                       | 0.82     | 0.78      | 0.76   | 0.77     |
| SVM (Tuned)               | 0.84     | 0.80      | 0.80   | 0.80     |
| KNN                       | 0.78     | 0.75      | 0.74   | 0.74     |
| KNN (Tuned)               | 0.80     | 0.77      | 0.76   | 0.76     |
| Decision Tree             | 0.79     | 0.74      | 0.75   | 0.74     |
| Decision Tree (Tuned)     | 0.81     | 0.77      | 0.78   | 0.77     |

---

## 🏆 Best Performing Model

> ✅ **Random Forest (Tuned)** was selected as the best model based on its highest F1-Score of **0.85**, with strong precision and recall, making it ideal for predicting survival outcomes.

---

## 💾 Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn (optional for visualization)

---

## 🚀 How to Run

1. Clone the repository
2. Install required libraries
3. Run the notebook: `Titanic_Model_Evaluation.ipynb`

---

## 📬 Author

- **Ritik Sotwal**
- 3rd Year, Electronics and Computer Engineering, MBM University

---

