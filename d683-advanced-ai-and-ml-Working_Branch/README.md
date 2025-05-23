# Customer Churn Prediction with Explainable AI

This project predicts customer churn using an optimized machine learning pipeline (XGBoost) and explains key features behind predictions using SHAP values. It combines model training, evaluation, and interpretability into one cohesive project.

---

## Key Features

- Data preprocessing and feature engineering
-  XGBoost model training and hyperparameter tuning with GridSearchCV
-  Evaluation using accuracy, precision, recall, and F1 score
-  5-Fold cross-validation for robust performance
-  SHAP values for explainable AI and insight generation

---

## Tech Stack

- Python 3.11
- Libraries: `pandas`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`
- IDE: PyCharm
- OS: Windows 11 Pro
- CPU: Intel i7 (Dell Latitude 7400), 16 GB RAM

---

## 📁 Files

| File              | Description                                              |
|-------------------|----------------------------------------------------------|
| `preprocess.py`   | Cleans and prepares raw dataset for modeling             |
| `train_model.py`  | Trains model, evaluates performance, applies CV + tuning |
| `algorithm.py`    | Additional model logic (from earlier XGBoost prototype)  |
| `sample_customer_data.csv` | Example dataset for testing                    |

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python preprocess.py
python train_model.py
