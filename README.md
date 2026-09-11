# 🛒 Online Shoppers Purchase Intention Prediction



\[!\[Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)

\[!\[Flask](https://img.shields.io/badge/Flask-Web%20App-black)](https://flask.palletsprojects.com/)

\[!\[Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)

\[!\[Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7)](https://render.com/)



An end-to-end machine learning project that predicts whether an online shopping session is likely to result in a purchase.



The final \*\*Gradient Boosting\*\* model is deployed as a \*\*Flask web application\*\* and hosted on \*\*Render\*\*.



An end-to-end machine learning project that predicts whether an online shopping session is likely to result in a purchase.

The final **Gradient Boosting** model is deployed as a **Flask web application** and hosted on **Render**.

## 🚀 Live Demo

### [Open the Live Application](https://online-shoppers-purchase-prediction.onrender.com/)

Enter shopping-session information and receive:

* **Purchase / No Purchase prediction**
* **Estimated purchase probability**

## 🎯 Project Objective

The objective is to predict whether an online shopping session will result in a purchase (`Revenue = 1`) or not (`Revenue = 0`).

The dataset contains **12,330 online shopping sessions** and **17 predictor variables**.

The target variable is imbalanced:

|Outcome|Percentage|
|-|-:|
|No Purchase|84.53%|
|Purchase|15.47%|

Because of this imbalance, model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Train-Test Performance Gap

## 🔄 Machine Learning Workflow

```text
Data Understanding
       ↓
Train-Test Split
       ↓
Categorical Variable Handling
       ↓
SMOTENC on Training Data
       ↓
Baseline Models
       ↓
Ensemble Models
       ↓
Hyperparameter Tuning
       ↓
Final Model Selection
       ↓
Model Serialization (.pkl)
       ↓
Flask Deployment
       ↓
GitHub
       ↓
Render Cloud Hosting
```

## 🤖 Models Evaluated

### Baseline Models

* Logistic Regression
* Decision Tree (Gini)
* Decision Tree (Entropy)
* SVM (Linear)
* SVM (RBF)
* SVM (Polynomial)
* SVM (Sigmoid)

### Ensemble Models

* Bagging
* Random Forest
* Extra Trees
* AdaBoost
* Gradient Boosting
* XGBoost
* Stacking

## 🏆 Final Model

The final deployed model is **Gradient Boosting**.

### Final Untuned Gradient Boosting Performance

|Metric|Score|
|-|-:|
|Test Accuracy|**0.8840**|
|Precision|**0.6017**|
|Recall|**0.7435**|
|F1-Score|**0.6651**|
|Train-Test Gap|**0.0392**|

Gradient Boosting achieved the highest F1-score among the evaluated models while maintaining strong accuracy and a relatively small train-test gap.

## ⚙️ Hyperparameter Tuning

Gradient Boosting, XGBoost and AdaBoost were tuned using **GridSearchCV with 5-fold cross-validation**.

The search used **F1-score** because the target class was imbalanced.

### Gradient Boosting

Best parameters:

```text
learning\\\_rate = 0.2
max\\\_depth = 3
n\\\_estimators = 100
```

### XGBoost

Best parameters:

```text
learning\\\_rate = 0.2
max\\\_depth = 4
n\\\_estimators = 150
```

### AdaBoost

Best parameters:

```text
learning\\\_rate = 0.2
n\\\_estimators = 150
estimator\\\_\\\_max\\\_depth = 2
```

The final deployed model remained the **untuned Gradient Boosting model**, as it provided the strongest overall test-set balance.

## 🧠 Model Improvement Journey

### 1\. Baseline Models

The initial models showed that class imbalance made the classifiers relatively conservative about predicting purchases. The Decision Trees also showed clear overfitting, with training accuracy reaching 1.00 while test performance was substantially lower.

### 2\. SMOTENC

SMOTENC was applied only to the training data to balance the purchase and non-purchase classes.

This substantially improved minority-class detection, especially recall and F1 for several models.

### 3\. Ensembling

Ensemble methods improved the overall trade-off between accuracy, precision and recall.

Gradient Boosting emerged as the strongest ensemble:

* Accuracy: **0.8840**
* Precision: **0.6017**
* Recall: **0.7435**
* F1: **0.6651**

Stacking did not outperform Gradient Boosting and showed a substantially larger train-test gap.

### 4\. Tuning

Gradient Boosting, XGBoost and AdaBoost were tuned.

Tuning improved some individual metrics, but the final comparison showed that the **untuned Gradient Boosting model** remained the strongest overall choice for deployment.

## 🚀 Deployment Architecture

The deployed application follows a train-once, serve-many architecture:

```text
Saved Model + Preprocessing
            ↓
         Flask App
            ↓
     User Input Form
            ↓
      Preprocessing
            ↓
   Gradient Boosting Model
            ↓
 Prediction + Probability
```

SMOTENC is used **only during training**. It is not applied to new user inputs during prediction.

## 📦 Project Structure

```text
online-shoppers-purchase-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
└── model/
    ├── gradient\\\_boosting\\\_model.pkl
    ├── ordinal\\\_encoder.pkl
    └── preprocessor.pkl
```

## 💻 Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Flask:

```bash
python -m flask --app app run
```

Then open:

```text
http://127.0.0.1:5000
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Flask
* Joblib
* Git
* GitHub
* Render

## 🌐 Deployment

The application is connected to GitHub and deployed on Render.

**Live URL:**

https://online-shoppers-purchase-prediction.onrender.com/

## 👤 Project

Built as an end-to-end machine learning project covering:

**Data Preparation → Imbalanced Learning → Model Comparison → Ensembling → Hyperparameter Tuning → Model Serialization → Flask Deployment → Cloud Hosting**

