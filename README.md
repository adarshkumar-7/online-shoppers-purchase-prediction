\# Online Shoppers Purchase Intention Prediction



An end-to-end machine learning project that predicts whether an online shopping session is likely to result in a purchase.



The final Gradient Boosting model is deployed as a Flask web application and hosted on Render.



\## Live Application



\*\*https://online-shoppers-purchase-prediction.onrender.com/\*\*



\## Project Objective



The objective is to predict whether an online shopping session will result in a purchase (`Revenue = 1`) or not (`Revenue = 0`).



The dataset contains 12,330 online shopping sessions and 17 predictor variables.



The target variable is imbalanced:



\- No Purchase: 84.53%

\- Purchase: 15.47%



Because of this imbalance, model evaluation considers:



\- Accuracy

\- Precision

\- Recall

\- F1-Score

\- Train-Test Performance Gap



\## Machine Learning Workflow



The project follows an end-to-end machine learning workflow:



1\. Data loading and exploration

2\. Train-test split with stratification

3\. Categorical variable handling

4\. SMOTENC for class balancing on the training set

5\. Model training and comparison

6\. Ensemble modelling

7\. Hyperparameter tuning

8\. Final model selection

9\. Model serialization using `.pkl`

10\. Flask deployment

11\. Cloud hosting using Render



\## Models Evaluated



\### Baseline Models



\- Logistic Regression

\- Decision Tree (Gini)

\- Decision Tree (Entropy)

\- SVM (Linear)

\- SVM (RBF)

\- SVM (Polynomial)

\- SVM (Sigmoid)



\### Ensemble Models



\- Bagging

\- Random Forest

\- Extra Trees

\- AdaBoost

\- Gradient Boosting

\- XGBoost

\- Stacking



\## Final Model



The final deployed model is \*\*Gradient Boosting\*\*.



\### Final Untuned Gradient Boosting Performance



| Metric | Score |

|---|---:|

| Test Accuracy | 0.8840 |

| Precision | 0.6017 |

| Recall | 0.7435 |

| F1-Score | 0.6651 |

| Train-Test Gap | 0.0392 |



Gradient Boosting achieved the highest F1-score among the models evaluated while maintaining strong accuracy and a relatively small train-test gap.



\## Hyperparameter Tuning



Gradient Boosting, XGBoost and AdaBoost were further tuned using GridSearchCV with 5-fold cross-validation.



The search was performed using F1-score because the target class was imbalanced.



\### Gradient Boosting



Best parameters:



```text

learning\_rate = 0.2

max\_depth = 3

n\_estimators = 100

XGBoost



Best parameters:



learning\_rate = 0.2

max\_depth = 4

n\_estimators = 150

AdaBoost



Best parameters:



learning\_rate = 0.2

n\_estimators = 150

estimator\_\_max\_depth = 2



Although tuning improved some metrics for individual models, the untuned Gradient Boosting model remained the strongest overall model based on the final test-set comparison.



Deployment



The trained model and preprocessing components are serialized using joblib.



The Flask application:



Loads the saved model and preprocessing artifacts

Accepts new shopping-session information

Applies the same preprocessing used during training

Generates a purchase prediction

Returns the estimated purchase probability



The application is hosted publicly using Render.



Project Structure

online-shoppers-purchase-prediction/

│

├── app.py

├── requirements.txt

├── README.md

│

└── model/

&#x20;   ├── gradient\_boosting\_model.pkl

&#x20;   ├── ordinal\_encoder.pkl

&#x20;   └── preprocessor.pkl

Run Locally



Install the required packages:



pip install -r requirements.txt



Run the Flask application:



python -m flask --app app run



Then open:



http://127.0.0.1:5000

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Imbalanced-learn

XGBoost

Flask

Joblib

Git / GitHub

Render

Deployment



The application is connected to GitHub and deployed on Render.



Live URL:



https://online-shoppers-purchase-prediction.onrender.com/







