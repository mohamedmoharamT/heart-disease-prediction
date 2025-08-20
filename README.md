Heart Disease Prediction Project – UCI Dataset
Table of Contents

Project Overview

Dataset Description

Libraries and Tools

Data Loading and Inspection

Exploratory Data Analysis (EDA)

Data Preprocessing

Feature Selection

Model Building

Model Evaluation

Results and Interpretation

Future Improvements

Project Overview

This project aims to predict heart disease presence in patients using the UCI Heart Disease Dataset.
We apply various Machine Learning techniques to classify whether a patient is likely to have heart disease based on clinical attributes.

Goals:

Understand the dataset through Exploratory Data Analysis (EDA).

Preprocess and clean the data for machine learning models.

Train and evaluate multiple predictive models.

Compare model performance and interpret results.

Dataset Description

The dataset comes from the UCI Machine Learning Repository, containing clinical data from patients.

Number of rows: 303
Number of features: 14

Features:

age – Age of the patient

sex – Sex (1 = male, 0 = female)

cp – Chest pain type (4 values)

trestbps – Resting blood pressure (in mm Hg)

chol – Serum cholesterol in mg/dl

fbs – Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)

restecg – Resting electrocardiographic results (values 0,1,2)

thalach – Maximum heart rate achieved

exang – Exercise-induced angina (1 = yes, 0 = no)

oldpeak – ST depression induced by exercise relative to rest

slope – Slope of the peak exercise ST segment

ca – Number of major vessels (0–3) colored by fluoroscopy

thal – Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)

target – Diagnosis of heart disease (1 = presence, 0 = absence)

Libraries and Tools

The following Python libraries are used in this project:

pip install pandas numpy matplotlib seaborn scikit-learn plotly


Libraries Explanation:

pandas – Data manipulation and analysis

numpy – Numerical computations

matplotlib & seaborn – Data visualization

scikit-learn – Machine learning models and evaluation

plotly – Interactive visualizations

Data Loading and Inspection

Steps:

Load the dataset into a Pandas DataFrame.

Check the first few rows with .head().

Examine basic info with .info() and .describe().

Check for missing values and data types.

Exploratory Data Analysis (EDA)

Analyze distributions of features using histograms and boxplots.

Examine relationships using correlation heatmaps.

Compare feature values between patients with and without heart disease.

Identify potential outliers or anomalies in the dataset.

Use interactive visualizations with Plotly to better understand trends.

Data Preprocessing

Handle missing values (if any) using imputation techniques.

Encode categorical variables (cp, thal, slope, restecg) using One-Hot Encoding.

Scale features where necessary (e.g., StandardScaler or MinMaxScaler).

Split the dataset into training and testing sets.

Feature Selection

Use correlation analysis to remove redundant features.

Apply feature importance techniques (e.g., Random Forest importance) to identify influential predictors.

Optionally, use PCA for dimensionality reduction.

Model Building

Several machine learning models are trained and tested, including:

Logistic Regression – Baseline model for binary classification.

Decision Tree Classifier – Tree-based model for interpretability.

Random Forest Classifier – Ensemble method to improve performance.

Support Vector Machine (SVM) – Classification using hyperplanes.

K-Nearest Neighbors (KNN) – Distance-based classification.

Gradient Boosting / XGBoost – Advanced ensemble boosting method.

Each model is trained on the training set and tuned using cross-validation.

Model Evaluation

Evaluation metrics used:

Accuracy – Overall correctness of predictions.

Precision – Correct positive predictions among all predicted positives.

Recall (Sensitivity) – Correct positive predictions among all actual positives.

F1-Score – Harmonic mean of precision and recall.

ROC-AUC – Area under the Receiver Operating Characteristic curve.

Visualizations:

Confusion matrices for each model.

ROC curves for comparison.

Results and Interpretation

Summarize model performance metrics.

Identify the best-performing model based on accuracy and AUC.

Discuss which features are most influential for predicting heart disease.

Provide insights that could assist healthcare professionals.

Future Improvements

Increase dataset size to improve model generalization.

Apply hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

Experiment with deep learning models (e.g., neural networks) for better accuracy.

Deploy the model as a web application for real-time predictions.
