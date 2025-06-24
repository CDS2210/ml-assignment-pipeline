#Machine Learning Classification Pipeline
This project was developed as part of a Machine Learning assignment at the University of the Witwatersrand. It implements a complete machine learning pipeline for supervised classification using scikit-learn, applied to a real-world tabular dataset.

#Project Overview
The main objectives were to:

Build a robust end-to-end pipeline for classification.

Perform data cleaning, imputation of missing values, and feature scaling.

Apply dimensionality reduction (PCA) to address high dimensionality.

Detect and handle outliers using Isolation Forest.

Train and evaluate multiple classification models:

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

Address class imbalance using sample weighting.

Evaluate model performance using:

Macro F1 Score

Confusion Matrix

Classification Report

#Tools & Libraries Used
Python 3.x

scikit-learn

pandas

numpy

matplotlib

#Pipeline Steps
Data Loading

Load training features and labels from CSV files.

Split into training and validation sets (80/20).

Preprocessing

Drop non-informative features.

Impute missing values with feature means.

Normalize data using StandardScaler.

Reduce dimensionality using PCA.

Detect and remove outliers using Isolation Forest.

#Model Training

Train models on the cleaned training set:

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

Handle class imbalance via sample weighting.

Model Evaluation

Evaluate models on the validation set.

Report F1 Scores, Confusion Matrices, and Classification Reports.
