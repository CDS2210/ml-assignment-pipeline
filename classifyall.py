# 1 - Split into Validation and Train from the very beginning 

pca_n_components_perc = 4
outlier_contanimation_perc = 0.07

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Load training features
X = pd.read_csv('traindata.txt',header=None,sep=',')

# Load training labels 
y = pd.read_csv('trainlabels.txt',header=None,sep=',')

X.columns = [f'F{i+1}' for i in range(X.shape[1])]

X = X.copy()
y['Target'] = y
y = y.drop(y.columns[0], axis=1)

# Display shapes to verify 
print("Training features shape:", X.shape)
print("Training labels shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(X, y, stratify=y, test_size=0.2, random_state = 1)

print("Training features shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Validation features shape:", X_validate.shape)
print("Validation labels shape:", y_validate.shape)

print(y_train.value_counts(normalize=True))
print(y_validate.value_counts(normalize=True))

# 2 - Do Pre-processing and Model Building on True X_train only 

X_train.drop(columns=["F47"], axis=1, inplace=True)
# no value of this variable to prediction, only one group of 25 data points with the rest purely unique 
# remove the feature 

## we can see some of the features have missing values , we need to impute those values , since these are all numeric variables , we will impute the values
## using the mean 

# Store mean of all features in X_train
feature_means = {}

for feature in X_train.columns:
    if X_train[feature].dtype in ['float64', 'int64']:
        mean_val = X_train[feature].mean()
        feature_means[feature] = mean_val
        print(f"{feature} mean value is {mean_val}")

# Impute missing values with mean 

missing = X_train.columns[X_train.isnull().any()]
print("Columns with missing values:", missing)

for feature in missing:
    if X_train[feature].dtype in ['float64', 'int64']:  
        mean_val = X_train[feature].mean()
        print(f"{feature} mean value is {mean_val}")
        X_train[feature] = X_train[feature].fillna(mean_val)

# visualise distribution of features 
X_train.hist(figsize=(10, 8), bins=100, grid=False)

## from the above , it is clear that we need to deal with high dimensionality and outliers for our features , we will use scikit learn Isolation Forest
## we have high dimensionality which may mask outliers, we will use PCA to reduce dimensionality before using an Isolation forest to deal with outliers 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# 2.1 - Normalise 

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
X_train_normalized = scaler.fit_transform(X_train)

# X_train_normalized now contains standardized values (mean = 0, std = 1)
X_train_normalized_df = pd.DataFrame(X_train_normalized, columns=X_train.columns, index=X_train.index)

X_train_normalized_df.describe()

# 2.2 - Apply Principle Components Analysis to reduce dimensionality 

if pca_n_components_perc != -1:
    pca = PCA(n_components = pca_n_components_perc)
    X_train_normalized_pca = pca.fit_transform(X_train_normalized_df)
    # Wrap back into DataFrame with original index and new PC column names
    n_components = X_train_normalized_pca.shape[1]
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    X_train_pca_df = pd.DataFrame(
        X_train_normalized_pca,
        columns=pca_columns,
        index=X_train_normalized_df.index  
    )
    print(f"Original shape: {X_train_normalized_df.shape}")
    print(f"Reduced shape: {X_train_pca_df.shape}")
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
    plt.axvline(4, color='red', linestyle='--', label='Selected Components = 4')
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    X_train_pca_df = X_train_normalized_df

# 2.3 -  Isolation Forest for Outlier Detection

if outlier_contanimation_perc != -1:
    iso_forest = IsolationForest(
        contamination = outlier_contanimation_perc,    # contanimation of outliers 
        random_state = 1
    )
    iso_forest.fit(X_train_pca_df)
    predictions = iso_forest.predict(X_train_pca_df)
    outlier_indices = [i for i, val in enumerate(predictions) if val == -1]
    print(f"Number of outliers detected: {len(outlier_indices)}")
else:
    outlier_indices = []

df_combined = pd.concat([X_train_pca_df,y_train],axis=1)
df_combined_clean = df_combined.drop(df_combined.index[outlier_indices])
y_train.name = "Target" 
X_train_final = df_combined_clean.drop(columns=y_train.name)
y_train_final = df_combined_clean[y_train.name]

# Data pre-processing checks 
print(f"Original shape: {X_train.shape}")
print(f"Reduced shape after scaling: {X_train_normalized_df.shape}")
print(f"Reduced shape after scaling and PCA: {X_train_pca_df.shape}")
print(f"Reduced shape after scaling and PCA and Outlier Detection: {df_combined_clean.shape}")

# At this point, we did the following:
# - imported the datasets in and analysed their datatypes and shapes 
# - we realised that the categorical variable with Italian sentences was not needed so we removed it 
# - we then saw that all other variables are numeric and some had missing values 
# - we imputed the missing values using the mean
# - once that was done , we checked the distributions of the X data and saw there are outliers as well as too many features 
# - we then decided that we needed to apply PCA and outlier detection to address the above (PCA needs normalisation beforehand so we did that as well)
# -- We now need to shift our focus to the "y" , our target/dependent variable and observe what is trying to be classified before fitting a model , we might
# -- need to potentially balance classes if they are imbalanced
# -- Scikit learn classifiers have the class weight parameter which have an in-built calculation to deal with class imbalance, we will rely on that for balancing 

# visualise distribution of features 
X_train_final.hist(figsize=(10, 8), bins=100, grid=False)

# 3 - Building Predictive Model 

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RandomizedSearchCV

# 3.1 - Apply pre-processing to validation dataset 

X_validate.drop(columns=["F47"], axis=1, inplace=True) 
X_validate = X_validate.fillna(feature_means)

# Normalization 
X_validate_normalized = scaler.transform(X_validate)
X_validate_normalized_df = pd.DataFrame(X_validate_normalized, columns=X_validate.columns, index=X_validate.index)
X_validate_normalized_df.describe()

# Apply PCA transform
if pca_n_components_perc != -1:
    X_validate_normalized_pca = pca.transform(X_validate_normalized_df)
    # Wrap back into DataFrame with original index and new PC column names
    n_components = X_validate_normalized_pca.shape[1]
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    X_validate_pca_df = pd.DataFrame(
        X_validate_normalized_pca,
        columns=pca_columns,
        index=X_validate_normalized_df.index  
    )
    print(f"Original shape: {X_validate_normalized_df.shape}")
    print(f"Reduced shape: {X_validate_pca_df.shape}")
else:
    X_validate_pca_df = X_validate_normalized_df

X_validate_final = X_validate_pca_df
y_validate_final = y_validate

# 3.2 - Build Base Model
models = {
    "Random Forest": RandomForestClassifier(
#        class_weight='balanced', 
        random_state=1
    ),
    "Logistic Regression": LogisticRegression(
#        class_weight='balanced', 
        multi_class='multinomial', 
        solver='lbfgs', 
        max_iter=1000, 
        random_state=1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=1
    )
}

for name, model in models.items():
    if name == "Gradient Boosting":
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_final)
        model.fit(X_train_final, y_train_final, sample_weight=sample_weights)
    else:
        model.fit(X_train_final, y_train_final)

    y_pred = model.predict(X_validate_final)

    print(f"\n=== {name} ===")
    print(f"Validation Macro F1 Score: {f1_score(y_validate_final, y_pred, average='macro'):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_validate_final, y_pred))
    print("Classification Report:")
    print(classification_report(y_validate_final, y_pred, digits=4))

# 3.3 - Gradient Boosting Classifier
model = GradientBoostingClassifier(
        random_state=1
    )

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_final)
model.fit(X_train_final, y_train_final, sample_weight=sample_weights)

y_pred = model.predict(X_validate_final)

print(f"\n=== Gradient Boosting ===")
print(f"Validation Macro F1 Score: {f1_score(y_validate_final, y_pred, average='macro'):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_validate_final, y_pred))
print("Classification Report:")
print(classification_report(y_validate_final, y_pred, digits=4))

model.get_params()
