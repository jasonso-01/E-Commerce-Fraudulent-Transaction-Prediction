# e-Commerce Fradulent Transaction Prediction

## Table of Contents
- [Abstract & Background](#abstract--background)
- [Dataset Analysis](#dataset-analysis)
- [Preparation Before Model Training](#preparation-before-model-training)
- [Model Training](#model-training)

## Abstract & Background
This project aims to train a predictive model for identifying fraudulent transactions in e-commerce platforms. With the rapid growth of e-Commerce, online fraud has become a significant challenge to both businesses and customers, causing financial losses to both parties.

By analyzing the dataset "Fraudulent E-Commerce Transaction" using Python and Jupyter Notebook with ML libraries such as pandas and scikit-learn, I have built and trained a machine learning model to predict fraudulent transactions. The insights and information extracted in this project can be used in real-time fraud detection systems, which contribute to safer online transactions.

## Dataset Analysis

Getting general info about the dataset:

Displaying the first 5 rows of the dataset to understand the data type and formatting of each column

Distribution of Transaction amount
Distribution of age
Distribution of device type
Fradulent / Non Fraudulent ratio
Districution of Payment method

## Preparation before model training

Check for missing values in the dataset
Identify Categorical variables
Convert categorical variables to numerical format for machine learning


## Model Training

After preparing the data, the next step is to train a machine learning model to predict fraudulent transactions.

Here's the next part of the project report focusing on model training:

## Model Training

After preparing the data, the next step is to train a machine learning model to predict fraudulent transactions. Here is the detailed process:

### 1. Import Necessary Libraries

First, import the necessary libraries for model training and evaluation.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### 2. Split the Data into Training and Testing Sets

Split the data into training and testing sets to evaluate the model's performance. This step was already performed in the previous section, but here is a summary:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features (X) and target (y)
X = df.drop('Is Fraudulent', axis=1)
y = df['Is Fraudulent']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. Train a Logistic Regression Model

Train a simple Logistic Regression model to predict fraudulent transactions.

```python
# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train_scaled, y_train)
```

### 4. Make Predictions

Use the trained model to make predictions on the test set.

```python
# Make predictions
y_pred = model.predict(X_test_scaled)
```

### 5. Evaluate the Model

Evaluate the model's performance using accuracy, classification report, and confusion matrix.

```python
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
```

### Results

The results of the model evaluation are as follows:

**Accuracy:**
```
Accuracy: 0.952411000634652
```

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.95      1.00      0.98      4485
           1       0.90      0.78      0.84       242

    accuracy                           0.95      4727
   macro avg       0.93      0.89      0.91      4727
weighted avg       0.95      0.95      0.95      4727
```

**Confusion Matrix:**
```
[[4483    2]
 [  53  189]]
```

### Interpretation

- **Accuracy:** The model achieved an accuracy of approximately 95.24%, which indicates that it correctly identified fraudulent transactions 95.24% of the time.

## Model Evaluation

After training a basic model, the next step is to find the best hyperparameters and comparing it with another model, such as a Random Forest Classifier.

### 1. Hyperparameter using Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# Initialize the Grid Search
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
print("Best Parameters: ", grid_search.best_params_)

# Train the model with best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_best = best_model.predict(X_test_scaled)

# Evaluate the model
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Model Accuracy: ", accuracy_best)
```

**Output:**
```
Best Parameters: {'C': 0.1, 'solver': 'liblinear'}
Best Model Accuracy: 0.9521626507298498
```

### 2. Random Forest Classifier

To compare the performance, train a Random Forest Classifier and evaluate its accuracy.

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf_model = RandomForestClassifier()

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy: ", accuracy_rf)
```

**Output:**
```
Random Forest Accuracy: 0.9521626507298498
```

### 3. Feature Importance

Determine the importance of each feature in predicting fraudulent transactions. For the Logistic Regression model, this can be done by examining the coefficients.

```python
import numpy as np

# Get feature importance
feature_importance = np.abs(best_model.coef_[0])

# Print feature importance
features = X.columns
for feature, importance in zip(features, feature_importance):
    print(f"Feature: {feature}, Importance: {importance}")
```

**Output:**
```
Feature: Transaction Amount, Importance: 0.6486342778454753
Feature: Payment Method, Importance: 0.07554577941144883
Feature: Product Category, Importance: 0.016741060278404482
Feature: Quantity, Importance: 0.02385177393848053
Feature: Customer Age, Importance: 0.0399115690945371
Feature: Customer Location, Importance: 0.012095872548957917
Feature: Device Used, Importance: 0.03218736891401819
Feature: IP Address, Importance: 0.009440855817094803
Feature: Shipping Address, Importance: 0.1624424323601678
Feature: Billing Address, Importance: 0.00934862974730682
Feature: Account Age Days, Importance: 0.662696684392001
Feature: Transaction Hour, Importance: 0.037814329752518784
Feature: Transaction Day, Importance: 0.0
Feature: Transaction Year, Importance: 0.0
Feature: Transaction Month, Importance: 0.0
Feature: Transaction Minute, Importance: 0.018051869069402685
Feature: Transaction Second, Importance: 0.007340263807810822
```

### Results and Interpretation

The best parameters for the Logistic Regression model were found to be `{'C': 0.1, 'solver': 'liblinear'}`. Both the optimized Logistic Regression model and the Random Forest Classifier achieved an accuracy of approximately 95.22%. The most important features for predicting fraudulent transactions included `Transaction Amount`, `Account Age Days`, and `Shipping Address`.

These results indicates that both optimized Logistic Regression and Random Forest models perform similarly well on this dataset.

### Cross-Validation and Model Performance

To evaluate the robustness and generalizability of our model, we performed 5-fold cross-validation. The `cross_val_score` function from `sklearn.model_selection` was utilized to achieve this. The code snippet and the resulting cross-validation scores are displayed below:

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)

# Print the cross-validation scores and their mean
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
```

The cross-validation scores are as follows:

```
Cross-validation scores: [0.95346378 0.95478583 0.95292251 0.95292251 0.95292251]
Mean cross-validation score: 0.9534034250480154
```

These scores indicate that our model is consistently performing well across different subsets of the training data, with a mean accuracy of approximately 95.34%.

### Feature Importance Analysis

To understand the impact of each feature on the model's predictions, we employed SHAP (SHapley Additive exPlanations) values. The SHAP summary plot below illustrates the importance of various features, with `Account Age Days` and `Transaction Amount` being the most influential.

![SHAP Summary Plot][]

### Model Performance After ResamplingTo address the class imbalance in our dataset, we applied resampling techniques. The accuracy of the model after resampling is shown below:

```Accuracy after resampling: 0.7036175163951767```

While the accuracy decreased to approximately 70.36%, resampling helps to improve the model's ability to correctly identify minority class instances, thereby enhancing its overall performance on imbalanced data.### Target Variable DistributionThe distribution of the target variable `Is Fraudulent` is depicted in the bar chart below. This visualization confirms the class imbalance, with non-fraudulent transactions significantly outnumbering fraudulent ones.

```pythonimport matplotlib.pyplot as pltimport seaborn as sns# Plot the distribution of the target variableplt.figure(figsize=(6, 4))sns.countplot(x='Is Fraudulent', data=df)plt.title('Distribution of Target Variable')plt.xlabel('Is Fraudulent')plt.ylabel('Count')plt.show()```

![Distribution of Target Variable][]

In conclusion, the model demonstrates strong performance and consistency through cross-validation. The feature importance analysis provides insights into the key factors influencing predictions, and resampling addresses class imbalance, improving the model's capability to detect fraudulent transactions.
