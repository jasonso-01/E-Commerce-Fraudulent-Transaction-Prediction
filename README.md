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
- **Precision and Recall:** The `precision` for class 0 (non-fraudulent) is 0.95, and for class 1 (fraudulent) is 0.90. The `recall` for class 0 is 1.00, and for class 1 is 0.78. This suggests that the model is very good at identifying non-fraudulent transactions but has some difficulty with fraudulent ones.
- **Confusion Matrix:** The confusion matrix shows that there are 4483 true negatives, 2 false positives, 53 false negatives, and 189 true positives. This indicates that the model is very effective at avoiding false positives but has room for improvement in reducing false negatives.

These results suggest that the Logistic Regression model performs well on this dataset, but further optimization and experimentation with more advanced models or additional features may be needed to improve the detection of fraudulent transactions further.
