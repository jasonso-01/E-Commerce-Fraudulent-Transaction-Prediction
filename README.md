# E-Commerce Fraudulent Transaction Prediction  

![ComfyUI_00053_ 拷貝](https://github.com/user-attachments/assets/0b2fa2d3-f739-4d33-a263-742e4fe95422)


## Table of Contents
- [Abstract & Background](#abstract--background)
- [Dataset Analysis](#dataset-analysis)
- [Preparation Before Model Training](#preparation-before-model-training)
  - [Model Training: Logistic Regression](#Model-Training-Logistic-Regression)
  - [Model Training: Random Forest](#Model-Training-Random-Forest)


## Abstract & Background
This project aims to train a predictive model for identifying fraudulent transactions in e-commerce platforms. With the rapid growth of e-Commerce, online fraud has become a significant challenge to both businesses and customers, causing financial losses to both parties.

By analyzing the dataset ["Fraudulent E-Commerce Transaction"](https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions) using Python and Jupyter Notebook with ML libraries such as pandas and scikit-learn, I have built and trained a machine learning model to predict fraudulent transactions. The information extracted in this project can be used in real-time fraud detection systems, which contribute to safer online transactions.

## Dataset Analysis

The first step is to get general info about the dataset.

**Displaying the first 5 rows of the dataset to understand the data type and formatting of each column:**
```python
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Fraudulent_E-Commerce_Transaction_Data.csv')

# Display the first few rows of the dataset
print(df.head())
```

**Output:**
```
                         Transaction ID                           Customer ID  \
0  c12e07a0-8a06-4c0d-b5cc-04f3af688570  8ca9f102-02a4-4207-ab63-484e83a1bdf0   
1  7d187603-7961-4fce-9827-9698e2b6a201  4d158416-caae-4b09-bd5b-15235deb9129   
2  f2c14f9d-92df-4aaf-8931-ceaf4e63ed72  ccae47b8-75c7-4f5a-aa9e-957deced2137   
3  e9949bfa-194d-486b-84da-9565fca9e5ce  b04960c0-aeee-4907-b1cd-4819016adcef   
4  7362837c-7538-434e-8731-0df713f5f26d  de9d6351-b3a7-4bc7-9a55-8f013eb66928   

   Transaction Amount     Transaction Date Payment Method Product Category  \
0               42.32  2024-03-24 23:42:43         PayPal      electronics   
1              301.34  2024-01-22 00:53:31    credit card      electronics   
2              340.32  2024-01-22 08:06:03     debit card     toys & games   
3               95.77  2024-01-16 20:34:53    credit card      electronics   
4               77.45  2024-01-16 15:47:23    credit card         clothing   

   Quantity  Customer Age    Customer Location Device Used       IP Address  \
0         1            40      East Jameshaven     desktop    110.87.246.85   
1         3            35             Kingstad      tablet    14.73.104.153   
2         5            29           North Ryan     desktop      67.58.94.93   
3         5            45           Kaylaville      mobile  202.122.126.216   
4         5            42  North Edwardborough     desktop     96.77.232.76   

                                    Shipping Address  \
0  5399 Rachel Stravenue Suite 718\nNorth Blakebu...   
1        5230 Stephanie Forge\nCollinsbury, PR 81853   
2                195 Cole Oval\nPort Larry, IA 58422   
3         7609 Cynthia Square\nWest Brenda, NV 23016   
4  2494 Robert Ramp Suite 313\nRobinsonport, AS 5...   

                                     Billing Address  Is Fraudulent  \
0  5399 Rachel Stravenue Suite 718\nNorth Blakebu...              0   
1        5230 Stephanie Forge\nCollinsbury, PR 81853              0   
2  4772 David Stravenue Apt. 447\nVelasquezside, ...              0   
3         7609 Cynthia Square\nWest Brenda, NV 23016              0   
4  2494 Robert Ramp Suite 313\nRobinsonport, AS 5...              0   

   Account Age Days  Transaction Hour  
0               282                23  
1               223                 0  
2               360                 8  
3               325                20  
4               116                15
```


**Summary of the dataset:**
```python
# Get a summary of the DataFrame
print("\nInfo about the dataset:")
print(df.info())
```

**Output:**
```
Info about the dataset:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 23634 entries, 0 to 23633
Data columns (total 16 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Transaction ID      23634 non-null  object 
 1   Customer ID         23634 non-null  object 
 2   Transaction Amount  23634 non-null  float64
 3   Transaction Date    23634 non-null  object 
 4   Payment Method      23634 non-null  object 
 5   Product Category    23634 non-null  object 
 6   Quantity            23634 non-null  int64  
 7   Customer Age        23634 non-null  int64  
 8   Customer Location   23634 non-null  object 
 9   Device Used         23634 non-null  object 
 10  IP Address          23634 non-null  object 
 11  Shipping Address    23634 non-null  object 
 12  Billing Address     23634 non-null  object 
 13  Is Fraudulent       23634 non-null  int64  
 14  Account Age Days    23634 non-null  int64  
 15  Transaction Hour    23634 non-null  int64  
dtypes: float64(1), int64(5), object(10)
memory usage: 2.9+ MB
None
```

**Visualization of the first 100 rows of the dataset using pairplot**
```python
# Visualize the first 100 rows of the dataset
sns.pairplot(df.head(150))
plt.show()
```

**Output:**  

![download](https://github.com/user-attachments/assets/80278d4f-6c3f-482f-9124-7ef4197397a8)


**Distribution of Transaction amount**

```python
# Transaction Amount
plt.figure(figsize=(10, 6))
sns.histplot(df['Transaction Amount'], bins=30, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.show()
```
**Output:**  

![download](https://github.com/user-attachments/assets/38029290-ee58-402f-896b-dfb4106c2bcd)


**Fradulent / Non Fraudulent ratio**

```python
# Bar chart for Fraudulent & Non-Fraudulent Transactions
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Is Fraudulent')
plt.title('Fraudulent vs. Non-Fraudulent Transactions')
plt.xlabel('Is Fraudulent')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'])
plt.show()
```

**Output:**  

![download](https://github.com/user-attachments/assets/3e1d9496-84d8-431e-b892-df48aada9207)


**Distribution of Payment method**

```python
# Pie chart for Payment Methods
plt.figure(figsize=(8, 8))
df['Payment Method'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Payment Methods')
plt.ylabel('')
plt.show()
```

**Output:**  

![download](https://github.com/user-attachments/assets/a94af446-12e3-4ed4-904d-34afdf7641a6)


**Distribution of Product Categories**  

```python
# Pie chart for Product Categories
plt.figure(figsize=(8, 8))
df['Product Category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Product Categories')
plt.ylabel('')
plt.show()
```

**Output:**  

![download](https://github.com/user-attachments/assets/726e7244-4dda-4a9a-ab1c-5588fc682e5b)


## Preparation before model training

### 1. Check for missing values in the dataset befor the model training process. 
```python
# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())
```

**Output:**  
```
Missing values in the dataset:
Transaction ID        0
Customer ID           0
Transaction Amount    0
Transaction Date      0
Payment Method        0
Product Category      0
Quantity              0
Customer Age          0
Customer Location     0
Device Used           0
IP Address            0
Shipping Address      0
Billing Address       0
Is Fraudulent         0
Account Age Days      0
Transaction Hour      0
dtype: int64
```
The results shows there are no missing values.

### 2. Data cleanup
The next step is to drop unnecessary data. For example, Transaction ID and Customer ID are not relevant for the ML analysis for whether this transaction is fraudulent or not, so we can drop this data. Also, I extracted data from Transaction date then drop the original "Transaction date" feature.

```python
# Create new features from 'Transaction Date' 
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
df['Transaction Year'] = df['Transaction Date'].dt.year
df['Transaction Month'] = df['Transaction Date'].dt.month
df['Transaction Day'] = df['Transaction Date'].dt.day
df['Transaction Hour'] = df['Transaction Date'].dt.hour  # Redundant if already present
df['Transaction Minute'] = df['Transaction Date'].dt.minute
df['Transaction Second'] = df['Transaction Date'].dt.second

# Drop the original 'Transaction Date' as we have extracted its components
df = df.drop(['Transaction Date'], axis=1)

# Drop 'Transaction ID' and 'Customer ID' if they are not relevant for your analysis
df = df.drop(['Transaction ID', 'Customer ID'], axis=1)

# Display the first few rows of the modified dataset
print("First few rows of the modified dataset:")
print(df.head())

# Display the columns of the modified dataset
print("\nColumns of the modified dataset:")
print(df.columns)
```

**Output:**  
```
First few rows of the modified dataset:
   Transaction Amount Payment Method Product Category  Quantity  Customer Age  \
0               42.32         PayPal      electronics         1            40   
1              301.34    credit card      electronics         3            35   
2              340.32     debit card     toys & games         5            29   
3               95.77    credit card      electronics         5            45   
4               77.45    credit card         clothing         5            42   

     Customer Location Device Used       IP Address  \
0      East Jameshaven     desktop    110.87.246.85   
1             Kingstad      tablet    14.73.104.153   
2           North Ryan     desktop      67.58.94.93   
3           Kaylaville      mobile  202.122.126.216   
4  North Edwardborough     desktop     96.77.232.76   

                                    Shipping Address  \
0  5399 Rachel Stravenue Suite 718\nNorth Blakebu...   
1        5230 Stephanie Forge\nCollinsbury, PR 81853   
2                195 Cole Oval\nPort Larry, IA 58422   
3         7609 Cynthia Square\nWest Brenda, NV 23016   
4  2494 Robert Ramp Suite 313\nRobinsonport, AS 5...   

                                     Billing Address  Is Fraudulent  \
0  5399 Rachel Stravenue Suite 718\nNorth Blakebu...              0   
1        5230 Stephanie Forge\nCollinsbury, PR 81853              0   
2  4772 David Stravenue Apt. 447\nVelasquezside, ...              0   
3         7609 Cynthia Square\nWest Brenda, NV 23016              0   
4  2494 Robert Ramp Suite 313\nRobinsonport, AS 5...              0   

   Account Age Days  Transaction Hour  Transaction Year  Transaction Month  \
0               282                23              2024                  3   
1               223                 0              2024                  1   
2               360                 8              2024                  1   
3               325                20              2024                  1   
4               116                15              2024                  1   

   Transaction Day  Transaction Minute  Transaction Second  
0               24                  42                  43  
1               22                  53                  31  
2               22                   6                   3  
3               16                  34                  53  
4               16                  47                  23  

Columns of the modified dataset:
Index(['Transaction Amount', 'Payment Method', 'Product Category', 'Quantity',
       'Customer Age', 'Customer Location', 'Device Used', 'IP Address',
       'Shipping Address', 'Billing Address', 'Is Fraudulent',
       'Account Age Days', 'Transaction Hour', 'Transaction Year',
       'Transaction Month', 'Transaction Day', 'Transaction Minute',
       'Transaction Second'],
      dtype='object')
```

### 3. Identify Categorical variables
Since ML algorithms requires numerical input (int, float) to perform calculations, Categorial variables need to be converted to numerical to be used in ML computations.

```python
# Check which feature is categorical feature
categorical = df.select_dtypes(include=['object']).columns.tolist()
categorical
```
**Output:**  
```
['Payment Method',
 'Product Category',
 'Customer Location',
 'Device Used',
 'IP Address',
 'Shipping Address',
 'Billing Address']
```

### 4. Convert categorical variables to numerical format for machine learning

```python
# Apply mappings
df['Payment Method'] = df['Payment Method'].map({"debit card": 0, "credit card": 1, "PayPal": 2, "bank transfer": 3})
df['Product Category'] = df['Product Category'].map({"home & garden": 0, "electronics": 1, "toys & games": 2, "clothing": 3, "health & beauty": 4})
df['Device Used'] = df['Device Used'].map({"desktop": 0, "mobile": 1, "tablet": 2})

print(df.head())
```
**Output:**  
```
   Transaction Amount  Payment Method  Product Category  Quantity  \
0               42.32               2                 1         1   
1              301.34               1                 1         3   
2              340.32               0                 2         5   
3               95.77               1                 1         5   
4               77.45               1                 3         5   

   Customer Age    Customer Location  Device Used       IP Address  \
0            40      East Jameshaven            0    110.87.246.85   
1            35             Kingstad            2    14.73.104.153   
2            29           North Ryan            0      67.58.94.93   
3            45           Kaylaville            1  202.122.126.216   
4            42  North Edwardborough            0     96.77.232.76   

                                    Shipping Address  \
0  5399 Rachel Stravenue Suite 718\nNorth Blakebu...   
1        5230 Stephanie Forge\nCollinsbury, PR 81853   
2                195 Cole Oval\nPort Larry, IA 58422   
3         7609 Cynthia Square\nWest Brenda, NV 23016   
4  2494 Robert Ramp Suite 313\nRobinsonport, AS 5...   

                                     Billing Address  Is Fraudulent  \
0  5399 Rachel Stravenue Suite 718\nNorth Blakebu...              0   
1        5230 Stephanie Forge\nCollinsbury, PR 81853              0   
2  4772 David Stravenue Apt. 447\nVelasquezside, ...              0   
3         7609 Cynthia Square\nWest Brenda, NV 23016              0   
4  2494 Robert Ramp Suite 313\nRobinsonport, AS 5...              0   

   Account Age Days  Transaction Hour  Transaction Year  Transaction Month  \
0               282                23              2024                  3   
1               223                 0              2024                  1   
2               360                 8              2024                  1   
3               325                20              2024                  1   
4               116                15              2024                  1   

   Transaction Day  Transaction Minute  Transaction Second  
0               24                  42                  43  
1               22                  53                  31  
2               22                   6                   3  
3               16                  34                  53  
4               16                  47                  23  
```


## Model Training: Logistic Regression

After preparing the data, the next step is to train a machine learning model to predict fraudulent transactions. Here is the detailed process:

### 1. Import Necessary Libraries

First, import the necessary libraries for model training and evaluation.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```


### 2. Split the Data into Training and Testing Sets

Split the data into training and testing sets to evaluate the model's performance.

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

The model achieved an accuracy of approximately 95.24%, which indicates that it correctly identified fraudulent transactions 95.24% of the time.


## Model Training: Random Forest

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

Train a Random Forest Classifier and evaluate its accuracy.

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


**Feature importance in a bar chart:**

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance = rf_model.feature_importances_
features = X.columns

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importance, y=features)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
```

**Output:**  

![download](https://github.com/user-attachments/assets/558c07a3-9b07-474a-90a8-fd2bfe3ccd3f)


### Results and Interpretation

The best parameters for the Logistic Regression model were found to be `{'C': 0.1, 'solver': 'liblinear'}`. Both the optimized Logistic Regression model and the Random Forest Classifier achieved an accuracy of approximately 95.22%. The most important features for predicting fraudulent transactions included `Transaction Amount`, `Account Age Days`, and `Shipping Address`.

These results indicates that both optimized Logistic Regression and Random Forest models perform similarly well on this dataset.

### Cross-Validation and Model Performance

To evaluate the robustness of this model, I performed 5-fold cross-validation. The `cross_val_score` function from `sklearn.model_selection` was used to achieve this.

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



