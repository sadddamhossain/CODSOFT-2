#!/usr/bin/env python
# coding: utf-8

# # Import library for read the dataset

# In[5]:


import pandas as pd

df1 = pd.read_csv("fraudTest.csv")

df2 = pd.read_csv("fraudTrain.csv")


# In[3]:


df1


# In[4]:


df2


# # Data analysis 

# In[6]:


df1.info()


# In[7]:


df2.info()


# In[9]:


print(df1.shape)


# In[10]:


print(df2.shape)


# In[13]:


print(df1.describe)


# In[14]:


print(df2.describe)


# In[80]:


print(df1.columns)
print(df1.dtypes)


# In[81]:


print(df2.columns)
print(df2.dtypes)


# In[19]:


selected_columns = df1[['job', 'city', 'gender', 'state', 'amt', 'is_fraud']]

print(selected_columns)


# In[21]:


selected_columns = df2[['job', 'gender', 'state', 'amt', 'is_fraud']]

print(selected_columns)


# In[23]:


selected_columns = df1['is_fraud']

print(selected_columns)


# In[24]:


# Filter the DataFrame to include only the rows where 'is_fraud' is True
fraud_data = df1[df1['is_fraud'] == 1]

# Display the fraud data
print(fraud_data)


# In[25]:


fraud_data = df1[df1['is_fraud'] > 0]

# Display the fraud data
print(fraud_data)


# In[26]:


fraud_data = df2[df2['is_fraud'] > 0]

# Display the fraud data
print(fraud_data)


# In[33]:


fraud_data = df1[df1['is_fraud'] > 0]['is_fraud']

# Display the fraud data
print(fraud_data)


# In[34]:


fraud_data = df2[df2['is_fraud'] > 0]['is_fraud']

# Display the fraud data
print(fraud_data)


# In[103]:


# Step 1: Extract 'is_fraud' column from each DataFrame separately
fraud_data_df1 = df1[df1['is_fraud'] > 0]
fraud_data_df2 = df2[df2['is_fraud'] > 0]

# Step 2: Concatenate the 'is_fraud' columns into one column
all_fraud_data = pd.concat([fraud_data_df1, fraud_data_df2], ignore_index=True)

# Step 3: Merge the data
df3 = pd.DataFrame()  # Create an empty DataFrame df3
df3['all_fraud_data'] = all_fraud_data['is_fraud']  # Add 'is_fraud' column

# Add additional columns from df1
for column in ['job', 'gender', 'state', 'amt','cc_num']:
    df3[column] = all_fraud_data[column]

# Display the merged DataFrame with all fraud data and selected columns
print(df3)
print(df3.dtypes)


# In[39]:


df1.isnull()


# In[41]:


df1.isnull().sum()


# In[42]:


df2.isnull()


# In[43]:


df2.isnull().sum()


# # Data Visualization

# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[65]:


# Step 1: Extract 'is_fraud' column from DataFrame
fraud_data = df1['is_fraud']

# Step 2: Plot the 'is_fraud' data
fraud_counts = fraud_data.value_counts()
plt.bar(fraud_counts.index, fraud_counts.values)
plt.xlabel('is_fraud')
plt.ylabel('Count')
plt.title('df1 Fraud Data Counts')
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.show()


# In[64]:


# Step 1: Extract 'is_fraud' column from DataFrame
fraud_data = df1[df1['is_fraud'] > 0]['is_fraud']

# Step 2: Plot the 'is_fraud' data
fraud_counts = fraud_data.value_counts()
plt.bar(fraud_counts.index, fraud_counts.values)
plt.xlabel('is_fraud')
plt.ylabel('Count')
plt.title('df1 Fraud Data Counts')
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.show()


# In[63]:


# Step 1: Count the occurrences of fraud and non-fraud transactions
fraud_counts = df1['is_fraud'].value_counts()

# Step 2: Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(fraud_counts, labels=['Not Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
plt.title('df1 Fraud vs. Non-Fraud Transactions')
plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.
plt.show()


# In[62]:


# Step 1: Extract 'is_fraud' column from DataFrame
fraud_data = df2['is_fraud']

# Step 2: Plot the 'is_fraud' data
fraud_counts = fraud_data.value_counts()
plt.bar(fraud_counts.index, fraud_counts.values)
plt.xlabel('is_fraud')
plt.ylabel('Count')
plt.title('df2 Fraud Data Counts')
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.show()


# In[61]:


# Step 1: Extract 'is_fraud' column from DataFrame
fraud_data = df2[df2['is_fraud'] > 0]['is_fraud']

# Step 2: Plot the 'is_fraud' data
fraud_counts = fraud_data.value_counts()
plt.bar(fraud_counts.index, fraud_counts.values)
plt.xlabel('is_fraud')
plt.ylabel('Count')
plt.title('df2 Fraud Data Counts')
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.show()


# In[60]:


# Step 1: Count the occurrences of fraud and non-fraud transactions
fraud_counts = df2['is_fraud'].value_counts()

# Step 2: Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(fraud_counts, labels=['Not Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['pink', 'white'])
plt.title('df2 Fraud vs. Non-Fraud Transactions')
plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.
plt.show()


# In[67]:


# Step 1: Plot a bar plot for the 'job' column
plt.figure(figsize=(10, 6))
df3['job'].value_counts().plot(kind='bar')
plt.xlabel('Job')
plt.ylabel('Count')
plt.title('Frequency of Jobs in Fraud Data')
plt.xticks(rotation=45)
plt.show()


# In[69]:


# Step 2: Plot a scatter plot for the 'amt' column against 'all_fraud_data'
plt.figure(figsize=(10, 6))
plt.scatter(df3['amt'], df3['all_fraud_data'])
plt.xlabel('Amount')
plt.ylabel('is_fraud')
plt.title('Scatter Plot of Amount vs. Fraud')
plt.show()


# # Machine Learning Models :-

# In[78]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# # logistic regression (df1) :-1

# In[126]:


# Step 1: Use label encoding for categorical variables
X = df1.drop(columns=['is_fraud', 'amt', 'cc_num'])
y = df1['is_fraud']

# You can use LabelEncoder from sklearn.preprocessing to encode categorical columns
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in X.select_dtypes(include='object'):
    X[col] = label_encoder.fit_transform(X[col])

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the Logistic Regression model
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Step 4: Calculate evaluation metrics for Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

print("\nLogistic Regression Metrics:df1")
print("\nAccuracy:", accuracy_lr)
print("\nPrecision:", precision_lr)
print("\nRecall:", recall_lr)
print("\nF1-score:", f1_lr)
print("\nConfusion Matrix:")
print(conf_matrix_lr)

# Step 5: Plot the confusion matrix heatmap for Logistic Regression
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_lr, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Logistic Regression (df1)")
plt.show()


# #  logistic regression (df2) :-1
# 

# In[125]:


# Step 1: Use label encoding for categorical variables
X = df2.drop(columns=['is_fraud', 'amt', 'cc_num'])
y = df2['is_fraud']

# You can use LabelEncoder from sklearn.preprocessing to encode categorical columns
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in X.select_dtypes(include='object'):
    X[col] = label_encoder.fit_transform(X[col])

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the Logistic Regression model
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Step 4: Calculate evaluation metrics for Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

print("\nLogistic Regression Metrics:df2")
print("\nAccuracy:", accuracy_lr)
print("\nPrecision:", precision_lr)
print("\nRecall:", recall_lr)
print("\nF1-score:", f1_lr)
print("\nConfusion Matrix:")
print(conf_matrix_lr)

# Step 5: Plot the confusion matrix heatmap for Logistic Regression
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_lr, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Logistic Regression (df2)")
plt.show()


# # Decision Tree model (df1) :-2
# 

# In[124]:


# Step 1: Use label encoding for categorical variables
X = df1.drop(columns=['is_fraud', 'amt', 'cc_num'])
y = df1['is_fraud']

label_encoder = LabelEncoder()
for col in X.select_dtypes(include='object'):
    X[col] = label_encoder.fit_transform(X[col])

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the Decision Tree model
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# Calculate evaluation metrics for Decision Trees
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

print("\nDecision Trees Metrics:df1")
print("\nAccuracy:", accuracy_dt)
print("\nPrecision:", precision_dt)
print("\nRecall:", recall_dt)
print("\nF1-score:", f1_dt)
print("\nConfusion Matrix:")
print(conf_matrix_dt)

# Plot the confusion matrix heatmap for Decision Trees
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_dt, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Decision Trees(df1)")
plt.show()


# # Decision Tree model (df2):-2

# In[123]:


# Step 1: Use label encoding for categorical variables
X = df2.drop(columns=['is_fraud', 'amt', 'cc_num'])
y = df2['is_fraud']

label_encoder = LabelEncoder()
for col in X.select_dtypes(include='object'):
    X[col] = label_encoder.fit_transform(X[col])

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the Decision Tree model
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# Calculate evaluation metrics for Decision Trees
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

print("\nDecision Trees Metrics:df2")
print("\nAccuracy:", accuracy_dt)
print("\nPrecision:", precision_dt)
print("\nRecall:", recall_dt)
print("\nF1-score:", f1_dt)
print("\nConfusion Matrix:")
print(conf_matrix_dt)

# Plot the confusion matrix heatmap for Decision Trees
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_dt, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Decision Trees(df2)")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




