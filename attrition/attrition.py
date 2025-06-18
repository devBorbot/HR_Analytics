# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
df = pd.read_csv('Data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Encode target variable
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

# Select features and target
X = df.drop(['Attrition', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
y = df['Attrition']

# Convert categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split data into train, calibration, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ------------------
# EDA Section
# ------------------
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Target variable distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title('Attrition Distribution')
plt.show()

# Numerical features distribution
num_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome']
df[num_cols].hist(figsize=(10,8), bins=20)
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(12,8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Categorical features analysis
cat_cols = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
plt.figure(figsize=(12,8))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(2,2,i)
    sns.countplot(x=col, hue='Attrition', data=df)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------
# Model Training
# ------------------
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()