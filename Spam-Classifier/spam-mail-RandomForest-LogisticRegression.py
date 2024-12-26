# Import libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Preprocess dataset
df.dropna(how='any', axis=1, inplace=True)
df.columns = ['label', 'message']
df['label'] = df['label'].replace({'ham': 1, 'spam': 0})

# Split data into features and labels
X = df['message']
Y = df['label']

# Split data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=42, test_size=0.2)

# Feature extraction
features = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
xtrain_features = features.fit_transform(xtrain)
xtest_features = features.transform(xtest)

# Logistic Regression Model
LR = LogisticRegression()
LR.fit(xtrain_features, ytrain)
train_pred = LR.predict(xtrain_features)
test_pred = LR.predict(xtest_features)

print("Logistic Regression Performance:")
print("Training Accuracy:", accuracy_score(ytrain, train_pred))
print("Test Accuracy:", accuracy_score(ytest, test_pred))
print("Confusion Matrix:\n", confusion_matrix(ytest, test_pred))
print("Classification Report:\n", classification_report(ytest, test_pred))

# Random Forest Classifier
RFC = RandomForestClassifier(n_estimators=200, criterion='entropy')
RFC.fit(xtrain_features, ytrain)
rf_pred = RFC.predict(xtest_features)

print("Random Forest Classifier Performance:")
print("Test Accuracy:", accuracy_score(ytest, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(ytest, rf_pred))
print("Classification Report:\n", classification_report(ytest, rf_pred))

# Test with custom input
input_mail = [str(input("Enter an email to classify: "))]
input_data_features = features.transform(input_mail)
prediction = RFC.predict(input_data_features)

if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")
