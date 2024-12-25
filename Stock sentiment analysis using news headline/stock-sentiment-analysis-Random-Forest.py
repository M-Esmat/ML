# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('data.csv', encoding='ISO-8859-1')

# Split the dataset
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

# Preprocess the training data
x = train.iloc[:, 2:27].replace("[^a-zA-Z]", " ", regex=True)
x.columns = [str(i) for i in range(25)]
for col in x.columns:
    x[col] = x[col].str.lower()

# Combine headlines into a single paragraph
headlines = []
for row in range(0, len(x.index)):
    headlines.append(' '.join(str(i) for i in x.iloc[row, :]))

# Vectorize the data using CountVectorizer
countvec = CountVectorizer(ngram_range=(2, 2))
train_data = countvec.fit_transform(headlines)

# Train a Random Forest Classifier
RFC = RandomForestClassifier(n_estimators=200, criterion='entropy')
RFC.fit(train_data, train['Label'])

# Preprocess the test data
test_headlines = [' '.join(row) for row in test.iloc[:, 2:27].values]
test_data = countvec.transform(test_headlines)

# Make predictions
predictions = RFC.predict(test_data)

# Evaluate the model
conf_matrix = confusion_matrix(test['Label'], predictions)
accuracy = accuracy_score(test['Label'], predictions)
report = classification_report(test['Label'], predictions)

# Print the results
print("Confusion Matrix:\n", conf_matrix)
print("\nAccuracy Score:", accuracy)
print("\nClassification Report:\n", report)
