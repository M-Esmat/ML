import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Constants
DATA_PATH = r'imdb-50K-movie-reviews\IMDB Dataset.csv'

def load_data(path):
    """Load the dataset from the given path."""
    return pd.read_csv(path)

def preprocess_text(text, lemmatizer, stopwords):
    """Preprocess text by cleaning, tokenizing, and lemmatizing."""
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = text.split()  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using accuracy and classification report."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return accuracy, report

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for a given text using the trained model."""
    text_vect = vectorizer.transform([text])
    return model.predict(text_vect)[0]

def main():
    # Load and preprocess data
    movies = load_data(DATA_PATH)
    movies['sentiment'] = movies['sentiment'].map({'positive': 1, 'negative': 0})

    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))

    movies['clean_review'] = movies['review'].apply(
        lambda x: preprocess_text(x, lemmatizer, stopwords_set)
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        movies['clean_review'], movies['sentiment'], test_size=0.2, random_state=42
    )

    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train models
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_vec, y_train)

    svm = SVC(kernel='linear')
    svm.fit(X_train_vec, y_train)

    # Evaluate models
    print("Naive Bayes Performance:")
    y_pred_nb = naive_bayes.predict(X_test_vec)
    acc_nb, report_nb = evaluate_model(y_test, y_pred_nb)
    print(f"Accuracy: {acc_nb}\nClassification Report:\n{report_nb}")

    print("SVM Performance:")
    y_pred_svm = svm.predict(X_test_vec)
    acc_svm, report_svm = evaluate_model(y_test, y_pred_svm)
    print(f"Accuracy: {acc_svm}\nClassification Report:\n{report_svm}")

if __name__ == "__main__":
    main()
