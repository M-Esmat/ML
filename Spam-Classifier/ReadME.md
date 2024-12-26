# Spam Email Classifier

This project implements a Spam Email Classifier using two machine learning models:
- Logistic Regression
- Random Forest Classifier

The dataset used for this project contains SMS messages labeled as "ham" (not spam) or "spam". The models are trained to classify incoming emails/messages as either spam or not.

## Project Steps

1. **Data Preprocessing**:
   - Load the dataset and clean it.
   - Convert text labels to numeric (ham = 1, spam = 0).
   - Split the dataset into training and test sets.

2. **Feature Extraction**:
   - Use `TfidfVectorizer` to extract text features.
   - Transform the text messages into numerical feature vectors.

3. **Model Training**:
   - Train a Logistic Regression model and a Random Forest Classifier.
   - Evaluate both models on the test dataset.

4. **Model Evaluation**:
   - Report metrics such as accuracy, confusion matrix, and classification report for both models.

5. **Custom Input Testing**:
   - Allow the user to input a custom email/message for classification.

## Dataset

The dataset used in this project is the SMS Spam Collection dataset. It can be downloaded from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

## Performance

### Logistic Regression:
- **Accuracy**: 95.25%
- **Confusion Matrix**:
