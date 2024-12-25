# News Headline Sentiment Classifier

This project implements a machine learning model to classify news headlines into positive or negative sentiment using a Random Forest Classifier. The project is built using Python and scikit-learn.

---

## 📁 Dataset
The dataset used in this project contains news headlines collected over time, with each headline labeled as **positive** (1) or **negative** (0).  
[Download the dataset here](https://www.kaggle.com/datasets/siddharthtyagi/news-headlines-dataset-for-stock-sentiment-analyze)

---

## ⚙️ Workflow
1. Preprocess news headlines:
   - Remove non-alphabetic characters.
   - Convert all text to lowercase.
   - Combine all headlines in a row into a single paragraph.
2. Vectorize text data using bigrams (`CountVectorizer` with n-gram range of (2, 2)).
3. Train a Random Forest Classifier with 200 estimators and entropy as the criterion.
4. Evaluate the model on the test data.

---

## 📊 Model Performance
| Metric                 | Score |
|-------------------------|-------|
| **Accuracy**            | 87%   |
| **Precision (Positive)**| 80%   |
| **Recall (Positive)**   | 97%   |
| **F1-Score (Positive)** | 88%   |

### Confusion Matrix:
|          | Predicted Negative | Predicted Positive |
|----------|--------------------|--------------------|
| **Actual Negative** | 140                  | 46                   |
| **Actual Positive** | 5                    | 187                  |

---

## 🛠️ Technologies Used
- Python
- Pandas
- scikit-learn
- Random Forest Classifier
- CountVectorizer

