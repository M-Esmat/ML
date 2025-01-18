# Sentiment Analysis on IMDB Movie Reviews 🎬📊

Welcome to my Sentiment Analysis project! This repository contains a Python-based machine learning project that analyzes sentiment from IMDB movie reviews. The goal is to classify reviews as either **positive** or **negative** using Natural Language Processing (NLP) techniques.

---

## 📌 Table of Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Results](#results)


---

## 🚀 About the Project

This project focuses on sentiment analysis using the **IMDB Movie Reviews Dataset**. It involves:
- Preprocessing text data (cleaning, tokenization, lemmatization).
- Feature extraction using **TF-IDF Vectorization**.
- Training and evaluating two models: **Naive Bayes** and **Support Vector Machine (SVM)**.

---

## 📂 Dataset

The dataset used in this project is the **IMDB Movie Reviews Dataset**, which contains 50,000 labeled reviews (25,000 positive and 25,000 negative). You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) or use the direct link below:

[📥 Download Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🛠 Technologies Used

- **Python** 🐍
- **Pandas** 🐼
- **NLTK** 📚
- **Scikit-learn** 🔧
- **Matplotlib** 📊

---

## ⚙️ Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/m-esmat/Machine-Learning/Sentiment Analysis - 50K Movie Reviews.git

## 📊 Results

Here are the performance metrics for the trained models:

### Naive Bayes
- **Accuracy**: 85.52%
- **Classification Report**:
  ```plaintext
              precision    recall  f1-score   support

           0       0.86      0.85      0.85      4961
           1       0.85      0.86      0.86      5039

    accuracy                           0.86     10000
   macro avg       0.86      0.86      0.86     10000
  weighted avg     0.86      0.86      0.86     10000

### SVM
- **Accuracy**: 88.66%
- **Classification Report**:
  ```plaintext
              precision    recall  f1-score   support

           0       0.89      0.87      0.88      4961
           1       0.88      0.90      0.89      5039

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
  weighted avg     0.89      0.89      0.89     10000
