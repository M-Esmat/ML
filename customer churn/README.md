# Customer Churn Prediction

## Overview
This project aims to predict customer churn in a telecommunications company by applying machine learning techniques. Using historical data, the goal is to classify customers into two categories: those likely to churn (leave the service) and those who will stay. By predicting churn, businesses can take proactive measures to retain valuable customers, ultimately improving customer retention strategies and overall business performance.

---

## Project Details

### Steps Involved

#### 1. **Data Importing & Exploration**
- The dataset is loaded from a CSV file and explored to understand the distribution of data.
- Categorical and numerical features are identified, with unique values for categorical columns examined.
- Data link: [https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn]

#### 2. **Data Cleaning & Preprocessing**
- Handling missing values by filling in the mean for numerical columns.
- One-hot encoding applied to categorical variables to convert them into a format suitable for machine learning algorithms.
- Feature scaling applied to numerical features to standardize values and improve model performance.

#### 3. **Model Training**
- Multiple machine learning models are trained and evaluated: Logistic Regression, Random Forest, Support Vector Machine (SVM), and XGBoost.
- The best-performing models are further optimized using GridSearchCV for hyperparameter tuning.

#### 4. **Model Evaluation**
- Models are evaluated based on accuracy, precision, recall, F1-score, and confusion matrix.
- A comprehensive classification report is generated for each model to assess its effectiveness.

#### 5. **Results & Conclusion**
- A final comparison of all models' performance allows the selection of the best model for predicting customer churn.
- The outcome helps businesses understand factors contributing to churn and take action to reduce customer attrition.

---

## Technologies & Libraries Used
- **Python**: Programming language used for data analysis and model building.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning models and preprocessing.
- **XGBoost**: For advanced gradient boosting models.
- **GridSearchCV**: For hyperparameter tuning and model optimization.

---

## How to Run the Code

1. Download the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. Ensure the dataset is in the correct directory (or update the file path in the code).
3. Run the provided Python script to load, clean, process the data, and build predictive models.
4. Review the output for accuracy, classification reports, and model performance comparison.

---

## Results

### Model Performance:

- **Logistic Regression**: 
    - Accuracy = 82.1%, Precision = 0.69, Recall = 0.60 for the churned class.
  
- **Random Forest**: 
    - Accuracy = 81.1%, Precision = 0.69, Recall = 0.52 for the churned class.
  
- **Support Vector Machine (SVM)**: 
    - Accuracy = 81.4%, Precision = 0.70, Recall = 0.52 for the churned class.
  
- **XGBoost**: 
    - Accuracy = 79.8%, Precision = 0.64, Recall = 0.54 for the churned class.
  
- **Random Forest (Tuned with GridSearchCV)**: 
    - Improved accuracy of 81.1%.

---

## Conclusion
The best-performing models were evaluated based on various metrics such as accuracy, precision, recall, and F1-score. This project highlights the importance of data preprocessing, model selection, and tuning for improving prediction accuracy. By leveraging machine learning techniques, businesses can use churn predictions to make more informed decisions about customer retention strategies.
