# Fake News Detection

## Overview
This project focuses on detecting fake news using machine learning techniques. The goal is to classify news articles as either "Fake" or "Not Fake" based on their content. The project uses a dataset containing both fake and real news articles, and several machine learning models are trained and evaluated to achieve the best performance.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Manual Testing](#manual-testing)
- [Conclusion](#conclusion)

## Introduction
Fake news has become a significant issue in today's digital age, where misinformation can spread rapidly. This project aims to build a machine learning model that can automatically detect fake news articles. The project uses a dataset of labeled news articles and applies various machine learning algorithms to classify them.

## Dataset
The dataset used in this project consists of two main files:
- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains real news articles.

Each dataset includes the following columns:
- `title`: The title of the news article.
- `text`: The content of the news article.
- `subject`: The subject or category of the news article.
- `date`: The date the article was published.

## Libraries Used
The following Python libraries are used in this project:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `seaborn` and `matplotlib`: For data visualization.
- `scikit-learn`: For machine learning model training and evaluation.
- `re`: For regular expressions and text preprocessing.
- `string`: For string manipulation.

## Data Preprocessing
The data preprocessing steps include:
1. **Loading the Dataset**: The fake and real news datasets are loaded into pandas DataFrames.
2. **Adding a Target Column**: A new column `class` is added to label the articles as fake (`0`) or real (`1`).
3. **Merging the Datasets**: The fake and real news datasets are merged into a single DataFrame.
4. **Shuffling the Data**: The merged dataset is shuffled to ensure randomness.
5. **Text Preprocessing**: The text data is cleaned by removing special characters, URLs, and other noise using regular expressions.

## Model Training
Several machine learning models are trained on the preprocessed data:
- **Logistic Regression (LR)**
- **Decision Tree Classifier (DT)**
- **Gradient Boosting Classifier (GBC)**
- **Random Forest Classifier (RFC)**

The text data is converted into numerical vectors using the `TfidfVectorizer` from scikit-learn, which transforms the text into a matrix of TF-IDF features.

## Model Evaluation
The performance of each model is evaluated using accuracy and a classification report, which includes precision, recall, and F1-score.

### Results:
- **Logistic Regression**: Accuracy of 98.65%
- **Decision Tree Classifier**: Accuracy of 99.54%
- **Gradient Boosting Classifier**: Accuracy of 99.37%
- **Random Forest Classifier**: Accuracy of 98.87%

## Manual Testing
A manual testing function is provided to allow users to input a news article and get predictions from all four models. The function preprocesses the input text and outputs the predictions from each model.

### Example Usage:
```python
news = "https://www.cnbc.com/2024/03/28/ukraine-war-live-updates-latest-news-on-russia-and-the-war-in-ukraine.html"
manual_testing(news)
```

### Output:
```
LR Prediction: Fake News
DT Prediction: Fake News
GBC Prediction: Fake News
RFC Prediction: Fake News
```

## Conclusion
This project demonstrates the effectiveness of machine learning models in detecting fake news. The Decision Tree Classifier achieved the highest accuracy, but all models performed well. Future work could include exploring more advanced models, such as deep learning approaches, and expanding the dataset to include more diverse news sources.

