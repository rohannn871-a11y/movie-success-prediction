# 🎬 Movie Success Prediction

## 📌 Project Overview

This project predicts the success category of a movie as **Hit, Average, or Flop** using Machine Learning.

The project uses movie-related features such as movie duration, genre, language, country, budget, gross revenue, Facebook likes, critic reviews, user reviews, and other available movie information.

## 🎯 Objective

The main objective of this project is to:

- Predict movie success categories.
- Analyze factors related to movie success.
- Compare different Machine Learning classification models.
- Build a simple web application for movie success prediction.

## 📊 Dataset

The dataset contains information about movies and their IMDb ratings.

The target variable `Classify` is created using IMDb score ranges:

- **1–3 → Flop**
- **3–6 → Average**
- **6–10 → Hit**

## 🤖 Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

## 🌐 Deployment

A Streamlit web application was developed to allow users to enter movie information and receive a predicted success category.

### Prediction Categories

- 🟢 **Hit**
- 🟡 **Average**
- 🔴 **Flop**

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Matplotlib
- Seaborn

## 👨‍💻 Project

**Movie Success Prediction using Machine Learning**
