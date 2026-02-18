# Titanic Classification & Bias-Variance Analysis

This repository contains two distinct machine learning implementations focused on model selection and error diagnostics.

---

## 🚢 1. Titanic Model Selection
This script provides a comprehensive pipeline for predicting passenger survival on the Titanic. It compares seven different classification algorithms to find the most accurate model.

### 🛠️ Methodology
* **Feature Engineering**: Includes `Pclass`, `Sex`, `Age`, `Fare`, `Cabin`, `Prefix`, `Q`, `S`, and `Family`.
* **Preprocessing**: Uses `StandardScaler` to normalize feature scales for distance-based models.
* **Optimization**: Includes iterative loops to find the optimal $k$ for **KNN** and the best `max_depth` for **Decision Trees**.

### 🤖 Models Evaluated
1. **K-Nearest Neighbors (KNN)**: Optimized at $k=9$.
2. **Decision Tree**: Optimized at `max_depth=3`.
3. **Random Forest**: Ensemble method for improved stability.
4. **Support Vector Machine (SVM)**: Uses RBF kernel with `gamma='auto'`.
5. **Gaussian Naive Bayes**: Probabilistic classification.
6. **Logistic Regression**: Baseline linear classification.
7. **Stochastic Gradient Descent (SGD)**: Linear modeling with iterative optimization.



---

## ⚖️ 2. Bias-Variance Evaluation
This script performs a diagnostic analysis on a **Linear Regression** model using the `mlxtend` library to decompose prediction error.

### 🔍 Key Concepts
Understanding the tradeoff between Bias and Variance is critical for model tuning:
* **Bias**: Error from erroneous assumptions (leads to Underfitting).
* **Variance**: Error from sensitivity to small fluctuations in the training set (leads to Overfitting).
* **MSE**: Calculated as $MSE = \text{Bias} + \text{Variance}$.



### 📊 Results Breakdown
The script utilizes 200 bootstrap rounds to provide a stable estimate of:
* **Mean Squared Error (MSE)**
* **Main Bias**
* **Main Variance**

---

## 🧰 Requirements
To run these scripts, ensure you have the following Python libraries installed:

```bash
pip install pandas matplotlib scikit-learn mlxtend
