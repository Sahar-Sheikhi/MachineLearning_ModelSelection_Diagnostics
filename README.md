# Machine Learning Course

A comprehensive collection of machine learning examples and projects demonstrating various algorithms and techniques for classification, regression, and model evaluation.

## 📚 Overview

This repository contains practical implementations of machine learning concepts, including:

- **Model Selection and Comparison**: Comparative analysis of multiple classification algorithms
- **Bias-Variance Tradeoff**: Understanding model performance and generalization
- **Hyperparameter Tuning**: Optimizing model parameters for better accuracy

## 📂 Contents

### 1. Titanic Model Selection (`TAITANIC model selection.py`)

A comprehensive analysis of the Titanic dataset using multiple machine learning algorithms for survival prediction. This project demonstrates:

- **Data Preprocessing**: StandardScaler normalization and feature engineering
- **Model Comparison**: Implementation and evaluation of 7 different algorithms:
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Logistic Regression
  - Stochastic Gradient Descent (SGD)

- **Hyperparameter Tuning**:
  - Finding optimal k value for KNN (k=9)
  - Determining best tree depth for Decision Tree (depth=3)
  
- **Performance Visualization**: Comparative accuracy plots for all models

**Best Results:**
- Random Forest: 77.27% accuracy
- SVM: 77.51% accuracy
- SGD: 76.79% accuracy

### 2. Bias and Variance Evaluation (`bias and variance evaluation.py`)

A practical implementation demonstrating how to evaluate bias and variance in regression models using the housing dataset.

- **Technique**: Bias-Variance Decomposition
- **Model**: Linear Regression
- **Library**: mlxtend.evaluate
- **Metrics**: Mean Squared Error (MSE), Bias, and Variance
- **Method**: 200-round bootstrap evaluation

This example helps understand the tradeoff between model complexity and generalization performance.

## 🛠️ Requirements

```python
pandas
numpy
scikit-learn
matplotlib
mlxtend
```

## 📦 Installation

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib mlxtend
```

## 🚀 Usage

### Running Titanic Model Selection

```python
# Ensure you have the Titanic dataset loaded
# The script assumes 'train' and 'test' DataFrames are available
python "TAITANIC model selection.py"
```

### Running Bias-Variance Evaluation

```python
# This script downloads the dataset automatically
python "bias and variance evaluation.py"
```

## 📊 Key Concepts Covered

1. **Model Selection**: Comparing multiple algorithms to find the best performing model
2. **Cross-Validation**: Train-test split methodology (80-20 split)
3. **Feature Scaling**: Using StandardScaler for normalization
4. **Hyperparameter Optimization**: Grid search approach for finding optimal parameters
5. **Bias-Variance Tradeoff**: Understanding model complexity and overfitting/underfitting
6. **Model Evaluation**: Using accuracy scores and visualization for performance assessment

## 📈 Learning Outcomes

By exploring this repository, you will learn:

- How to implement and compare multiple ML algorithms
- Best practices for data preprocessing and feature engineering
- Techniques for hyperparameter tuning
- How to evaluate and visualize model performance
- Understanding bias-variance tradeoff in practical scenarios

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements or additional machine learning examples.

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Sahar Sheikhi**

---

*This repository is created for educational purposes to demonstrate practical machine learning implementations.*
