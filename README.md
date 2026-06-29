# Iris Flower Classification using Naive Bayes

## Project Overview

This project implements a **Gaussian Naive Bayes** classifier to predict the species of Iris flowers using the well-known **Iris dataset**. The model is trained using four flower measurements—sepal length, sepal width, petal length, and petal width—to classify flowers into one of three species: **Iris-setosa**, **Iris-versicolor**, and **Iris-virginica**.

The project demonstrates the complete machine learning workflow, including data preprocessing, model training, performance evaluation, user-based prediction, and visualization.

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

---

## Libraries

```python
pandas
matplotlib
scikit-learn
```

---

## Dataset

The project uses the **Iris.csv** dataset containing 150 flower samples with the following features:

| Column | Description |
|---------|-------------|
| Id | Unique identifier |
| SepalLengthCm | Sepal length (cm) |
| SepalWidthCm | Sepal width (cm) |
| PetalLengthCm | Petal length (cm) |
| PetalWidthCm | Petal width (cm) |
| Species | Iris flower species |

---

## Methodology

The project follows these steps:

- Load the Iris dataset from a CSV file.
- Check for missing values and remove duplicate records.
- Remove the unnecessary **Id** column.
- Encode the target labels using LabelEncoder.
- Split the dataset into training and testing sets.
- Train a Gaussian Naive Bayes classifier.
- Evaluate the model using accuracy and a confusion matrix.
- Predict the species based on user-provided flower measurements.
- Visualize the confusion matrix and the distribution of Iris species.

---

## Model Performance

The Gaussian Naive Bayes model provides high classification accuracy on the Iris dataset and efficiently classifies flowers into their respective species.

Performance evaluation includes:

- Model Accuracy
- Confusion Matrix
- Species Distribution Analysis

---

## Future Improvements

- Compare Gaussian Naive Bayes with other Naive Bayes variants.
- Apply cross-validation for more reliable model evaluation.
- Perform feature selection to analyze feature importance.
- Compare model performance with Decision Tree, KNN, Random Forest, and SVM.
- Deploy the model as a simple web application using Streamlit or Flask.

---

## Author

**Britto Domnic Aro J**

Machine Learning Enthusiast | Python Developer | AI Learner
