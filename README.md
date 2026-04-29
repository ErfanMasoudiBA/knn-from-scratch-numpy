# K-Nearest Neighbors (KNN) from Scratch using NumPy

This project demonstrates the implementation of the K-Nearest Neighbors (KNN) classification algorithm entirely from scratch, using only **NumPy** for matrix operations and mathematical calculations. The model is built without relying on high-level machine learning frameworks like scikit-learn for the core algorithm, showcasing a deep understanding of the underlying mechanics.

## 📌 Project Overview

The goal of this project is to classify handwritten digits using a custom-built KNN model. The pipeline includes data loading, exploratory data analysis (EDA), distance calculation, hyperparameter tuning, and performance evaluation.

## 🛠️ Tech Stack

- **Python 3**
- **NumPy**: Core algorithm implementation (vectorized Euclidean distance, sorting, voting).
- **Matplotlib & Seaborn**: Data visualization, plotting optimal $K$, and confusion matrix heatmap.
- **Scikit-Learn**: Solely for loading the dataset (`load_digits`) and generating the confusion matrix for evaluation.

## 📊 Dataset

The project uses the `digits` dataset from scikit-learn (an offline alternative to MNIST), which consists of 1,797 samples of $8 \times 8$ pixel handwritten digits (0-9).

## 🚀 Features

- **Custom KNN Class**: A fully functional `fit` and `predict` pipeline.
- **Vectorized Operations**: Utilized NumPy's broadcasting to optimize distance calculations without using slow nested Python `for` loops.
- **Hyperparameter Tuning**: Evaluated multiple values of $K$ to find the optimal number of neighbors, plotting the accuracy curve.
- **Error Analysis**: Generated a confusion matrix to evaluate precision/recall and visualized specific misclassified images to understand the model's limitations.

## 📈 Results

- The model achieved high accuracy on the test set.
- The optimal value for $K$ was found to be $K=1$ (or $K=3$), yielding an accuracy of **~99%**.
- Misclassified images were analyzed and found to be highly ambiguous digits.

## 📂 Project Structure

- `notebooks/`: Contains the step-by-step Jupyter Notebook with EDA, training, and evaluation.
- `src/`: Contains the pure Python implementation of the `KNN` class.

## ⚙️ How to Run

1. Clone the repository:
   `git clone https://github.com/your-username/knn-from-scratch.git`
2. Install the dependencies:
   `pip install -r requirements.txt`
3. Run the Jupyter Notebook in the `notebooks/` directory.
