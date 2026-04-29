import numpy as np

class KNN:
    """
    K-Nearest Neighbors classifier implemented from scratch using NumPy.
    """
    def __init__(self, k=3):
        """
        Initialize the KNN classifier.
        
        Parameters:
        k (int): Number of nearest neighbors to use.
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        Since KNN is a lazy learner, it just stores the data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the class labels for the provided data.
        """
        # Store predictions for all samples in X
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        Predict the class label for a single sample.
        """
        # 1. Calculate Euclidean distances between x and all training samples
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        
        # 2. Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Extract the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # 4. Majority voting (find the most common label)
        most_common = np.argmax(np.bincount(k_nearest_labels))
        
        return most_common
