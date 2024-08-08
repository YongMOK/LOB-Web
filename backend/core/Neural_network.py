import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

class SLFN:
    def __init__(self, K, sigma=1.0, lambda_=0.01):
        self.K = K
        self.sigma = sigma
        self.lambda_ = lambda_
        self.V = None
        self.W = None

    def _initialize_hidden_layer_parameters(self, X):
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(X)
        self.V = kmeans.cluster_centers_

    def _rbf(self, x, v):
        return np.exp(-np.linalg.norm(x - v)**2 / (2 * self.sigma**2))

    def _compute_hidden_layer_outputs(self, X):
        N = X.shape[0]
        H = np.zeros((N, self.K))
        for i, x in enumerate(X):
            for k, v in enumerate(self.V):
                H[i, k] = self._rbf(x, v)
        return H

    def _compute_output_layer_parameters(self, H, Y):
        I = np.eye(H.shape[1])
        Y_one_hot = np.zeros((Y.size, Y.max() + 1))
        Y_one_hot[np.arange(Y.size), Y.flatten()] = 1
        self.W = np.linalg.inv(H.T @ H + self.lambda_ * I) @ (H.T @ Y_one_hot)

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y,dtype=int)
        self._initialize_hidden_layer_parameters(X)
        H = self._compute_hidden_layer_outputs(X)
        self._compute_output_layer_parameters(H, Y)
        print("Training completed...")

    def _classify_new_data(self, x):
        h = np.array([self._rbf(x, v) for v in self.V])
        o = self.W.T @ h
        return np.argmax(o)

    def predict(self, X_new):
        X_new = np.array(X_new)
        predictions = np.array([self._classify_new_data(x) for x in X_new])
        return predictions
    


