import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Perceptron:

    def __init__(self, learning_rate=0.001, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        """ Step activation function. """
        return np.tanh(x)

    def fit(self, X, y):
        """ Train the perceptron on the dataset. """
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(0, 1, size=n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                features = x_i
                y_pred = self.predict(features)
                delta = y[idx] - y_pred
                error = self.learning_rate * delta
                self.weights += error * features
                self.bias += error

    def predict(self, X):
        """ Make predictions on new data. """
        z = np.dot(X, self.weights)
        z = np.sum(z)
        z += self.bias
        return self.activation(z)

#------------------------------------------------------------------------------

data = load_breast_cancer()
X = data.data
y = data.target

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
y = np.where(y == 0, -1, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

perceptron = Perceptron(learning_rate=0.001, epochs=10)
perceptron.fit(X_train, y_train)

y_preds = []

for i in X_test:
    y_pred = perceptron.predict(i)

    if y_pred >= 0:
        y_preds.append(1)
    else:
        y_preds.append(-1)

"""
print(f"Weights: {perceptron.weights}")
print(f"Bias: {perceptron.bias}")
print(f"Features: {X[0]}")
print(f"Label: {y[0]}")    
print(f"Prediction: {perceptron.predict(X[0])}")
"""

# Calculate Metrics
accuracy = accuracy_score(y_test, y_preds)
precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
