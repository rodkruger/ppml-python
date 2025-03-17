import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        # self.weights = np.random.uniform(-0.1, 0.1, size=n_features)
        self.weights = [0.03373467, -0.00025997, -0.03883165, 0.06173363, -0.00649833, 0.04869294, -0.02772681,
                        -0.00845026, -0.08583626, -0.05578794, 0.00677842, -0.06297108, 0.016218, -0.0001006,
                        -0.02879055, -0.03708422, 0.04747317, 0.02290419, 0.02182636, -0.09092148, 0.07830223,
                        -0.01524018, 0.07498792, 0.06051367, -0.05838291, -0.01896154, 0.09503015, 0.03494426,
                        -0.04586892, 0.08849982]

        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                np.set_printoptions(suppress=True, linewidth=1000, threshold=np.inf)

                features = x_i
                activation = self.predict(features)
                error = y[idx] - activation
                delta = self.learning_rate * error
                new_weights = features * delta
                self.weights += new_weights
                self.bias += delta

                """
                print(f"label: {y[idx]}")
                print(f"activation: {activation}")
                print(f"error: {error}")
                print(f"delta: {delta}")
                print(f"new_weights: {new_weights}")
                print(f"self.weights: {self.weights}")
                print(f"self.bias: {self.bias}")
                print(f"Press any key to continue:")
                s = input()
                """

    def predict(self, X):
        """ Make predictions on new data. """
        linear_dot = np.dot(X, self.weights)
        sum_linear_dot = np.sum(linear_dot)
        sum_linear_dot_bias = self.bias + sum_linear_dot
        activation = self.activation(sum_linear_dot_bias)

        print(f"features: {X}")
        print(f"self.weights: {self.weights}")
        print(f'linear dot: {linear_dot}')
        print(f'sum: {sum_linear_dot}')
        print(f'sum + bias: {sum_linear_dot_bias}')
        print(f'activation: {activation}')

        return activation


# ------------------------------------------------------------------------------

CONTENT_PATH = '/home/rkruger/Doutorado'

X_train_tanh = pd.read_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/training_features_range11.csv',
                           header=None).to_numpy()
y_train_tanh = pd.read_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/training_labels_range11.csv',
                           header=None).to_numpy().ravel()
X_test_tanh = pd.read_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/testing_features_range11.csv', header=None).to_numpy()
y_test_tanh = pd.read_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/testing_labels_range11.csv',
                          header=None).to_numpy().ravel()

y_train_tanh[y_train_tanh == 0] = -1
y_test_tanh[y_test_tanh == 0] = -1

perceptron = Perceptron(learning_rate=0.01, epochs=1)

print("---------- Training ----------")
perceptron.fit(X_train_tanh, y_train_tanh)

print("---------- Testing ----------")
y_preds = []

for i in X_test_tanh:
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
accuracy = accuracy_score(y_test_tanh, y_preds)
precision = precision_score(y_test_tanh, y_preds)
recall = recall_score(y_test_tanh, y_preds)
f1 = f1_score(y_test_tanh, y_preds)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
