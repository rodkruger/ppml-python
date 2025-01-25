import os
import pandas as pd
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

# ----------------------------------------------------------------------------------------------------------------------

CONTENT_PATH = '/home/rkruger/Doutorado'


# ----------------------------------------------------------------------------------------------------------------------

class ExperimentParams:

    def __init__(self, epochs, dataset_length, training_ratio, training_length=0, testing_length=0, ring_dimension=0,
                 multiplicative_depth=0, encrypting_time=0, training_time=0, testing_time=0):
        self.epochs = epochs
        self.dataset_length = dataset_length
        self.training_ratio = training_ratio
        self.training_length = training_length
        self.testing_length = testing_length
        self.ring_dimension = ring_dimension
        self.multiplicative_depth = multiplicative_depth
        self.encrypting_time = encrypting_time
        self.training_time = training_time
        self.testing_time = testing_time

    def save_to_file(self, p_exp_name):
        params = {"epochs": self.epochs,
                  "datasetLength": self.dataset_length,
                  "trainingRatio": self.training_ratio,
                  "trainingLength": self.training_length,
                  "testingLength": self.testing_length,
                  "ringDimension": self.ring_dimension,
                  "multiplicativeDepth": self.multiplicative_depth,
                  "encryptingTime": self.encrypting_time,
                  "trainingTime": self.training_time,
                  "testingTime": self.testing_time}

        # Specify the output CSV file name
        output_file = f'{CONTENT_PATH}/{p_exp_name}/parameters.csv'

        # Write the parameters to a key-value formatted file
        with open(output_file, mode='w') as file:
            for key, value in params.items():
                file.write(f"{key} = {value}\n")


# ----------------------------------------------------------------------------------------------------------------------

def evaluate(p_exp_name, p_params, ml_model, x_train, y_train, x_test, y_test):
    start_time = time.time()

    ml_model.fit(x_train, y_train)

    end_time = time.time()
    p_params.training_time = end_time - start_time

    start_time = time.time()

    y_pred = ml_model.predict(x_test)

    end_time = time.time()
    p_params.testing_time = end_time - start_time

    # Create the experiment folder
    l_exp_folder = f'{CONTENT_PATH}/{p_exp_name}'
    if not os.path.exists(l_exp_folder):
        os.makedirs(l_exp_folder)

    # Save the predictions to the file
    df_data = {"True_Label": y_test,
               "Predicted_Label": y_pred}
    pd.DataFrame(df_data).to_csv(f'{l_exp_folder}/predictions.csv', index=False, header=False)

    # Save the params to the file
    p_params.training_length = len(x_train)
    p_params.testing_length = len(x_test)
    p_params.save_to_file(p_exp_name)


# ----------------------------------------------------------------------------------------------------------------------

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Normalize the dataset for sigmoid [-2, 2]
sigmoid_scaler = MinMaxScaler(feature_range=(-2, 2))
X_sigmoid = sigmoid_scaler.fit_transform(X)

# Normalize the dataset for tanh [0, 1]
tanh_scaler = MinMaxScaler(feature_range=(0, 1))
X_tanh = tanh_scaler.fit_transform(X)

# Split the datasets into training and testing sets
X_train_sigmoid, X_test_sigmoid, y_train_sigmoid, y_test_sigmoid = (
    train_test_split(X_sigmoid, y, test_size=0.3, stratify=y, random_state=42))
X_train_tanh, X_test_tanh, y_train_tanh, y_test_tanh = (
    train_test_split(X_tanh, y, test_size=0.3, stratify=y, random_state=42))

# Train Perceptron models with sigmoid and tanh activation functions
for epoch in range(1, 11):
    # Sigmoid activation
    g_exp_name = f'plain_sigmoid_{epoch}'
    g_params = ExperimentParams(epoch, len(X), 0.7)

    perceptron_sigmoid = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=epoch, random_state=42,
                                       solver='sgd', learning_rate_init=0.01)

    evaluate(g_exp_name, g_params, perceptron_sigmoid, X_train_sigmoid, y_train_sigmoid, X_test_sigmoid, y_test_sigmoid)

    # Tanh activation
    g_exp_name = f'plain_tanh_{epoch}'
    g_params = ExperimentParams(epoch, len(X), 0.7)

    perceptron_sigmoid = MLPClassifier(hidden_layer_sizes=(), activation='tanh', max_iter=epoch, random_state=42,
                                       solver='sgd', learning_rate_init=0.01)

    evaluate(g_exp_name, g_params, perceptron_sigmoid, X_train_sigmoid, y_train_sigmoid, X_test_sigmoid, y_test_sigmoid)
