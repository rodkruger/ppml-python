import os
import pandas as pd
import time

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

def read_dataset(p_dataset_name, p_range):
    l_train_dataset_path = f"{CONTENT_PATH}/Datasets/{p_dataset_name}/training"
    l_test_dataset_path = f"{CONTENT_PATH}/Datasets/{p_dataset_name}/testing"
    l_range = f'range{abs(p_range[0])}{abs(p_range[1])}'

    x_train = pd.read_csv(f'{l_train_dataset_path}_features_{l_range}.csv', header=None).to_numpy()
    y_train = pd.read_csv(f'{l_train_dataset_path}_labels_{l_range}.csv', header=None).to_numpy().ravel()
    x_test = pd.read_csv(f'{l_test_dataset_path}_features_{l_range}.csv', header=None).to_numpy()
    y_test = pd.read_csv(f'{l_test_dataset_path}_labels_{l_range}.csv', header=None).to_numpy().ravel()

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------------------------------------------------

def evaluate_models(p_dataset_name, p_range):
    x_train, y_train, x_test, y_test = read_dataset(p_dataset_name, p_range)

    # Build a Perceptron with sigmoid and tanh activation functions
    for epoch in range(1, 11):
        # Sigmoid activation --------------------------------------------------
        l_exp_name = f'Predictions/{p_dataset_name}/plain_sigmoid_{epoch}'
        l_params = ExperimentParams(epoch, len(x_train) + len(x_test), 0.7)

        perceptron_sigmoid = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=epoch,
                                           random_state=42, solver='sgd', learning_rate_init=0.005)

        evaluate(l_exp_name, l_params, perceptron_sigmoid, x_train, y_train, x_test, y_test)

        # Tanh activation -----------------------------------------------------
        l_exp_name = f'Predictions/{p_dataset_name}/plain_tanh_{epoch}'
        l_params = ExperimentParams(epoch, len(x_train) + len(x_test), 0.7)

        perceptron_tanh = MLPClassifier(hidden_layer_sizes=(), activation='tanh', max_iter=epoch,
                                        random_state=42, solver='sgd', learning_rate_init=0.005)
        evaluate(l_exp_name, l_params, perceptron_tanh, x_train, y_train, x_test, y_test)


# ----------------------------------------------------------------------------------------------------------------------

evaluate_models("breast_cancer", (-1, 1))
evaluate_models("diabetes", (-1, 1))
