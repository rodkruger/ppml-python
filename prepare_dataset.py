import pandas as pd
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

CONTENT_PATH = '/home/rkruger/Doutorado'

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Normalize the dataset for sigmoid activation [-2, 2]
sigmoid_scaler = MinMaxScaler(feature_range=(-2, 2))
X_sigmoid = sigmoid_scaler.fit_transform(X)

# Normalize the dataset for tanh activation [0, 1]
tanh_scaler = MinMaxScaler(feature_range=(0, 1))
X_tanh = tanh_scaler.fit_transform(X)

# Split the datasets into training and testing sets
X_train_sigmoid, X_test_sigmoid, y_train_sigmoid, y_test_sigmoid = (
    train_test_split(X_sigmoid, y, test_size=0.3, stratify=y, random_state=42))

# Save splitted datasets into the repository
pd.DataFrame(X_train_sigmoid).to_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/training_features_range22.csv',
                                     index=False, header=False)
pd.DataFrame(y_train_sigmoid).to_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/training_labels_range22.csv',
                                     index=False, header=False)
pd.DataFrame(X_test_sigmoid).to_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/testing_features_range22.csv',
                                    index=False, header=False)
pd.DataFrame(y_test_sigmoid).to_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/testing_labels_range22.csv',
                                    index=False, header=False)

# Save splitted datasets into the repository
X_train_tanh, X_test_tanh, y_train_tanh, y_test_tanh = (
    train_test_split(X_tanh, y, test_size=0.3, stratify=y, random_state=42))

pd.DataFrame(X_train_tanh).to_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/training_features_range01.csv',
                                  index=False, header=False)
pd.DataFrame(y_train_tanh).to_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/training_labels_range01.csv',
                                  index=False, header=False)
pd.DataFrame(X_test_tanh).to_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/testing_features_range01.csv',
                                 index=False, header=False)
pd.DataFrame(y_test_tanh).to_csv(f'{CONTENT_PATH}/Datasets/breast_cancer/testing_labels_range01.csv',
                                 index=False, header=False)
