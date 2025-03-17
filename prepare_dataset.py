import pandas as pd
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

CONTENT_PATH = '/home/rkruger/Doutorado'


# ----------------------------------------------------------------------------------------------------------------------

def split_dataset(p_dataset_name, p_features, p_labels, p_range):
    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(p_range[0], p_range[1]))
    p_features_norm = scaler.fit_transform(p_features)

    # Split the datasets into training and testing sets
    x_train, x_test, y_train, y_test = (
        train_test_split(p_features_norm, p_labels, test_size=0.3, stratify=p_labels, random_state=42))

    # Save splitted datasets into the repository
    pd.DataFrame(x_train).to_csv(
        f'{CONTENT_PATH}/Datasets/{p_dataset_name}/training_features_range{abs(p_range[0])}{abs(p_range[1])}.csv',
        index=False, header=False)

    pd.DataFrame(y_train).to_csv(
        f'{CONTENT_PATH}/Datasets/{p_dataset_name}/training_labels_range{abs(p_range[0])}{abs(p_range[1])}.csv',
        index=False, header=False)

    pd.DataFrame(x_test).to_csv(
        f'{CONTENT_PATH}/Datasets/{p_dataset_name}/testing_features_range{abs(p_range[0])}{abs(p_range[1])}.csv',
        index=False, header=False)

    pd.DataFrame(y_test).to_csv(
        f'{CONTENT_PATH}/Datasets/{p_dataset_name}/testing_labels_range{abs(p_range[0])}{abs(p_range[1])}.csv',
        index=False, header=False)


# ----------------------------------------------------------------------------------------------------------------------
# Split Breast Cancer dataset
# DATA = load_breast_cancer()
# FEATURES = data.data
# LABELS = data.target

# split_dataset("breast_cancer", FEATURES, LABELS, (-2, 2))
# split_dataset("breast_cancer", FEATURES, LABELS, (-1, 1))
# split_dataset("breast_cancer", FEATURES, LABELS, (0, 1))

# ----------------------------------------------------------------------------------------------------------------------
# Split Pima Indians Diabetes Database (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
DATASET_NAME = "diabetes"
DATA = pd.read_csv(f"{CONTENT_PATH}/Datasets/{DATASET_NAME}/diabetes.csv")
FEATURES = DATA.iloc[:, :-1].to_numpy()
LABELS = DATA.iloc[:, -1].to_numpy()

split_dataset(DATASET_NAME, FEATURES, LABELS, (-2, 2))
split_dataset(DATASET_NAME, FEATURES, LABELS, (-1, 1))
split_dataset(DATASET_NAME, FEATURES, LABELS, (0, 1))
