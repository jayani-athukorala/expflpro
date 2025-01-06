import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset

fds = None  # Global variable to cache FederatedDataset

# Function to load federated data for a given partition
def load_data(partition_id: int, num_partitions: int, dataset_path="data"):
    """
    Load and preprocess federated data for a specific partition.

    Parameters:
    - partition_id (int): Partition ID to load.
    - num_partitions (int): Total number of partitions.
    - dataset_path (str): Path to the dataset.

    Returns:
    - partition_data (DataFrame): Data for the specified partition.
    """
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(dataset=dataset_path, partitioners={"train": partitioner})

    # Load data for the specified partition
    partition = fds.load_partition(partition_id, "train")
    partition_data = pd.DataFrame(partition)

    X_train, X_test, X_explain, y_train, y_test, y_explain, mappings = prepare_data_fl(partition_data)
    return X_train, X_test, X_explain, y_train, y_test, y_explain, mappings

# Prepare data for FL environment
def prepare_data_fl(data):
    """Preprocess the dataset: encode, normalize, and split."""
    mappings = {}
   
    # Encode categorical features
    le_gender = LabelEncoder()
    data['Gender'] = le_gender.fit_transform(data['Gender'])  # Male: 1, Female: 0
    mappings['Gender'] = dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))
    
    le_bmicase = LabelEncoder()
    data['BMIcase'] = le_bmicase.fit_transform(data['BMIcase'])

    # Save unnormalized data for explanations
    explanation_data = data.copy()

    # Normalize numerical features
    scaler = MinMaxScaler()
    data[['Weight', 'Height', 'BMI', 'Age']] = scaler.fit_transform(data[['Weight', 'Height', 'BMI', 'Age']])

    # Extract features and target
    features = ['Weight', 'Height', 'BMI', 'Age', 'Gender']
    target = 'Exercise Recommendation Plan'

    X = data[features]
    y = data[target] - 1  # Adjust target to start from 0

    # Split into train, test, and explanation sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_explain, y_test, y_explain = train_test_split(X_temp, y_temp, test_size=0.333, random_state=42)

    # Keep explanation set unnormalized
    X_explain = explanation_data.iloc[X_explain.index][features]

    return X_train, X_test, X_explain, y_train, y_test, y_explain, mappings

# Prepare data for centralized environment
def prepare_data(data):
    """Preprocess the dataset: encode, normalize, and split."""
    
    mappings = {}
   
    # Encode categorical features
    le_gender = LabelEncoder()
    data['Gender'] = le_gender.fit_transform(data['Gender'])  # Male: 1, Female: 0
    mappings['Gender'] = dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))
    
    le_bmicase = LabelEncoder()
    data['BMIcase'] = le_bmicase.fit_transform(data['BMIcase'])

    # Save unnormalized data for explanations
    explanation_data = data.copy()

    # Normalize numerical features
    scaler = MinMaxScaler()
    data[['Weight', 'Height', 'BMI', 'Age']] = scaler.fit_transform(data[['Weight', 'Height', 'BMI', 'Age']])

    # Extract features and target
    features = ['Weight', 'Height', 'BMI', 'Age', 'Gender']
    target = 'Exercise Recommendation Plan'

    X = data[features]
    y = data[target] - 1  # Adjust target to start from 0

    # Split into train, test, and explanation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _, X_explain, _, y_explain = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Keep explanation set unnormalized
    X_explain = explanation_data.iloc[X_explain.index][features]

    return X_train, X_test, X_explain, y_train, y_test, y_explain, mappings
