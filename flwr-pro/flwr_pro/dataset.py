import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_from_disk
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
import pandas as pd
import os, sys

# Add BMI Category
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Preprocess and save the dataset
def preprocess_and_save(data_dir="flwr_pro/data/raw", save_dir="flwr_pro/data/processed"):
    """
    Preprocess the exercise dataset and save it in csv format.

    Parameters:
    - data_dir (str): Directory containing the raw dataset.
    - save_dir (str): Directory to save the processed dataset.

    Returns:
    - save_dir (str): Path to the saved dataset.
    """
    file_path = os.path.join(data_dir, "exercise_dataset.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    # Load and preprocess the dataset
    data = pd.read_csv(file_path).dropna()

    # Encode categorical variables
    if "Gender" in data.columns:
        data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
    if "Exercise" in data.columns:
        data["Exercise"] = data["Exercise"].map({
            "Exercise 1": 0, "Exercise 2": 1, "Exercise 3": 2, "Exercise 4": 3,
            "Exercise 5": 4, "Exercise 6": 5, "Exercise 7": 6, "Exercise 8": 7,
            "Exercise 9": 8, "Exercise 10": 9
        })
    if "Weather Conditions" in data.columns:
        data["Weather Conditions"] = data["Weather Conditions"].map({"Sunny": 1, "Cloudy": 2, "Rainy" : 3})


    # Ensure data is not empty
    if data.empty:
        raise ValueError("The input dataset is empty after preprocessing.")

    # Define features and target
    feature_columns = ['Calories Burn','Dream Weight', 'Actual Weight', 'Age', 'Gender',
                       'Duration', 'Heart Rate', 'BMI', 'Exercise Intensity']
    target_column = 'Exercise'

    X = data[feature_columns]
    y = data[target_column]

    # Combine features and target
    processed_data = pd.concat([X, y], axis=1)

    # Ensure processed data is not empty
    if processed_data.empty:
        raise ValueError("Processed data is empty after combining features and target.")

    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "exercise_dataset.csv")
    processed_data.to_csv(csv_path, index=False)

    return 0

# Load the preprocessed dataset
def load_data(partition_id: int, num_partitions: int, data_dir="flwr_pro/data/raw", save_dir="flwr_pro/data/processed"):
    """
    Load and partition exercise dataset for federated learning using FederatedDataset.

    Parameters:
    - partition_id (int): ID of the partition to load.
    - num_partitions (int): Number of partitions (clients).
    - data_dir (str): Directory containing exercise_dataset.csv.
    - save_dir (str): Directory containing processed dataset.

    Returns:
    - train_loader (DataLoader): DataLoader for training data.
    - test_loader (DataLoader): DataLoader for test data.
    """
    # Preprocess and save the dataset if not already done
    if not os.path.exists(save_dir):
        preprocess_and_save(data_dir, save_dir)

    # Load the processed dataset
    dataset_path = save_dir

    # Initialize FederatedDataset
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset=dataset_path,  # Pass the path to the processed dataset
        partitioners={"train": partitioner},
    )

    # Load partition and split into train and test
    partition = fds.load_partition(partition_id, "train")
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Convert to PyTorch-compatible tensors
    train_data = partition_train_test["train"].to_pandas()
    test_data = partition_train_test["test"].to_pandas()

    feature_columns = ['Calories Burn','Dream Weight', 'Actual Weight', 'Age', 'Gender',
                       'Duration', 'Heart Rate', 'BMI', 'Exercise Intensity']
    target_column = 'Exercise'

    
    X_train = train_data[feature_columns].values
    y_train = train_data[target_column].values
    X_test = test_data[feature_columns].values
    y_test = test_data[target_column].values

    #sys.exit(y_train)
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

