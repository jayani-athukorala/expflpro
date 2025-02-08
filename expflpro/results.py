import numpy as np
import os
import pandas as pd
import shutil
import json
import csv

# Save evaluation metrics for centralized models
def save_evaluation_metrics(metrics, output_folder='../results/ml/evaluations', filename='evaluation_metrics.json'):
    """
    Save evaluation metrics as a JSON file.

    Parameters:
    - metrics (dict): Dictionary of evaluation metrics (e.g., RMSE, MAE, Precision@K).
    - output_folder (str): Path to save the metrics (default: '../results/ml/evaluations').
    - filename (str): Name of the output JSON file (default: 'evaluation_metrics.json').
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(output_folder, filename)

    # Save metrics to a JSON file
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)  # Indent for better readability
    print(f"Metrics saved to {file_path}")

# Save evaluation metrics for FL model rounds
def save_local_evaluations(metrics, current_round, partition_id, results_dir="results/fl/evaluations"):
    """
    Save evaluation metrics for a specific client (partition) and round to a JSON file.

    Parameters:
    - metrics (dict): Dictionary containing evaluation metrics for the current round.
    - current_round (int): Current federated learning round number.
    - partition_id (str): Unique identifier for the client or partition.
    - results_dir (str): Directory to save the client-specific results files (default: 'results/fl/evaluations').
    """
    # Ensure the directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Generate the file name for the client/partition
    filename = f"{partition_id}_evaluations.json"
    filepath = os.path.join(results_dir, filename)

    # If the file exists, load existing results; otherwise, initialize an empty list
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            client_results = json.load(f)  # Load existing results
    else:
        client_results = []

    # Add the current round's metrics to the client results
    metrics["round"] = current_round
    client_results.append(metrics)

    # Save the updated results back to the client-specific JSON file
    with open(filepath, "w") as f:
        json.dump(client_results, f, indent=4)

    print(f"Round {current_round} metrics for partition {partition_id} saved to {filepath}")

# Save evaluation results for each round in federated learning globally
def save_evaluation_results_round(results, round_number, results_dir="results/fl/evaluations", filename="evaluation_metrics.json"):
    """
    Append evaluation results of a specific round to a JSON file for federated learning.

    Parameters:
    - results (dict): Dictionary containing evaluation metrics for the round.
    - round_number (int): Current federated learning round number.
    - results_dir (str): Directory to save the results file (default: 'results/fl/evaluations').
    - filename (str): Name of the results file (default: 'evaluation_metrics.json').
    """
    # Ensure the directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Full path to the results file
    filepath = os.path.join(results_dir, filename)

    # If the file exists, load existing results; otherwise, initialize an empty list
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            all_results = json.load(f)  # Load existing results
    else:
        all_results = []

    # Append the current round's results to the list
    results["round"] = round_number
    all_results.append(results)

    # Save the updated results back to the JSON file
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=4)  # Indent for better readability
    print(f"Round {round_number} results saved to {filepath}")

# Reset the results folder at the start of a new run
def reset_results_folder(results_folder="results"):
    """
    Clear all contents of the results folder to prepare for a new run.

    Parameters:
    - results_folder (str): Path to the results folder (default: 'results').
    """
    # If the folder exists, remove it and all its contents
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)  # Delete the folder and its contents

    # Recreate the folder to start fresh
    os.makedirs(results_folder)
    print(f"Results folder reset: {results_folder}")



def save_list_as_json(data, output_folder, filename, partition_id=None):
    """
    Save a list of data as a JSON file in the specified output folder.

    Parameters:
    - data (list): The list to save (recommendations and explanations).
    - output_folder (str): Path to the output folder.
    - filename (str): Name of the JSON file.
    - partition_id (int): If specified, create a subfolder with this name inside the output folder.

    Returns:
    - str: Path to the saved JSON file.
    """
    # Ensure the filename has the correct extension
    if not filename.endswith('.json'):
        raise ValueError("Filename must end with '.json'")

    # If a partition ID is provided, append it to the output folder path
    if partition_id is not None:
        output_folder = os.path.join(output_folder, f"client_{partition_id}")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full path for the output file
    output_file_path = os.path.join(output_folder, filename)

    # Save the data as a JSON file with UTF-8 encoding
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"File saved successfully at {output_file_path}")
    return output_file_path  # Return the file path for reference


def save_list_as_csv(data, output_folder, filename, partition_id=None):
    """
    Save a list of dictionaries as a CSV file in the specified output folder using pandas.

    Parameters:
    - data (list of dict): List of dictionaries to save.
    - output_folder (str): Path to the output folder.
    - filename (str): Name of the CSV file.
    - partition_id (int): If specified, create a subfolder with this name inside the output folder.

    Returns:
    - str: Path to the saved CSV file.
    """
    # Ensure the filename has the correct extension
    if not filename.endswith('.csv'):
        raise ValueError("Filename must end with '.csv'")

    # If a partition ID is provided, append it to the output folder path
    if partition_id is not None:
        output_folder = os.path.join(output_folder, f"client_{partition_id}")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full path for the output file
    output_file_path = os.path.join(output_folder, filename)

    # Convert list of dictionaries to a DataFrame and save to CSV
    recommendations_df = pd.DataFrame(data)
    recommendations_df.to_csv(output_file_path, index=False)

    print(f"CSV file saved successfully at {output_file_path}")
    return output_file_path  # Return the file path for reference


def save_dict_as_csv(data, output_folder, filename, partition_id=None):
    """
    Save a dictionary of data as a CSV file in the specified output folder.

    Each dictionary key represents a row, and the values are stored in a single cell as lists or strings.

    Parameters:
    - data (dict): Dictionary to save.
    - output_folder (str): Path to the output folder.
    - filename (str): Name of the CSV file.
    - partition_id (int): If specified, create a subfolder with this name inside the output folder.

    Returns:
    - str: Path to the saved CSV file.
    """
    # Ensure the filename has the correct extension
    if not filename.endswith('.csv'):
        raise ValueError("Filename must end with '.csv'")

    # If a partition ID is provided, append it to the output folder path
    if partition_id is not None:
        output_folder = os.path.join(output_folder, f"client_{partition_id}")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full path for the output file
    output_file_path = os.path.join(output_folder, filename)

    # Open the CSV file for writing
    with open(output_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write rows dynamically
        for key, value in data.items():
            # Ensure the value is a list or string for writing
            if not isinstance(value, (list, tuple, str)):
                value = [value]
            writer.writerow([key] + list(value))  # Write key as the first column, followed by the values

    print(f"CSV file saved successfully at {output_file_path}")
    return output_file_path  # Return the file path for reference

