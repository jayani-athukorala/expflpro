import os
import json
import pandas as pd
import shutil

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


# Save evaluation results for each round in federated learning
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


# Save a list as a JSON file
def save_list_as_json(data, output_folder, filename, partition_id=None):
    """
    Save a list of data as a JSON file in the specified output folder.

    Parameters:
    - data (list): The list to save (e.g., predictions, metrics, or other data).
    - output_folder (str): Path to the output folder.
    - filename (str): Name of the JSON file (e.g., 'filename.json').
    - partition_id (str or int, optional): If specified, create a subfolder with this name inside the output folder.

    Returns:
    - str: Path to the saved JSON file.
    """
    # Ensure the filename has the correct extension
    if not filename.endswith('.json'):
        raise ValueError("Filename must end with '.json'")

    # If a partition ID is provided, append it to the output folder path
    if partition_id is not None:
        output_folder = os.path.join(output_folder, str(partition_id))

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full path for the output file
    output_file_path = os.path.join(output_folder, filename)

    # Save the data as a JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # Indent for better readability

    print(f"File saved successfully at {output_file_path}")
    return output_file_path  # Return the file path for reference


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

