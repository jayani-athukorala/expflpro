import json
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

def load_json(filepath):
    """
    Load JSON data from a file.
    
    Parameters:
    - filepath (str): Path to the JSON file.
    
    Returns:
    - dict or list: Parsed JSON data.
    """
    with open(filepath, "r") as f:
        return json.load(f)

def plot_all_metrics_comparison(json1_path, json2_path, output_folder):
    """
    Generate and save comparison graphs for all metrics in the JSON datasets.

    Parameters:
    - json1_path (str): Path to the first JSON file (e.g., FL evaluations with rounds).
    - json2_path (str): Path to the second JSON file (e.g., ML constant metrics).
    - output_folder (str): Folder to save the generated plots.

    Returns:
    - None
    """
    # Load JSON data
    json1_data = load_json(json1_path)
    json2_data = load_json(json2_path)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract rounds from the first JSON
    rounds = [entry["round"] for entry in json1_data]

    # Get all metrics available in the first JSON (excluding "round")
    metrics = [key for key in json1_data[0] if key != "round"]

    # Generate comparison plots for each metric
    for metric in metrics:
        # Extract metric values from the first JSON
        json1_values = [entry[metric] for entry in json1_data]

        # Extract the constant value for the metric from the second JSON
        json2_value = json2_data.get(metric)
        if json2_value is None:
            print(f"Metric '{metric}' not found in the second JSON. Skipping...")
            continue

        # Prepare a constant line for the second JSON
        json2_values = [json2_value] * len(rounds)

        # Plot the metric comparison
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, json1_values, label=f"{metric} (FL)", marker='o', linestyle='-', color='blue')
        plt.plot(rounds, json2_values, label=f"{metric} (ML - Constant)", linestyle='--', color='red')
        plt.title(f"Comparison of {metric} Across Rounds", fontsize=16)
        plt.xlabel("Rounds", fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

        # Save the plot
        file_path = os.path.join(output_folder, f"{metric}_comparison.png")
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
        plt.close()


def generate_combined_evaluation_graphs(num_clients, results_dir="../results/fl/evaluations", plots_dir="../results/plots/evaluations"):
    """
    Generate and save graphs for each metric, showing all clients' metrics and global metrics across rounds.

    Parameters:
    - num_clients (int): Number of clients (partitions) in the federated learning setup.
    - results_dir (str): Directory containing evaluation result files (default: 'results/fl/evaluations').
    - plots_dir (str): Directory to save the generated plots (default: 'results/plots/evaluations').
    """
    # Ensure the plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Load global evaluation metrics
    global_metrics_file = os.path.join(results_dir, "evaluation_metrics.json")
    with open(global_metrics_file, "r") as f:
        global_results = json.load(f)

    # Extract rounds and global metrics
    rounds = [entry["round"] for entry in global_results]
    global_metrics = {key: [entry[key] for entry in global_results if key != "round"] for key in global_results[0] if key != "round"}

    # Collect metrics for all clients
    client_metrics = {}
    for client_id in range(num_clients):
        client_file = os.path.join(results_dir, f"{client_id}_evaluations.json")
        with open(client_file, "r") as f:
            client_results = json.load(f)

        # Extract client metrics
        for metric_name in client_results[0]:
            if metric_name != "round":
                if metric_name not in client_metrics:
                    client_metrics[metric_name] = {}
                client_metrics[metric_name][client_id] = [entry[metric_name] for entry in client_results]

    # Generate a graph for each metric
    for metric_name in client_metrics:
        plt.figure(figsize=(12, 8))

        # Plot each client's metric
        for client_id, metrics in client_metrics[metric_name].items():
            plt.plot(rounds, metrics, label=f"Client {client_id}", marker="o")

        # Plot global metric
        plt.plot(rounds, global_metrics[metric_name], label="Global", marker="x", linestyle="--", linewidth=2, color="black")

        # Add labels, legend, and title
        plt.xlabel("Rounds")
        plt.ylabel(metric_name.capitalize())
        plt.title(f"Comparison of {metric_name.capitalize()} Across Rounds (All Clients and Global)")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_filename = f"{metric_name}_all_clients_vs_global.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved plot: {plot_path}")

def main():

    # Plot the comparison of ML and FL 
    json1_path = "../results/fl/evaluations/evaluation_metrics.json"
    json2_path = "../results/ml/evaluations/evaluation_metrics.json"
    output_folder = "../results/plots"
    plot_all_metrics_comparison(json1_path, json2_path, output_folder)

    ###############################################
    # Plot the comparison of local and blobal evaluations FL     
    num_clients = 10
    generate_combined_evaluation_graphs(num_clients)

    ###############################################
    json1_path = "../results/ml/explanations/shap/explanations_for_user_204_recommendations.json"  # Metrics of FL rounds
    json2_path = "../results/ml/explanations/lime/explanations_for_user_204_recommendations.json"

    # Load JSON data
    lime_json = load_json(json2_path)
    shap_json = load_json(json1_path)

    # Create the output folder if it doesn't exist
    output_folder = "../results/plots/explanations"
    os.makedirs(output_folder, exist_ok=True)

    # plot_shap_explanations(json1_path, output_folder)
    plot_lime_explanations(lime_json, output_folder)
    plot_lime_pie_chart(lime_json, output_folder)
    plot_lime_bar_chart(lime_json, output_folder)
    plot_lime_pos_neg_contributions(lime_json, output_folder)
    plot_lime_cumulative_contributions(lime_json, output_folder)

    # plot_shap_bar_chart(shap_json, output_folder)
    # plot_shap_waterfall(shap_json, output_folder)

# Run the script
if __name__ == "__main__":
    main()
    