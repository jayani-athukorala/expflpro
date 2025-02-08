import os
import json
from scipy.stats import spearmanr
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_explanations(path, explanation_type="lime"):
    filename = (
        "lime_explanations_for_user_0_recommendations.json"
        if explanation_type == "lime"
        else "shap_explanations_for_user_0_recommendations.json"
    )
    file_path = os.path.join(path, filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_local_explanations(path, explanation_type="lime"):
    filename = (
        "lime_explanations_for_user_0_recommendations.json"
        if explanation_type == "lime"
        else "shap_explanations_for_user_0_recommendations.json"
    )
    explanations = {}
    for client_id in range(10):
        client_folder = os.path.join(path, f"client_{client_id}", "explanations")
        file_path = os.path.join(client_folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                explanations[f"client_{client_id}"] = json.load(f)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    return explanations


def evaluate_privacy(explanations, sensitive_features, explanation_type="lime"):
    dominance = {}
    for key, explanation in explanations.items():
        if explanation_type == "lime":
            sensitive_importance = sum(
                f["importance"]
                for f in explanation["feature_importances"]
                if any(sf in f["feature"] for sf in sensitive_features)
            )
        elif explanation_type == "shap":
            sensitive_importance = sum(
                abs(row[i])
                for row in explanation["shap_values"]
                for i, feature in enumerate(explanation["feature_values"])
                if feature in sensitive_features
            )
        else:
            raise ValueError(f"Unsupported explanation type: {explanation_type}")
        dominance[key] = sensitive_importance
    return dominance


def visualize_results(results, title, xlabel, ylabel, save_path):
    """
    Visualize results as a bar plot and save to the specified path.
    Args:
        results (dict): Data to visualize.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        save_path (str): File path to save the plot.
    """
    df = pd.DataFrame(results.items(), columns=["Metric", "Value"])
    sns.barplot(x="Metric", y="Value", data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_results_to_csv(results, save_path):
    """
    Save results to a CSV file.
    Args:
        results (dict): Results data to save.
        save_path (str): File path to save the CSV file.
    """
    df = pd.DataFrame(results.items(), columns=["Metric", "Value"])
    df.to_csv(save_path, index=False)


def evaluate_global_vs_local(global_explanation, local_explanations):
    """
    Compare global explanations with averaged local explanations.
    Args:
        global_explanation (dict): Global explanation data.
        local_explanations (dict): Local explanation data for all clients.
    Returns:
        float: Spearman correlation between global and averaged local explanations.
    """
    # Extract the global feature importances (access inside "explanations")
    global_importances = [
        f["importance"] for f in global_explanation["feature_importances"]
    ]
    
    # Compute average feature importances across all local explanations
    avg_local_importances = [
        sum(
            [
                client_exp["feature_importances"][i]["importance"]
                for client_exp in local_explanations.values()
            ]
        ) / len(local_explanations)
        for i in range(len(global_importances))
    ]

    # Compute Spearman correlation
    spearman, _ = spearmanr(global_importances, avg_local_importances)
    return spearman



def evaluate_global_vs_centralized(global_explanation, centralized_explanation):
    """
    Compare global explanations (from FL) with centralized explanations (from ML).
    
    Args:
        global_explanation (dict): Global explanation data from federated learning.
        centralized_explanation (dict): Centralized explanation data from traditional ML.
    
    Returns:
        float: Spearman correlation between global and centralized feature importances.
    """
    # Extract the feature importances for both global and centralized explanations
    global_importances = [
        f["importance"] for f in global_explanation["feature_importances"]
    ]
    centralized_importances = [
        f["importance"] for f in centralized_explanation["feature_importances"]
    ]

    # Ensure the lengths match (align features)
    if len(global_importances) != len(centralized_importances):
        raise ValueError("Mismatch in the number of features between global and centralized explanations.")

    # Compute Spearman correlation
    spearman, _ = spearmanr(global_importances, centralized_importances)
    return spearman


def evaluate_shap_global_vs_local(global_explanation, local_explanations):
    """
    Compare global SHAP explanations with averaged local SHAP explanations.
    Args:
        global_explanation (dict): Global SHAP explanation.
        local_explanations (dict): Local SHAP explanations for all clients.
    Returns:
        float: Spearman correlation between global and averaged local SHAP values.
    """
    # Number of features
    num_features = len(global_explanation["shap_values"][0])

    # Average SHAP values across local explanations (feature-wise)
    avg_local_shap_values = [
        sum(
            sum(client_exp["shap_values"][row][feature] for row in range(len(client_exp["shap_values"])))
            for client_exp in local_explanations.values()
        ) / (len(local_explanations) * len(next(iter(local_explanations.values()))["shap_values"]))
        for feature in range(num_features)
    ]

    # Aggregate global SHAP values (average across rows for each feature)
    global_shap_values = [
        sum(row[feature] for row in global_explanation["shap_values"]) / len(global_explanation["shap_values"])
        for feature in range(num_features)
    ]

    # Compute Spearman correlation
    spearman, _ = spearmanr(global_shap_values, avg_local_shap_values)
    return spearman

# Evaluate pairwise consistency correlations
def evaluate_local_consistency_correlations(local_explanations, explanation_type="lime"):
    """
    Compare explanations across local clients for consistency using Spearman correlations.
    Args:
        local_explanations (dict): Local explanations for all clients.
        explanation_type (str): Explanation type ("lime" or "shap").
    Returns:
        dict: Pairwise Spearman correlations between clients.
    """
    client_names = list(local_explanations.keys())
    correlations = {}

    for i, client_a in enumerate(client_names):
        for j, client_b in enumerate(client_names):
            if i < j:
                if explanation_type == "lime":
                    # Extract feature importances for LIME
                    values_a = [f["importance"] for f in local_explanations[client_a]["feature_importances"]]
                    values_b = [f["importance"] for f in local_explanations[client_b]["feature_importances"]]
                elif explanation_type == "shap":
                    # Extract feature importances for SHAP
                    values_a = [
                        sum(abs(row[i]) for row in local_explanations[client_a]["shap_values"]) 
                        for i in range(len(local_explanations[client_a]["feature_values"]))
                    ]
                    values_b = [
                        sum(abs(row[i]) for row in local_explanations[client_b]["shap_values"]) 
                        for i in range(len(local_explanations[client_b]["feature_values"]))
                    ]
                else:
                    raise ValueError(f"Unsupported explanation type: {explanation_type}")

                # Compute Spearman Correlation
                spearman, _ = spearmanr(values_a, values_b)
                correlations[f"{client_a} vs {client_b}"] = spearman

    return correlations


def evaluate_shap_global_vs_centralized(global_explanation, centralized_explanation):
    """
    Compare global SHAP explanations with centralized SHAP explanations.
    Args:
        global_explanation (dict): Global SHAP explanation.
        centralized_explanation (dict): Centralized SHAP explanation.
    Returns:
        float: Spearman correlation between global and centralized SHAP values.
    """
    # Aggregate global SHAP values (average across rows for each feature)
    global_shap_values = [
        sum(feature_shap_values) / len(feature_shap_values)
        for feature_shap_values in zip(*global_explanation["shap_values"])
    ]

    # Aggregate centralized SHAP values (average across rows for each feature)
    centralized_shap_values = [
        sum(feature_shap_values) / len(feature_shap_values)
        for feature_shap_values in zip(*centralized_explanation["shap_values"])
    ]

    # Compute Spearman correlation
    spearman, _ = spearmanr(global_shap_values, centralized_shap_values)
    return spearman


def evaluate_local_consistency(explanations, explanation_type="lime", top_k=None):
    """
    Evaluate the consistency of local explanations across clients.
    Args:
        explanations (dict): Explanations for all clients.
        explanation_type (str): Explanation type ("lime" or "shap").
        top_k (int): Number of top features to consider for Jaccard similarity (optional).
    Returns:
        dict: Average Spearman and Jaccard scores across all clients.
    """
    client_keys = list(explanations.keys())
    if len(client_keys) < 2:
        return {"Spearman Avg": None, "Jaccard Avg": None}

    spearman_scores = []
    jaccard_scores = []

    for i, client_a in enumerate(client_keys):
        for j, client_b in enumerate(client_keys):
            if i < j:
                # Extract feature importances
                if explanation_type == "lime":
                    imp_a = [f["importance"] for f in explanations[client_a]["feature_importances"]]
                    imp_b = [f["importance"] for f in explanations[client_b]["feature_importances"]]
                elif explanation_type == "shap":
                    imp_a = [
                        sum(abs(row[i]) for row in explanations[client_a]["shap_values"]) 
                        for i in range(len(explanations[client_a]["feature_values"]))
                    ]
                    imp_b = [
                        sum(abs(row[i]) for row in explanations[client_b]["shap_values"]) 
                        for i in range(len(explanations[client_b]["feature_values"]))
                    ]
                else:
                    raise ValueError(f"Unsupported explanation type: {explanation_type}")

                # Spearman Correlation
                spearman, _ = spearmanr(imp_a, imp_b)
                spearman_scores.append(spearman)

                # Jaccard Similarity for top-k features
                if top_k:
                    top_k_a = set(sorted(range(len(imp_a)), key=lambda x: -imp_a[x])[:top_k])
                    top_k_b = set(sorted(range(len(imp_b)), key=lambda x: -imp_b[x])[:top_k])
                    jaccard = len(top_k_a.intersection(top_k_b)) / len(top_k_a.union(top_k_b))
                    jaccard_scores.append(jaccard)

    return {
        "Spearman Avg": sum(spearman_scores) / len(spearman_scores),
        "Jaccard Avg": sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else None,
    }



import matplotlib.pyplot as plt
import numpy as np

def visualize_multi_bar(data, title, x_label, y_label, output_path):
    """
    Create a grouped bar chart for multiple categories with specific colors.
    
    Args:
        data (dict): Nested dictionary with format:
                     {category1: {sub_cat1: value, sub_cat2: value, ...}, 
                      category2: {sub_cat1: value, sub_cat2: value, ...}, ...}.
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        output_path (str): File path to save the plot.
    """
    # Extract category names and sub-categories
    categories = list(data.keys())
    sub_categories = list(data[categories[0]].keys())
    
    # Number of groups and bars
    num_groups = len(sub_categories)
    num_categories = len(categories)
    
    # Set positions for the bars
    bar_width = 0.2
    x = np.arange(num_groups)
    
    # Define colors for bars
    colors = ['tomato', 'royalblue']  # Add more colors if needed for more categories
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    for i, category in enumerate(categories):
        plt.bar(
            x + i * bar_width,
            [data[category][sub_cat] for sub_cat in sub_categories],
            width=bar_width,
            label=category,
            color=colors[i % len(colors)]  # Cycle through colors if more than two categories
        )
    
    # Customize the plot
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x + bar_width * (num_categories / 2 - 0.5), sub_categories)
    plt.legend()
    
    # Add value annotations
    for i, category in enumerate(categories):
        for j, value in enumerate(data[category].values()):
            plt.text(
                x[j] + i * bar_width,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def load_overhead_data(base_path, clients_range, global_path, centralized_path):
    """
    Load overhead data from local, global, and centralized JSON files.
    """
    data = {"local": {}, "global": {}, "centralized": {}}

    # Load local overheads
    for client in clients_range:
        client_path = os.path.join(base_path, f"client_{client}", "overheads", "overhead_for_user_0_recommendation_explanations.json")
        if os.path.exists(client_path):
            with open(client_path, 'r') as f:
                data["local"][f"client_{client}"] = json.load(f)

    # Load global overheads
    global_file = os.path.join(global_path, "overhead_for_user_0_recommendation_explanations.json")
    if os.path.exists(global_file):
        with open(global_file, 'r') as f:
            data["global"] = json.load(f)

    # Load centralized overheads
    centralized_file = os.path.join(centralized_path, "overhead_for_user_0_recommendation_explanations.json")
    if os.path.exists(centralized_file):
        with open(centralized_file, 'r') as f:
            data["centralized"] = json.load(f)

    return data

def organize_data(data):
    """
    Organize overhead data into a DataFrame for analysis.
    """
    rows = []
    for client, overheads in data["local"].items():
        computation = overheads.get("computation", {})
        rows.append([client, "Recommendation Only", "local", computation["recommendation"].get("Computation Time (s)", 0), computation["recommendation"].get("Memory Usage (MB)", 0)])
        rows.append([client, "Recommendation + SHAP", "local", computation["shap"].get("Computation Time (s)", 0), computation["shap"].get("Memory Usage (MB)", 0)])
        rows.append([client, "Recommendation + LIME", "local", computation["lime"].get("Computation Time (s)", 0), computation["lime"].get("Memory Usage (MB)", 0)])

    for impl in ["global", "centralized"]:
        computation = data[impl].get("computation", {})
        rows.append([impl, "Recommendation Only", impl, computation["recommendation"].get("Computation Time (s)", 0), computation["recommendation"].get("Memory Usage (MB)", 0)])
        rows.append([impl, "Recommendation + SHAP", impl, computation["shap"].get("Computation Time (s)", 0), computation["shap"].get("Memory Usage (MB)", 0)])
        rows.append([impl, "Recommendation + LIME", impl, computation["lime"].get("Computation Time (s)", 0), computation["lime"].get("Memory Usage (MB)", 0)])

    df = pd.DataFrame(rows, columns=["Client", "Method", "Implementation", "Computation Time (s)", "Memory Usage (MB)"])
    return df

def plot_overheads(df, save_dir="../results/plots/exp_evaluations"):
    """
    Generate and save grouped bar plots for overhead data, highlighting centralized and global FL.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Color coding for each method
    method_colors = {
        "Recommendation Only": "gray",
        "Recommendation + SHAP": "royalblue",
        "Recommendation + LIME": "tomato"
    }

    implementations = ["centralized", "global"] + [f"client_{i}" for i in range(10)]
    bar_width = 0.2

    for metric in ["Computation Time (s)", "Memory Usage (MB)"]:
        plt.figure(figsize=(16, 8))
        x = range(len(implementations))

        # Plot grouped bars for each method
        for i, method in enumerate(["Recommendation Only", "Recommendation + SHAP", "Recommendation + LIME"]):
            subset = df[df["Method"] == method]

            # Extract values
            values = [
                subset[subset["Client"] == impl][metric].values[0] if not subset[subset["Client"] == impl].empty else 0
                for impl in implementations
            ]

            # Apply highlighting for Centralized and Global FL
            bar_widths = [bar_width * 1.3 if impl in ["centralized", "global"] else bar_width for impl in implementations]
            border_thickness = [2.5 if impl in ["centralized", "global"] else 0.8 for impl in implementations]
            edge_colors = ['black' if impl in ["centralized", "global"] else 'none' for impl in implementations]

            # Plot each bar individually to apply custom styles
            for j, (value, width, edge, thickness) in enumerate(zip(values, bar_widths, edge_colors, border_thickness)):
                plt.bar(
                    x[j] + bar_width * i,
                    value,
                    width=width,
                    color=method_colors[method],
                    edgecolor=edge,
                    linewidth=thickness,
                    alpha=0.9,
                    label=method if j == 0 else ""
                )

        plt.xlabel("Implementation (Centralized, Global FL, Local Clients)", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f"{metric} Comparison Before and After Explainability", fontsize=14)
        plt.xticks([p + bar_width for p in x], implementations, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot
        filename = f"{metric.replace(' ', '_')}_comparison.png"
        plt.savefig(os.path.join(save_dir, filename))
        print(f"Saved plot: {filename}")
        plt.close()


def main():
    # Paths for saving plots and tables
    output_dir = "../results/plots/exp_evaluations"
    os.makedirs(output_dir, exist_ok=True)

    # Load explanations
    lime_local = load_local_explanations("../results/fl", "lime")
    lime_global = load_explanations("../results/fl/global/explanations", "lime")
    lime_centralized = load_explanations("../results/ml/explanations", "lime")

    # Evaluate LIME results
    local_lime_consistency = evaluate_local_consistency(lime_local, explanation_type="lime", top_k=3)
    global_vs_local_lime_consistency = evaluate_global_vs_local(lime_global, lime_local)
    global_vs_centralized_lime_consistency = evaluate_global_vs_centralized(lime_global, lime_centralized)

    # Load SHAP explanations
    shap_local = load_local_explanations("../results/fl", "shap")
    shap_global = load_explanations("../results/fl/global/explanations", "shap")
    shap_centralized = load_explanations("../results/ml/explanations", "shap")

    # Load SHAP explanations
    local_shap_consistency = evaluate_local_consistency(shap_local, explanation_type="shap", top_k=3)
    global_vs_local_shap_consistency = evaluate_shap_global_vs_local(shap_global, shap_local)
    global_vs_centralized_shap_consistency = evaluate_shap_global_vs_centralized(shap_global, shap_centralized)

    consistencies_data = {
        "LIME": {
            "Local": local_lime_consistency["Spearman Avg"],
            "Global vs Local": global_vs_local_lime_consistency,
            "Global vs Centralized": global_vs_centralized_lime_consistency,
        },
        "SHAP": {
            "Local": local_shap_consistency["Spearman Avg"],
            "Global vs Local": global_vs_local_shap_consistency,
            "Global vs Centralized": global_vs_centralized_shap_consistency,
        }
    }

    visualize_multi_bar(
        consistencies_data,
        "Consistency Comparison: LIME vs SHAP",
        "Consistency Type",
        "Spearman Correlation",
        f"{output_dir}/consistencies_comparison.png"
    )

    lime_privacy = evaluate_privacy(lime_local, ["Age", "Gender"], "lime")
    visualize_results(lime_privacy, "Privacy Dominance LIME", "Client", "Importance", f"{output_dir}/lime_privacy_dominance.png")
    shap_privacy = evaluate_privacy(shap_local, ["Age", "Gender"], "shap")
    visualize_results(shap_privacy, "Privacy Dominance SHAP", "Client", "Importance", f"{output_dir}/shap_privacy_dominance.png")
   
    lime_correlations = evaluate_local_consistency_correlations(lime_local, explanation_type="lime")
    shap_correlations = evaluate_local_consistency_correlations(shap_local, explanation_type="shap")
    visualize_results(lime_correlations, "Pairwise LIME Consistancies Local", "Client", "Importance", f"{output_dir}/lime_pairwise_comparison.png")
    visualize_results(shap_correlations, "Pairwise SHAP Consistancies Local", "Client", "Importance", f"{output_dir}/shap_pairwise_comparison.png")

    base_path = "../results/fl"
    clients_range = range(10)
    global_path = "../results/fl/global/overheads"
    centralized_path = "../results/ml/overheads"

    data = load_overhead_data(base_path, clients_range, global_path, centralized_path)
    df = organize_data(data)
    plot_overheads(df)

# Run the script
if __name__ == "__main__":
    main()
    
