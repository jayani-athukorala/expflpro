import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

# Function to generate and save SHAP explanation graphs
def plot_shap_explanations(shap_json, output_folder, user_index):
    """
    Generate and save SHAP explanation plots from the SHAP JSON file.

    Args:
        json1_path (str): Path to the SHAP explanation JSON file.
        output_folder (str): Path to the folder where plots will be saved.

    Returns:
        None
    """
   
    shap_values = np.array(shap_json["shap_values"][0])  # Select the first instance's SHAP values
    feature_values = shap_json["feature_values"]
    feature_names = list(feature_values.keys())
    
    # Check for shape mismatch
    if len(shap_values) != len(feature_names):
        print("Warning: Shape mismatch detected. Truncating or padding values.")
        shap_values = shap_values[:len(feature_names)]

    # Force plot visualization (bar plot representation)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, shap_values, color="skyblue", alpha=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"SHAP Explanation for User {user_index}")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"shap_explanation_user_{user_index}.png"))
    plt.close()
    print(f"SHAP explanations plot saved to {output_folder}")

# Function to generate and save LIME explanation graphs
def plot_lime_explanations(lime_json, output_folder, user_index):
    # Extract data
    feature_importances = lime_json['feature_importances']
    feature_values = lime_json['feature_values']
    
    features = [item['feature'] for item in feature_importances]
    importances = [item['importance'] for item in feature_importances]
    
    # Create a plot for LIME feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title(f"LIME Explanation for User {user_index}")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"lime_explanation_user_{user_index}.png"))
    plt.close()

# Function to generate personalized recommendation graphs
def generate_recommendation_graphs(results, output_folder, user_index):
    shap_json = results["shap"]
    lime_json = results["lime"]
    
    # Generate SHAP and LIME graphs
    plot_shap_explanations(shap_json, output_folder, user_index)
    plot_lime_explanations(lime_json, output_folder, user_index)

    print(f"Explanation graphs saved in {output_folder} for User {user_index}")


# ------------Lime explanation graphs--------------------------------------------------
# Visualize features contribute to the prediction as a proportion
def plot_lime_bar_chart(lime_json, output_folder):
    feature_importances = lime_json["feature_importances"]
    features = [imp["feature"] for imp in feature_importances]
    importance_values = [imp["importance"] for imp in feature_importances]

    plt.figure(figsize=(10, 6))
    plt.barh(features, importance_values, color="salmon", alpha=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("LIME Importance")
    plt.ylabel("Feature")
    plt.title("LIME Feature Contributions")
    plt.tight_layout()

    plot_path = os.path.join(output_folder, "lime_bar_chart.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"LIME bar chart saved to {plot_path}")

def plot_lime_pos_neg_contributions(lime_json, output_folder):
    """
    Generate a horizontal bar chart for LIME feature contributions with positive and negative contributions separated.

    Args:
        lime_json (dict): LIME explanation JSON containing feature_importances and intercept.
        output_folder (str): Path to the folder where plots will be saved.

    Returns:
        None
    """
    feature_importances = lime_json["feature_importances"]
    features = [imp["feature"] for imp in feature_importances]
    importance_values = [imp["importance"] for imp in feature_importances]

    # Horizontal bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, importance_values, color=["salmon" if v < 0 else "skyblue" for v in importance_values])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("LIME Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("LIME Explanation: Positive/Negative Contributions", fontsize=14)
    plt.tight_layout()

    # Annotate the bars
    for bar in bars:
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.3f}',
            ha='left' if bar.get_width() < 0 else 'right',
            va='center',
            color='black'
        )

    # Save the plot
    bar_chart_path = os.path.join(output_folder, "lime_pos_neg_contributions.png")
    plt.savefig(bar_chart_path)
    plt.close()
    print(f"LIME positive/negative contributions chart saved to {bar_chart_path}")

# visualization of how features contribute to the prediction as a proportion
def plot_lime_pie_chart(lime_json, output_folder):
    """
    Generate a pie chart for LIME feature contributions.

    Args:
        lime_json (dict): LIME explanation JSON containing feature_importances and intercept.
        output_folder (str): Path to the folder where plots will be saved.

    Returns:
        None
    """
    feature_importances = lime_json["feature_importances"]
    features = [imp["feature"] for imp in feature_importances]
    importance_values = [abs(imp["importance"]) for imp in feature_importances]  # Use absolute values

    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(importance_values, labels=features, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("LIME Feature Contribution Proportions", fontsize=14)
    plt.tight_layout()

    # Save the plot
    pie_chart_path = os.path.join(output_folder, "lime_pie_chart.png")
    plt.savefig(pie_chart_path)
    plt.close()
    print(f"LIME pie chart saved to {pie_chart_path}")


def plot_lime_trend(feature_importances_list, output_folder):
    """
    Generate a line plot showing feature importance trends across multiple instances.

    Args:
        feature_importances_list (list): List of LIME explanation JSONs for different instances.
        output_folder (str): Path to the folder where plots will be saved.

    Returns:
        None
    """
    # Extract features and their importance across instances
    all_features = {imp["feature"] for exp in feature_importances_list for imp in exp["feature_importances"]}
    feature_trends = {feature: [] for feature in all_features}

    for lime_json in feature_importances_list:
        for feature in feature_trends:
            importance = next((imp["importance"] for imp in lime_json["feature_importances"] if imp["feature"] == feature), 0)
            feature_trends[feature].append(importance)

    # Plot each feature's trend
    plt.figure(figsize=(12, 8))
    for feature, values in feature_trends.items():
        plt.plot(values, label=feature)

    plt.xlabel("Instances", fontsize=12)
    plt.ylabel("Importance", fontsize=12)
    plt.title("LIME Explanation: Feature Importance Trends", fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    # Save the plot
    trend_plot_path = os.path.join(output_folder, "lime_feature_trends.png")
    plt.savefig(trend_plot_path)
    plt.close()
    print(f"LIME feature importance trend plot saved to {trend_plot_path}")

# Cumulative contributions of top-N features.
def plot_lime_cumulative_contributions(lime_json, output_folder):
    """
    Generate a bar chart showing cumulative contributions of top-N features.

    Args:
        lime_json (dict): LIME explanation JSON containing feature_importances and intercept.
        output_folder (str): Path to the folder where plots will be saved.

    Returns:
        None
    """
    feature_importances = lime_json["feature_importances"]
    sorted_features = sorted(feature_importances, key=lambda x: abs(x["importance"]), reverse=True)
    
    features = [imp["feature"] for imp in sorted_features]
    importance_values = [imp["importance"] for imp in sorted_features]

    # Cumulative sum
    cumulative_importance = [sum(importance_values[:i+1]) for i in range(len(importance_values))]

    # Bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(features, cumulative_importance, color="orchid", alpha=0.8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Cumulative Contribution", fontsize=12)
    plt.title("LIME Explanation: Cumulative Contributions", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    cumulative_plot_path = os.path.join(output_folder, "lime_cumulative_contributions.png")
    plt.savefig(cumulative_plot_path)
    plt.close()
    print(f"LIME cumulative contributions plot saved to {cumulative_plot_path}")

#--------------------------------Generate shap explanations-------------   
def plot_shap_bar_chart(shap_json, output_folder, target_class=0):
    shap_values = [class_data["values"] for class_data in shap_json["shap_values"] if class_data["class"] == target_class][0]
    feature_values = shap_json["feature_values"]
    feature_names = list(feature_values.keys())

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, shap_values, color="skyblue", alpha=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("SHAP Value")
    plt.ylabel("Feature")
    plt.title(f"SHAP Feature Contributions for Class {target_class}")
    plt.tight_layout()

    plot_path = os.path.join(output_folder, f"shap_bar_chart_class_{target_class}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"SHAP bar chart saved to {plot_path}")

def plot_shap_waterfall(shap_json, output_folder, target_class=0):
    shap_values = [class_data["values"] for class_data in shap_json["shap_values"] if class_data["class"] == target_class][0]
    feature_values = shap_json["feature_values"]
    feature_names = list(feature_values.keys())

    shap.Explanation(
        values=shap_values,
        base_values=shap_json["expected_value"],
        data=feature_names
    ).waterfall_plot()

    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"shap_waterfall_class_{target_class}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"SHAP waterfall plot saved to {plot_path}")