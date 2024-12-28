"""Flower Server for Federated Hybrid Recommendation System."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from expflpro.task import load_model, get_model_params, set_initial_params
from expflpro.dataset import load_data
from expflpro.result import save_evaluation_results_round, reset_results_folder
import sys
from expflpro.dataset import (
    load_data, 
    preprocess_data,
    generate_interactions_data,
    build_user_item_matrix,
    prepare_data,
    get_features
)

from expflpro.explainer import (

    generate_lime_explanations,
    generate_lime_explanations_for_all,
    generate_shap_explanations,
    generate_shap_explanations_for_all
)

from expflpro.result import save_list_as_json
# Global round counter
round_counter = 0

def weighted_average(metrics):
    """
    Weighted averaging function for federated metrics aggregation.

    Parameters:
    - metrics: List of tuples containing (num_examples, metrics_dict).

    Returns:
    - Aggregated metrics as a dictionary.
    """
    precision_at_k = [num_examples * m["Precision@K"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated_metrics = {
        key: sum(num_examples * m[key] for num_examples, m in metrics) / total_examples
        for key in metrics[0][1] if key != "RMSE"
    }
    aggregated_metrics["RMSE"] = sum(num_examples * m["RMSE"] for num_examples, m in metrics) / total_examples

    # Save aggregated metrics to JSON
    global round_counter
    round_counter += 1
    save_evaluation_results_round(aggregated_metrics, round_counter)

    return {"Precision@K": sum(precision_at_k) / total_examples}


def server_fn(context: Context):

    # Reset the results folder at the beginning of the run
    reset_results_folder("results/fl")
    partition_id=0  # Use partition 0 for the global explain dataset
    num_partitions=1
    # Load the global explainer dataset
    partition_data = load_data(partition_id, num_partitions)
    processed_data, mappings = preprocess_data(partition_data)
    partition_interaction_data = generate_interactions_data(processed_data)
    user_item_matrix = build_user_item_matrix(partition_interaction_data)
    # Generate user features and workout features
    # user_features, worout_features = get_features(processed_data)
    _, _, explainset, _ = prepare_data(partition_interaction_data)


    # Initialize the global model
    model = load_model()  # SVD
    set_initial_params(model)

    # Convert initial parameters to Flower format
    params = ndarrays_to_parameters(get_model_params(model))

    # Define federated averaging strategy
    strategy = FedAvg(
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Configure server rounds
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    current_round = context.run_id
    print(current_round)
    # Define a callback for end of training
    if(current_round == num_rounds):
        """
        Callback to execute at the end of all training rounds.
        """
        print("Generating global explanations...")
        
        random_user_id = np.random.choice(explainset['user_id'].unique())
        # GEenerate recommendations
        recommended_workouts = recommend(random_user_id, model, mappings['workout_type'])
        print(f"Top recommendations for random user {random_user_id}:", recommended_workouts)
        save_list_as_json(recommended_workouts, "results/fl/recommendations/global/", f"recommended_workouts_for{random_user_id}.json")

        lime_explanations = generate_lime_explanations(random_user_id, recommended_workouts, model, user_item_matrix)
        print(f"LIME explanation for user {random_user_id}:", lime_explanations)
        save_list_as_json(lime_explanations, "results/fl/explanations/global/lime", f"explanations_for_user_{random_user_id}_workouts.json")

        shap_explanations = generate_shap_explanations(random_user_id, recommended_workouts, model, user_item_matrix)
        print(f"SHAP explanation for user {random_user_id}:", shap_explanations)
        save_list_as_json(shap_explanations, "results/fl/explanations/global/shap", f"explanations_for_user_{random_user_id}_workouts.json")

    return ServerAppComponents(strategy=strategy, config=config)

# Create Flower ServerApp
app = ServerApp(server_fn=server_fn)
