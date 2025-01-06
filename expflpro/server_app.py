from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import os
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common.logger import log
from logging import ERROR, INFO, WARN
from expflpro.model import load_model
from expflpro.dataset import load_data
from expflpro.results import reset_results_folder, save_evaluation_results_round
from expflpro.graph import generate_recommendation_graphs
from expflpro.task import generate_recommendations_with_explanations
# Directory to save the final model
FINAL_MODEL_DIR = "fl_model"

class FedAvgWithFinalModelSaving(FedAvg):
    """Custom FedAvg strategy that saves the final global model."""

    def __init__(self, num_rounds: int, save_path: str, *args, **kwargs):
        self.num_rounds=num_rounds
        self.save_path = Path(save_path)
        # Ensure directory exists
        self.save_path.mkdir(exist_ok=True, parents=True)
        super().__init__(*args, **kwargs)

    def save_final_global_model(self, parameters):
        """Save the final global model to disk."""
        ndarrays = parameters_to_ndarrays(parameters)
        data = {"global_parameters": ndarrays}
        filename = str(self.save_path / "final_model.pkl")
        with open(filename, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        log(INFO, f"Final model saved to: {filename}")

    def evaluate(self, server_round: int, parameters):
        """Evaluate and save final model at the end."""
        # Save the final model if this is the last round
        if server_round == self.num_rounds:
            self.save_final_global_model(parameters)
            partition_id = 0
            num_partitions = 1
           
            print("Prepare data...")
            _, _, X_explain, _, _, y_explain, mappings = load_data(partition_id, num_partitions)
 
            # Convert Parameters to weights
            ndarrays = parameters_to_ndarrays(parameters)
            # Load model and data
            model = load_model(5, 7)
            model.set_weights(ndarrays)
            if model is None:
                raise ValueError("Model could not be loaded correctly!")
            
            # Personalized recommendations using Global model
            print("Generating personalized recommendations...")
            top_k = 1
            #---------------------------
            # Select a random user for recommendations
            random_index = np.random.choice(len(X_explain))
            random_user = X_explain.iloc[random_index:random_index+1]

            recommendations_with_explanations = generate_recommendations_with_explanations(
                X_explain, model, mappings, random_index, random_user, top_k, mode="fl"
            )

            output_folder = "results/fl/global/explanations"
            os.makedirs(output_folder, exist_ok=True)

            user_index = random_index  # Random user index
            generate_recommendation_graphs(recommendations_with_explanations, output_folder, user_index)
            
            #-------------------------
            # Recommendations for new user
            new_user = np.array([[70, 1.75, 24.5, 35, 1]]) 
            # Convert to a DataFrame
            new_user_df = pd.DataFrame(new_user, columns=["Weight", "Height", "BMI", "Age", "Gender"])
            # Personalized recommendations using fl global model   
            recommendations_with_explanations = generate_recommendations_with_explanations(
                X_explain, model, mappings, 0, new_user_df, top_k=1, mode="fl"
            )
            print("Recommendations for the new user Using ML model:", recommendations_with_explanations)
            # Generate explanations
            generate_recommendation_graphs(recommendations_with_explanations, output_folder, user_index=0)

        # Call the parent evaluate method
        return super().evaluate(server_round, parameters)

current_round = 0

def weighted_average(metrics):
    """
    Federated averaging function for aggregating evaluation metrics across clients.
    """
    global current_round
    # Extract individual metric values and weight them by the number of examples
    accuracy = [num_examples * m["Accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)

    # Compute the aggregated metrics
    aggregated_metrics = {
        key: sum(num_examples * m[key] for num_examples, m in metrics) / total_examples
        for key in metrics[0][1] if key != "Loss"
    }
    aggregated_metrics["Loss"] = sum(num_examples * m["Loss"] for num_examples, m in metrics) / total_examples

    # Increment the global round counter
    current_round += 1

    # Save the aggregated metrics for the current round
    save_evaluation_results_round(aggregated_metrics, current_round)

    return {"Accuracy": sum(accuracy) / total_examples}

def evaluate_config(server_round: int):
    """Generate evaluation configuration for each round."""
    return {"current_round": server_round}

def server_fn(context: Context):
    # Reset results folder to start with a clean slate
    print("Resetting results folder...")
    reset_results_folder("results/fl")

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model(5, 7).get_weights())

    # Define custom strategy with checkpoint saving
    strategy = FedAvgWithFinalModelSaving(
        num_rounds=num_rounds,
        save_path=FINAL_MODEL_DIR,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)