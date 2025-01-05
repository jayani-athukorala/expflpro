"""expflpro: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from expflpro.model import load_model
from expflpro.results import save_evaluation_results_round, reset_results_folder

current_round = 0

def weighted_average(metrics):
    """
    Federated averaging function for aggregating evaluation metrics across clients.

    Parameters:
    - metrics (list): List of tuples containing (num_examples, metrics_dict).

    Returns:
    - Aggregated metrics as a dictionary.
    """
    global current_round
    # print("byeeeee", server_round)
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
    # Create the configuration dictionary
    config = {
        "current_round": server_round,        
    }
    return config

def server_fn(context: Context):
    # Reset results folder to start with a clean slate
    print("Resetting results folder...")
    reset_results_folder("results/fl")
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model(5, 7).get_weights())

    # Define strategy
    strategy = strategy = FedAvg(
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
