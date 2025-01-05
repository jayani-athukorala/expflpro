"""expflpro: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from expflpro.dataset import load_data
from expflpro.model import load_model
from expflpro.task import generate_evaluation_metrics, recommend_exercise_plans
from expflpro.results import save_local_evaluations
import numpy as np

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, X_train, X_test, X_explain, y_train, y_test, y_explain,
        partition_id, epochs, batch_size, verbose
    ):
        self.model = model
        self.X_train, self.X_test, self.X_explain, self.y_train, self.y_test, self.y_explain = X_train, X_test, X_explain, y_train, y_test, y_explain
        self.partition_id = partition_id
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            {'User_Features': self.X_train},
            self.y_train,
            validation_data=({'User_Features': self.X_test}, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate({'User_Features': self.X_test}, self.y_test, verbose=0)

        # Get predicted probabilities
        y_proba = self.model.predict({'User_Features': self.X_test}, batch_size=64)
        available_classes = np.unique(self.y_test)
        y_proba_filtered = y_proba[:, available_classes]
        # Re-normalize to ensure rows sum to 1
        y_proba_filtered = y_proba_filtered / np.sum(y_proba_filtered, axis=1, keepdims=True)

        # Convert probabilities to predicted class labels
        y_pred = np.argmax(y_proba, axis=1)

        # Compute metrics
        metrics = generate_evaluation_metrics(self.y_test, y_pred, y_proba_filtered, loss)

        save_local_evaluations(metrics, config["current_round"], self.partition_id)
        return loss, len(self.X_test), metrics


def client_fn(context: Context):
   
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    # Load partition data
    X_train, X_test, X_explain, y_train, y_test, y_explain = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    
    # Load model and data
    net = load_model(input_shape=X_train.shape[1], num_classes=y_train.nunique())
    # Return Client instance
    return FlowerClient(
        net, X_train, X_test, X_explain, y_train, y_test, y_explain, 
        partition_id, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
