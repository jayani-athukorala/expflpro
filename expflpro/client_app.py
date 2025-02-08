"""expflpro: A Flower / TensorFlow app with Differential Privacy."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from expflpro.dataset import load_data
from expflpro.model import load_model
from expflpro.task import generate_evaluation_metrics, generate_recommendations_with_explanations
from expflpro.results import save_local_evaluations
from expflpro.graph import generate_recommendation_graphs
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

# Define Flower Client with DP
class FlowerClient(NumPyClient):
    def __init__(
        self, model, X_train, X_test, X_explain, y_train, y_test, y_explain, mappings,
        partition_id, num_server_rounds, epochs, batch_size, verbose,
        dp_epsilon=3.0, dp_delta=1e-5, dp_clip_norm=1.0
    ):
        self.model = model
        self.X_train, self.X_test, self.X_explain, self.y_train, self.y_test, self.y_explain, self.mappings = (
            X_train, X_test, X_explain, y_train, y_test, y_explain, mappings
        )
        self.partition_id = partition_id
        self.num_server_rounds = num_server_rounds
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_clip_norm = dp_clip_norm  # Gradient clipping norm for DP

        # Apply Differentially Private Optimizer
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=DPKerasSGDOptimizer(
                l2_norm_clip=self.dp_clip_norm,  # Clipping threshold
                noise_multiplier=self.dp_epsilon,  # Noise added for privacy
                num_microbatches=batch_size,  # DP is applied per microbatch
                learning_rate=0.01
            ),
            metrics=["accuracy"]
        )

    def fit(self, parameters, config):
        """Train the model with DP-SGD."""
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

        # At the final round after the model evaluations, generate local recommendations with explanations
        if config["current_round"] == self.num_server_rounds:
            recommendations, user_index = self.generate_local_recommendations_with_explanations()
            output_folder = f"results/fl/client_{self.partition_id}/explanations"

            generate_recommendation_graphs(recommendations, output_folder, user_index)

            recommendations, new_user_index = self.generate_local_recommendations_for_new_user()

            generate_recommendation_graphs(recommendations, output_folder, new_user_index)

        return loss, len(self.X_test), metrics

    def generate_local_recommendations_with_explanations(self):
        top_k = 1
        random_index = np.random.choice(len(self.X_explain))
        random_user = self.X_explain.iloc[random_index:random_index+1]

        results = generate_recommendations_with_explanations(
            X_explain=self.X_explain, 
            model=self.model, 
            mappings=self.mappings,
            top_k=top_k,
            user_index=random_index,
            user_features=random_user, 
            mode="fl", 
            partition_id=self.partition_id
        )

        return results, random_index

    def generate_local_recommendations_for_new_user(self):
        top_k = 1
        # Recommendations for a new user
        new_user = np.array([[70, 1.75, 24.5, 35, 1]]) 
        new_user_index = 0
        # Convert to a DataFrame
        new_user_df = pd.DataFrame(new_user, columns=["Weight", "Height", "BMI", "Age", "Gender"])
        # Generate personalized recommendations using FL global model   
        recommendations_with_explanations = generate_recommendations_with_explanations(
            X_explain=self.X_explain, 
            model=self.model, 
            mappings=self.mappings,
            top_k=top_k,
            user_index=new_user_index,
            user_features=new_user_df, 
            mode="fl", 
            partition_id=self.partition_id
        )

        return recommendations_with_explanations, new_user_index


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_server_rounds = context.run_config["num-server-rounds"]

    # Load partition data
    X_train, X_test, X_explain, y_train, y_test, y_explain, mappings = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Load model and data
    net = load_model(input_shape=X_train.shape[1], num_classes=y_train.nunique())

    # Return Client instance with DP parameters
    return FlowerClient(
        net, X_train, X_test, X_explain, y_train, y_test, y_explain, mappings,
        partition_id, num_server_rounds, epochs, batch_size, verbose,
        dp_epsilon=3.0, dp_delta=1e-5, dp_clip_norm=1.0  # DP parameters
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
