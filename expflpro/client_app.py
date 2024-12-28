"""expflpro: A Flower / sklearn app."""

import warnings
import sys
import surprise
from sklearn.metrics import log_loss
from surprise import accuracy
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from surprise import Dataset, Reader
import numpy as np
from expflpro.task import (
    load_model,
    train, evaluate,
    calculate_evaluation_metrics,
    recommend, recommend_for_all,
    # evaluate_recommendations,
    set_model_params, get_model_params, set_initial_params,
    # generate_ranking_metrics
)
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

class FlowerClient(NumPyClient):
    def __init__(self, model, trainset, testset, explainset, trainset_len, mappings, user_item_matrix, user_features, partition_id):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.explainset = explainset
        self.trainset_len = trainset_len
        self.mappings = mappings
        self.user_item_matrix = user_item_matrix
        self.user_features = user_features
        self.partition_id = partition_id

    def fit(self, parameters, config):
        # Set weights 
        set_model_params(self.model, parameters)
        
        # Train and validate the SVD model
        model = train(self.model, self.trainset)
        
        # Return model weights
        return get_model_params(self.model), self.trainset_len, {}


    def evaluate(self, parameters, config):
        """
        Evaluate the model on local test data and return evaluation metrics.
        """

        self.model.trainset = self.trainset
        set_model_params(self.model, parameters)
        
        
        metrics = evaluate(self.model, self.testset, self.user_item_matrix)

        # Generate recommendations for individual users
        # Select a random user ID from explainset['user_id']
        random_user_id = np.random.choice(self.explainset['user_id'].unique())
        # GEenerate recommendations
        recommended_workouts = recommend(random_user_id, self.model, self.mappings['workout_type'])
        print(f"Top recommendations for random user {random_user_id}:", recommended_workouts)
        save_list_as_json(recommended_workouts, "results/fl/recommendations/local/", f"recommended_workouts_for{random_user_id}.json", self.partition_id)

        lime_explanations = generate_lime_explanations(random_user_id, recommended_workouts, self.model, self.user_item_matrix)
        print(f"LIME explanation for user {random_user_id}:", lime_explanations)
        save_list_as_json(lime_explanations, "results/fl/explanations/local/lime", f"explanations_for_user_{random_user_id}_workouts.json", self.partition_id)

        shap_explanations = generate_shap_explanations(random_user_id, recommended_workouts, self.model, self.user_item_matrix)
        print(f"SHAP explanation for user {random_user_id}:", shap_explanations)
        save_list_as_json(shap_explanations, "results/fl/explanations/local/shap", f"explanations_for_user_{random_user_id}_workouts.json", self.partition_id)

        # recommendations = recommend_for_all(self.model, self.explainset, self.mappings['workout_type'])
        
        # lime_explanations_all = generate_lime_explanations_for_all(self.model, self.explainset, self.user_item_matrix)
        # # print("LIME values", lime_explanations)
        # shap_explanations_all = generate_shap_explanations_for_all(self.model, self.explainset, self.user_item_matrix, top_n=2)
                
        rmse = metrics["RMSE"]
        
        return rmse, len(self.explainset), metrics



def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    partition_data = load_data(partition_id, num_partitions)
    processed_data, mappings = preprocess_data(partition_data)
    partition_interaction_data = generate_interactions_data(processed_data)
    user_item_matrix = build_user_item_matrix(partition_interaction_data)
    # Generate user features and workout features
    user_features, worout_features = get_features(processed_data)
    trainset, testset, explainset, trainset_len = prepare_data(partition_interaction_data)
    
    
    # Create SVD Model
    model = load_model()

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)
    
    return FlowerClient(
        model, trainset, testset, explainset, trainset_len,
        mappings,
        user_item_matrix,
        user_features,
        partition_id
        ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
