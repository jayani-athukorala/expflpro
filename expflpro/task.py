"""expflpro: A Flower / sklearn app."""
import sys
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
from surprise import SVD, Dataset, Reader, accuracy
from collections import defaultdict
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from recommenders.utils.timer import Timer
from recommenders.evaluation.python_evaluation import (
    rmse,
    mae,
    rsquared,
    exp_var,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    get_top_k_items,
)
from recommenders.models.surprise.surprise_utils import (
    predict,
    compute_ranking_predictions,
)

# Function to load the Surprise SVD model
def load_model():
    """
    Load a new instance of the Surprise SVD model.

    Returns:
        model: An untrained Surprise SVD model.
    """
    model = SVD()
    return model

# Function to train the SVD model
def train(model, trainset): 
    """
    Train the Surprise SVD model using the provided training data.

    Parameters:
        model: An untrained Surprise SVD model.
        trainset: Surprise training dataset containing user-item interactions.

    Returns:
        model: The trained SVD model.
    """
    # Measure training time
    with Timer() as train_time:
        model.fit(trainset)

    print(f"Took {train_time.interval} seconds for training.")
    return model

# Function to evaluate the SVD model
def evaluate(model, testset, user_item_matrix):
    """
    Evaluate the trained SVD model on a test dataset.

    Parameters:
        model: The trained Surprise SVD model.
        testset: Surprise test dataset with user-item interactions.
        user_item_matrix: DataFrame mapping users to items and ratings.

    Returns:
        metrics: A dictionary containing various evaluation metrics.
    """
    # Measure testing time
    with Timer() as test_time:
        predictions = model.test(testset)

    print(f"Took {test_time.interval} seconds for prediction.")
    # Calculate evaluation metrics
    metrics = calculate_evaluation_metrics(predictions, user_item_matrix, K=2, relevance_threshold=4.0)
    return metrics

# Function to recommend workouts for a specific user
def recommend(user_id, model, workout_mappings, K=2):
    """
    Recommend top-K workouts for a single user.

    Parameters:
        user_id: ID of the user for whom recommendations are generated.
        model: The trained Surprise SVD model.
        workout_mappings: List of all possible workout types.
        K: Number of top recommendations to generate.

    Returns:
        top_k_workouts: List of top-K workouts and their predicted scores.
    """
    # Define workout types (excluding unknowns if applicable)
    workout_types = [0, 1, 2, 3, 4, 5]

    # Predict ratings for unrated workout types
    predicted_ratings = []
    for workout in workout_types:
        # Predict the score for each workout
        prediction = model.predict(user_id, workout).est
        predicted_ratings.append((workout, prediction))

    # Sort by predicted scores in descending order and select top-K
    top_k_workouts = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:K]
    return top_k_workouts

# Function to recommend workouts for all users
def recommend_for_all(model, explainset, workout_mappings, K=2):
    """
    Recommend top-K workouts for each user in the dataset.

    Parameters:
        model: The trained Surprise SVD model.
        explainset: DataFrame with columns ['user_id', 'workout_type', 'rating'].
        workout_mappings: List of all possible workout types.
        K: Number of top recommendations to generate.

    Returns:
        recommendations: A dictionary where keys are user IDs and values are lists 
                         of top-K recommended workouts with their scores.
    """
    recommendations = {}
    user_ids = explainset['user_id'].unique()

    for user_id in user_ids:
        # Recommend workouts for the current user
        recommendations[user_id] = recommend(user_id, model, workout_mappings)
    return recommendations

# Function to calculate various evaluation metrics
def calculate_evaluation_metrics(predictions, user_item_matrix, K=5, relevance_threshold=4.0):
    """
    Evaluate the recommendation system using multiple metrics.

    Parameters:
        predictions: List of predictions from Surprise's model.test().
        user_item_matrix: DataFrame mapping users to items and ratings.
        K: Number of top recommendations to consider.
        relevance_threshold: Minimum rating to consider an item as relevant.

    Returns:
        metrics: A dictionary containing RMSE, MAE, Precision@K, Recall@K, F1-Score@K,
                 Hit Ratio, Coverage, and NDCG@K.
    """
    # Calculate RMSE and MAE
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    # Group predictions by user
    user_predictions = defaultdict(list)
    for pred in predictions:
        user_predictions[pred.uid].append((pred.iid, pred.est, pred.r_ui))
    
    # Sort predictions by estimated rating for each user
    for user, preds in user_predictions.items():
        preds.sort(key=lambda x: x[1], reverse=True)
        user_predictions[user] = preds

    # Initialize metrics
    precision_sum, recall_sum, f1_sum, ndcg_sum = 0, 0, 0, 0
    hit_count = 0
    recommended_items = set()
    total_users = len(user_predictions)
    all_items = set(user_item_matrix.columns)

    # Calculate metrics for each user
    for user, preds in user_predictions.items():
        top_k_items = [iid for iid, _, _ in preds[:K]]
        actual_items = set(user_item_matrix.loc[user][user_item_matrix.loc[user] >= relevance_threshold].index)

        recommended_items.update(top_k_items)

        hits = len(set(top_k_items) & actual_items)
        precision = hits / K
        recall = hits / len(actual_items) if actual_items else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if hits > 0:
            hit_count += 1

        dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(top_k_items) if item in actual_items])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual_items), K))])
        ndcg = dcg / idcg if idcg > 0 else 0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        ndcg_sum += ndcg

    # Calculate overall metrics
    precision_at_k = precision_sum / total_users
    recall_at_k = recall_sum / total_users
    f1_at_k = f1_sum / total_users
    hit_ratio = hit_count / total_users
    coverage = len(recommended_items) / len(all_items)
    ndcg_at_k = ndcg_sum / total_users

    # Compile metrics into a dictionary
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "Precision@K": precision_at_k,
        "Recall@K": recall_at_k,
        "F1-Score@K": f1_at_k,
        "Hit Ratio": hit_ratio,
        "Coverage": coverage,
        "NDCG@K": ndcg_at_k
    }
    return metrics

# Functions to extract, set, and initialize model parameters for federated learning

def get_model_params(model):
    """
    Extract parameters from the Surprise SVD model.

    Returns:
        A list containing the parameters of the SVD model.
    """
    return [model.pu, model.qi, model.bu, model.bi, model.global_mean]

def set_model_params(model, params):
    """
    Set parameters for the Surprise SVD model.

    Parameters:
        model: An untrained or partially trained Surprise SVD model.
        params: List of parameters to set in the model.

    Returns:
        The updated model with the provided parameters.
    """
    model.pu, model.qi, model.bu, model.bi, model.global_mean = params
    return model

def set_initial_params(model):
    """
    Initialize parameters for the Surprise SVD model.

    Returns:
        The initialized model.
    """
    n_users = 0  # Initialize with zero users (can be updated dynamically)
    n_items = 6  # Total number of items (workout types)
    n_factors = 5  # Number of latent factors

    model.pu = np.zeros((n_users, n_factors))  # User latent factors
    model.qi = np.zeros((n_items, n_factors))  # Item latent factors
    model.bu = np.zeros(n_users)  # User biases
    model.bi = np.zeros(n_items)  # Item biases
    model.global_mean = 0.0  # Global mean rating

    return model




