import os
import numpy as np
import json
import lime
from lime.lime_tabular import LimeTabularExplainer
import shap
import pandas as pd
from expflpro.task import recommend_for_all
import sys

def generate_lime_explanations(user_id, top_workouts, model, user_item_matrix, K=2):
    """
    Generate LIME explanations for an individual user's top recommended workouts.

    Parameters:
    - user_id (int/str): The user ID to generate explanations for.
    - top_workouts (list): List of top recommended workouts [(workout, score)].
    - model: Trained collaborative filtering model (Surprise SVD).
    - user_item_matrix (DataFrame): User-item matrix with user preferences for features.
    - K (int): Number of top features to use in the explanation.

    Returns:
    - explanations (list): List of dictionaries containing explanations for each recommendation.
    """

    # Extract feature names from the user-item matrix for LIME
    feature_names = list(user_item_matrix.columns)

    # Use the full user-item matrix as the training data for LIME explainer
    lime_data = user_item_matrix.values

    # Initialize the LIME explainer for regression tasks
    explainer = LimeTabularExplainer(
        training_data=lime_data,
        feature_names=feature_names,
        mode='regression'
    )

    # Initialize a list to store explanations
    explanations = []

    # Get the feature vector for the specific user
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)

    # Define a custom prediction function for LIME to predict user preferences
    def predict_fn(x):
        predictions = []
        for instance in x:  # For each input sample
            pred_scores = [
                model.predict(user_id, feature_names[i]).est for i in range(len(feature_names))
            ]
            predictions.append(pred_scores)
        return np.array(predictions)

    # Generate LIME explanations for each recommended workout
    for workout, score in top_workouts:
        # LIME explanation for the current recommendation
        exp = explainer.explain_instance(
            data_row=user_vector[0],  # Feature vector of the user
            predict_fn=predict_fn,   # Custom prediction function
            num_features=K           # Limit to top K features
        )
        explanations.append({
            "workout_type": workout,
            "explanation": exp.as_list()  # List of feature contributions
        })

    return explanations

def generate_lime_explanations_for_all(model, explainset, user_item_matrix, top_n=2):
    """
    Generate and save LIME explanations for all users' recommendations.

    Parameters:
    - model: Trained collaborative filtering model.
    - explainset (DataFrame): Dataset containing user and workout relationships.
    - user_item_matrix (DataFrame): User-item matrix for feature-based recommendations.
    - top_n (int): Number of top recommendations to explain.

    Returns:
    - lime_explanations (dict): Dictionary with explanations for all users.
    """

    # Initialize a dictionary to store explanations for all users
    lime_explanations = {}

    # Map workout IDs to labels (or indices)
    workout_mappings = [0, 1, 2, 3, 4, 5]

    # Generate recommendations for all users
    recommendations = recommend_for_all(model, explainset, workout_mappings, K=top_n)

    # Generate LIME explanations for each user
    for user_id, top_workouts in recommendations.items():
        lime_explanations[user_id] = generate_lime_explanations(user_id, top_workouts, model, user_item_matrix, K=top_n)

    return lime_explanations

def generate_shap_explanations(user_id, top_workouts, model, user_item_matrix):
    """
    Generate SHAP explanations for an individual user's top recommended workouts.

    Parameters:
    - user_id (int/str): The user ID to generate explanations for.
    - top_workouts (list): Top recommendations for the user [(workout, score)].
    - model: Trained collaborative filtering model.
    - user_item_matrix (DataFrame): User-item matrix for recommendations.

    Returns:
    - explanations (list): List of dictionaries containing explanations for each recommendation.
    """

    # Initialize a list to store explanations
    explanations = []

    # Extract the user's feature vector
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)

    # Define a custom prediction function for SHAP
    def predict_fn(x):
        predictions = []
        for instance in x:  # For each input sample
            pred_scores = [
                model.predict(user_id, user_item_matrix.columns[i]).est for i in range(len(user_item_matrix.columns))
            ]
            predictions.append(pred_scores)
        return np.array(predictions)

    # Initialize the SHAP explainer with the custom prediction function
    explainer = shap.Explainer(predict_fn, user_item_matrix.values)

    # Compute SHAP values for the user's feature vector
    shap_values = explainer(user_vector)

    # Add explanations for each recommended workout
    for workout, score in top_workouts:
        explanations.append({
            "workout_type": workout,
            "shap_values": shap_values.values[0].tolist(),  # SHAP values for the user's features
            "feature_importance": shap_values.data[0].tolist()  # Corresponding feature values
        })

    return explanations

def generate_shap_explanations_for_all(model, explainset, user_item_matrix, top_n=2):
    """
    Generate and save SHAP explanations for all users' recommendations.

    Parameters:
    - model: Trained collaborative filtering model.
    - explainset (DataFrame): Dataset containing user and workout relationships.
    - user_item_matrix (DataFrame): User-item matrix for content similarity.
    - top_n (int): Number of top recommendations to explain.

    Returns:
    - shap_explanations (dict): Dictionary with SHAP explanations for all users.
    """

    # Initialize a dictionary to store SHAP explanations for all users
    shap_explanations = {}

    # Map workout IDs to labels (or indices)
    workout_mappings = [0, 1, 2, 3, 4, 5]

    # Generate recommendations for all users
    recommendations = recommend_for_all(model, explainset, workout_mappings, K=top_n)

    # Generate SHAP explanations for each user
    for user_id, top_workouts in recommendations.items():
        shap_explanations[user_id] = generate_shap_explanations(user_id, top_workouts, model, user_item_matrix)

    return shap_explanations
