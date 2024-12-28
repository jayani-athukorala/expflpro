import pandas as pd
import numpy as np
import os
import json
import sys

# Importing data preprocessing, recommendation generation, and explanation functions from expflpro
from expflpro.dataset import (
    preprocess_data, 
    generate_interactions_data, 
    prepare_collaborative_data, 
    build_user_item_matrix,
    get_features,
)
from expflpro.explainer import (
    generate_lime_explanations, generate_lime_explanations_for_all, 
    generate_shap_explanations, generate_shap_explanations_for_all,
)
from expflpro.result import save_evaluation_metrics, save_list_as_json
from expflpro.task import load_model, train, evaluate, calculate_evaluation_metrics, recommend, recommend_for_all


# Main script to orchestrate the centralized recommendation system
def main():
    """
    Centralized pipeline for building, training, and evaluating a recommendation system.
    This script includes data preprocessing, model training, evaluation, recommendation generation,
    and explainability using LIME and SHAP.
    """
    
    # Step 1: Load and preprocess the dataset
    print("Loading dataset...")
    data = pd.read_csv('../data/exercise_dataset.csv')  # Load the raw dataset
    
    print("Preprocessing data...")
    processed_data, mappings = preprocess_data(data)  # Preprocess raw data and create mappings
    
    # Step 2: Generate interaction data for collaborative filtering
    print("Generating user-workout interaction data...")
    interaction_data = generate_interactions_data(processed_data)  # User-workout interaction matrix
    
    print("Preparing collaborative filtering datasets...")
    trainset, testset, explainset = prepare_collaborative_data(interaction_data)  # Split data into train/test sets
    
    # Step 3: Build user-item matrix and extract features
    print("Generating user and workout features...")
    user_features, workout_features = get_features(processed_data)  # Extract features for users and workouts
    
    print("Building user-item matrix...")
    user_item_matrix = build_user_item_matrix(interaction_data)  # Matrix for recommendation generation

    # Step 4: Load, train, and evaluate the recommendation model
    print("Loading recommendation model...")
    model = load_model()  # Load a Surprise SVD model
    
    print("Training model...")
    model = train(model, trainset)  # Train the model with the training dataset

    print("Evaluating model...")
    metrics = evaluate(model, testset, user_item_matrix)  # Evaluate the model on the test dataset
    print("Evaluation Metrics on Test Data:", json.dumps(metrics, indent=4))  # Print metrics in a readable format
    
    # Save evaluation metrics to a results folder
    print("Saving evaluation metrics...")
    save_evaluation_metrics(metrics)  # Save metrics as a JSON file in a centralized location

    # Step 5: Generate recommendations for a random user
    print("Generating recommendations for a random user...")
    random_user_id = np.random.choice(explainset['user_id'].unique())  # Select a random user ID
    recommended_workouts = recommend(random_user_id, model, mappings['workout_type'])  # Recommend top workouts
    print(f"Top recommendations for random user {random_user_id}:", json.dumps(recommended_workouts, indent=4))
    
    # Save recommendations for the random user
    save_list_as_json(recommended_workouts, "../results/ml/recommendations/", f"recommended_workouts_for_{random_user_id}.json")

    # Step 6: Generate explainability using LIME for the random user
    print("Generating LIME explanations for recommendations...")
    lime_explanations = generate_lime_explanations(random_user_id, recommended_workouts, model, user_item_matrix)
    print(f"LIME explanation for user {random_user_id}:", json.dumps(lime_explanations, indent=4))
    
    # Save LIME explanations
    save_list_as_json(lime_explanations, "../results/ml/explanations/lime", f"explanations_for_user_{random_user_id}_workouts.json")

    # Step 7: Generate explainability using SHAP for the random user
    print("Generating SHAP explanations for recommendations...")
    shap_explanations = generate_shap_explanations(random_user_id, recommended_workouts, model, user_item_matrix)
    print(f"SHAP explanation for user {random_user_id}:", json.dumps(shap_explanations, indent=4))
    
    # Save SHAP explanations
    save_list_as_json(shap_explanations, "../results/ml/explanations/shap", f"explanations_for_user_{random_user_id}_workouts.json")

    # Step 8: Generate recommendations and explanations for all users
    print("Generating recommendations for all users...")
    recommendations_all = recommend_for_all(model, explainset, mappings['workout_type'])
    save_list_as_json(recommendations_all, "../results/ml/recommendations/", "workout_recommendations.json")

    print("Generating LIME explanations for all users...")
    lime_explanations_all = generate_lime_explanations_for_all(model, explainset, user_item_matrix)
    save_list_as_json(lime_explanations_all, "../results/ml/explanations/lime/", "lime_explanations.json")
    
    print("Generating SHAP explanations for all users...")
    shap_explanations_all = generate_shap_explanations_for_all(model, explainset, user_item_matrix, top_n=2)
    save_list_as_json(shap_explanations_all, "../results/ml/explanations/shap/", "shap_explanations.json")


# Run the script
if __name__ == "__main__":
    main()
