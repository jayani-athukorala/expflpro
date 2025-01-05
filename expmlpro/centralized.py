import pandas as pd
import numpy as np
import os
import json
import sys

# Importing data preprocessing, recommendation generation, and explanation functions from expflpro
from expflpro.dataset import prepare_data
from expflpro.model import load_model
from expflpro.explainer import(
    generate_lime_explanations, generate_lime_explanations_for_all, 
    generate_shap_explanations, generate_shap_explanations_for_all
 )
from expflpro.results import reset_results_folder, save_evaluation_metrics, save_list_as_csv, save_list_as_json
# from expflpro.result import save_evaluation_metrics, save_list_as_json, save_dict_as_csv, reset_results_folder
from expflpro.task import generate_evaluation_metrics, recommend_exercise_plans, recommend_exercise_plans_for_all


# Main script to orchestrate the centralized recommendation system
def main():
    """
    Centralized pipeline for building, training, and evaluating a recommendation system.
    This script includes data preprocessing, model training, evaluation, recommendation generation,
    and explainability using LIME and SHAP.
    """
    # Reset results folder to start with a clean slate
    print("Resetting results folder...")
    reset_results_folder("../results/ml")

    # Load and prepare the dataset
    print("Loading dataset...")
    data = pd.read_csv('../data/exercise_recommendation_dataset.csv')  # Load the raw dataset
    
    print("Prepare data...")
    X_train, X_test, X_explain, y_train, y_test, y_explain = prepare_data(data)


    # Build and train model
    model = load_model(input_shape=X_train.shape[1], num_classes=y_train.nunique())
    # Train the model
    model.fit({'User_Features': X_train}, y_train, validation_data=({'User_Features': X_test}, y_test), epochs=20, batch_size=64)

    # Evaluate the model
    loss, accuracy = model.evaluate({'User_Features': X_test}, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Get predicted probabilities
    y_proba = model.predict({'User_Features': X_test}, batch_size=64)

    # Convert probabilities to predicted class labels
    y_pred = np.argmax(y_proba, axis=1)

    # Compute metrics
    metrics = generate_evaluation_metrics(y_test, y_pred, y_proba=y_proba, loss=loss)
    print("Evaluation Metrics on Test Data:", json.dumps(metrics, indent=4))  # Print metrics in a readable format
    
    # Save evaluation metrics to a results folder
    print("Saving evaluation metrics...")
    save_evaluation_metrics(metrics)  # Save metrics as a JSON file in a centralized location

    top_k = 1
    # # Generate recommendations and explanations for all users in explain dataset
    # print("Generating recommendations for all users...")
    # recommendations_all = recommend_exercise_plans_for_all(model, X_explain, top_k)
    # # print("recommendations", recommendations_all)
    
    # save_list_as_csv(recommendations_all, "../results/ml/recommendations/", "workout_recommendations.csv")

    # print("Generating LIME explanations for all users...")
    # feature_names=X_explain.columns
    # lime_explanations_all = generate_lime_explanations_for_all(model, X_explain, feature_names, top_k)
    # # print("lime", lime_explanations_all)
    # save_list_as_csv(lime_explanations_all, "../results/ml/explanations/lime/", "lime_explanations.csv")
    
    # print("Generating SHAP explanations for all users...")
    # shap_explanations_all = generate_shap_explanations_for_all(model, X_explain, top_k)
    # # print("shap", shap_explanations_all)
    # save_list_as_csv(shap_explanations_all, "../results/ml/explanations/shap/", "shap_explanations.csv")

    # Personalized recommendations
    # Select a random user
    random_index = np.random.choice(len(X_explain))
    random_user = X_explain.iloc[random_index:random_index+1]
    # Select a random user and recommend exercise plans
    recommendations = recommend_exercise_plans(X_explain, model, top_k)
    print("Recommendations:", json.dumps(recommendations, indent=4))
    # Save SHAP explanations
    save_list_as_json(recommendations, "../results/ml/recommendations", f"recommendations_for_user_{random_index}.json")

    # Generate SHAP explanations
    shap_explanations = generate_shap_explanations(model, X_train, random_user)
    print("SHAP Explanations:", json.dumps(shap_explanations, indent=4))
    # Save SHAP explanations
    save_list_as_json(shap_explanations, "../results/ml/explanations/shap", f"explanations_for_user_{random_index}_recommendations.json")

    # Generate LIME explanations
    lime_explanations = generate_lime_explanations(model, X_train, random_user)
    print("LIME Explanations:", json.dumps(lime_explanations, indent=4))
    # Save SHAP explanations
    save_list_as_json(lime_explanations, "../results/ml/explanations/lime", f"explanations_for_user_{random_index}_recommendations.json")


# Run the script
if __name__ == "__main__":
    main()
