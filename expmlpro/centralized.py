import pandas as pd
import numpy as np
import os
import json
import sys
from expflpro.dataset import prepare_data
from expflpro.model import load_model
from expflpro.results import reset_results_folder, save_evaluation_metrics
from expflpro.graph import generate_recommendation_graphs
from expflpro.task import generate_evaluation_metrics, generate_recommendations_with_explanations, recommend_exercise_plans_for_all

FINAL_MODEL_DIR = "../ml_model"
FINAL_MODEL_FILENAME = "final_model.h5"

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
    X_train, X_test, X_explain, y_train, y_test, y_explain, mappings = prepare_data(data)


    # Build and train model
    model = load_model(input_shape=X_train.shape[1], num_classes=y_train.nunique())
    # Train the model
    model.fit({'User_Features': X_train}, y_train, validation_data=({'User_Features': X_test}, y_test), epochs=20, batch_size=64)

    # Save the trained model
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    final_model_path = os.path.join(FINAL_MODEL_DIR, FINAL_MODEL_FILENAME)
    model.save(final_model_path)
    print(f"Final ML model saved to: {final_model_path}")

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
    
    # Personalized recommendations
    print("Generating personalized recommendations...")

    #---------------------------
    # Select a random user for recommendations
    random_index = np.random.choice(len(X_explain))
    random_user = X_explain.iloc[random_index:random_index+1]
    recommendations_with_explanations = generate_recommendations_with_explanations(X_explain, model, mappings, random_index, random_user, top_k, mode="ml")

    output_folder = "../results/ml/explanations"
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
        X_explain, model, mappings, 0, new_user_df, top_k=1, mode="ml"
    )
    print("Recommendations for the new user Using ML model:", recommendations_with_explanations)
    # Generate explanations
    generate_recommendation_graphs(recommendations_with_explanations, output_folder, user_index=0)

# Run the script
if __name__ == "__main__":
    main()
