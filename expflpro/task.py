"""expflpro: A Flower / TensorFlow app."""
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score
)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from expflpro.explainer import(
    generate_lime_explanations, generate_lime_explanations_for_all, 
    generate_shap_explanations, generate_shap_explanations_for_all
 )
from expflpro.results import reset_results_folder, save_evaluation_metrics, save_list_as_csv, save_list_as_json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def generate_evaluation_metrics(y_true, y_pred, y_proba=None, loss=None):
    """
    Calculate evaluation metrics for the model.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like): Predicted probabilities (for ROC AUC, log loss).
        loss (float): Loss value.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {}

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1score = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Advanced metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    cks = cohen_kappa_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
    logloss = log_loss(y_true, y_proba)

    # Compile metrics into a dictionary
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1score,
        "ROC-AUC": roc_auc,
        "Log-Loss": logloss,
        "Matthews-Correlation-Coefficient": mcc,
        "Cohen-Kappa-Score": cks,
        "Loss": loss,
    }

    return metrics


def recommend_exercise_plans(user_features, model, scaler, k=2):
    """
    Recommend top-K exercise plans for a random user in X_explain.

    Args:
        user_features : ----------.
        model (tf.keras.Model): Trained model.
        k (int): Number of top recommendations to return.

    Returns:
        dict: Recommendations as JSON.
        np.array: The selected user's features.
    """
    user_features = user_features.copy()
    user_features.loc[:, ['Weight', 'Height', 'BMI', 'Age']] = scaler.fit_transform(
        user_features[['Weight', 'Height', 'BMI', 'Age']]
    )

    # Predict probabilities
    probabilities = model.predict({'User_Features': user_features}, verbose=0)[0]

    # Get top-K recommendations
    top_k_indices = probabilities.argsort()[-k:][::-1]
    top_k_probs = probabilities[top_k_indices]

    # Prepare recommendations as JSON
    recommendations = [
        {"exercise_plan": int(idx), "probability": float(prob)}
        for idx, prob in zip(top_k_indices, top_k_probs)
    ]
    return {"recommendations": recommendations}


def recommend_exercise_plans_for_all(model, X_explain, k=1):
    """
    Provide top-K recommendations with probabilities for all users in X_explain.
    
    Args:
        X_explain (pd.DataFrame): The dataset to explain.
        model (tf.keras.Model): The trained model.
        k (int): The number of top recommendations to return.

    Returns:
        List[List[Tuple[int, float]]]: Top-K recommendations for all users.
    """
    predictions = model.predict({'User_Features': X_explain}, verbose=0)  # Get probabilities for each class
    top_k_recommendations = []

    for user_probs in predictions:
        # Get top K indices and probabilities
        top_k_indices = user_probs.argsort()[-k:][::-1]  # Indices of top K probabilities
        top_k_probs = user_probs[top_k_indices]          # Corresponding probabilities
        
        # Pair recommendations with their probabilities
        top_k_recommendations.append(
            [(int(idx), float(prob)) for idx, prob in zip(top_k_indices, top_k_probs)]
        )
    
    return top_k_recommendations

def generate_recommendations_with_explanations(X_explain, model, mappings, user_index, user_features, top_k, mode="fl", partition_id=None):
    """
    Generate recommendations and explanations for both ML and FL implementations.

    Parameters:
    - X_explain: The dataset used for generating explanations.
    - model: The trained recommendation model.
    - top_k: The number of top recommendations to generate.
    - mode (str): 'ml' for global, 'fl' for client-specific.
    - partition_id (int, optional): The client ID (required for 'fl' mode).

    Returns:
    - dict: A dictionary containing recommendations, SHAP explanations, and LIME explanations.
    """

    # Normalize numerical features
    scaler = MinMaxScaler()
    X_explain[['Weight', 'Height', 'BMI', 'Age']] = scaler.fit_transform(X_explain[['Weight', 'Height', 'BMI', 'Age']])

    # Generate recommendations
    recommendations = recommend_exercise_plans(user_features, model, scaler, top_k)

    # Define the base directory
    if mode == "fl":
        if partition_id is None:
            base_dir = "results/fl/global"
        else:
            base_dir = f"results/fl/client_{partition_id}"
    elif mode == "ml":
        base_dir = "../results/ml"
    else:
        raise ValueError("Invalid mode. Use 'ml' or 'fl'.")

    # Save recommendations
    save_list_as_json(
        recommendations, 
        os.path.join(base_dir, "recommendations"), 
        f"recommendations_for_user_{user_index}.json"
    )

    # Generate SHAP explanations
    shap_explanations = generate_shap_explanations(model, scaler, mappings, X_explain, user_features)
    save_list_as_json(
        shap_explanations, 
        os.path.join(base_dir, "explanations"), 
        f"shap_explanations_for_user_{user_index}_recommendations.json"
    )

    # Generate LIME explanations
    lime_explanations = generate_lime_explanations(model, scaler, mappings, X_explain, user_features)
    save_list_as_json(
        lime_explanations, 
        os.path.join(base_dir, "explanations"), 
        f"lime_explanations_for_user_{user_index}_recommendations.json"
    )

    # Return the generated data
    return {
        "recommendations": recommendations,
        "shap": shap_explanations,
        "lime": lime_explanations
    }

# This function will generate and save recommendations and indiviual explanations for all users in the explainset
def generate_personlalized_recommendations_for_all():
    
    # Generate recommendations and explanations for all users in explain dataset
    print("Generating recommendations for all users...")
    recommendations_all = recommend_exercise_plans_for_all(model, X_explain, top_k)
    # print("recommendations", recommendations_all)
    
    save_list_as_csv(recommendations_all, "../results/ml/recommendations/", "workout_recommendations.csv")

    print("Generating LIME explanations for all users...")
    feature_names=X_explain.columns
    lime_explanations_all = generate_lime_explanations_for_all(model, X_explain, feature_names, top_k)
    # print("lime", lime_explanations_all)
    save_list_as_csv(lime_explanations_all, "../results/ml/explanations/lime/", "lime_explanations.csv")
    
    print("Generating SHAP explanations for all users...")
    shap_explanations_all = generate_shap_explanations_for_all(model, X_explain, top_k)
    # print("shap", shap_explanations_all)
    save_list_as_csv(shap_explanations_all, "../results/ml/explanations/shap/", "shap_explanations.csv")