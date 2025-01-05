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


def recommend_exercise_plans(user_features, model, k=2):
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
