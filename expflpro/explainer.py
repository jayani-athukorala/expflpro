import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import MinMaxScaler

def generate_shap_explanations(model, scaler, mappings, X_explain, user_features):
    """
    Generate SHAP explanations for a random user.

    Args:
        model (tf.keras.Model): Trained model.
        scaler (MinMaxScaler): Fitted scaler for normalization.
        mappings (dict): Dictionary mapping encoded values to their original labels.
        X_explain (pd.DataFrame): Training dataset for SHAP initialization.
        user_features (pd.DataFrame): Features of the selected user.

    Returns:
        dict: SHAP explanations as JSON.
    """
    # Ensure user_features is a copy to avoid SettingWithCopyWarning
    user_features = user_features.copy()
    # Use a representative subset for the background
    background_data = shap.sample(X_explain, 100)

    # Normalize user features
    user_features_normalized = user_features.copy()
    user_features_normalized[['Weight', 'Height', 'BMI', 'Age']] = scaler.transform(
        user_features[['Weight', 'Height', 'BMI', 'Age']]
    )

    explainer = shap.KernelExplainer(
        lambda x: model.predict({'User_Features': x}),
        background_data,
        l1_reg="num_features(10)"
    )
    shap_values = explainer.shap_values(user_features_normalized)[0]

    # Denormalize Gender
    reverse_gender_mapping = {v: k for k, v in mappings['Gender'].items()}
    user_features['Gender'] = user_features['Gender'].map(reverse_gender_mapping)

    # Convert SHAP explanations to JSON
    explanations = {
        "expected_value": explainer.expected_value[0],
        "shap_values": shap_values.tolist(),
        "feature_values": user_features.iloc[0].to_dict()  # Use unnormalized values
    }
    return explanations

def generate_lime_explanations(model, scaler, mappings, X_explain, user_features):
    """
    Generate LIME explanations for a random user.

    Args:
        model (tf.keras.Model): Trained model.
        scaler (MinMaxScaler): Fitted scaler for normalization.
        mappings (dict): Dictionary mapping encoded values to their original labels.
        X_explain (pd.DataFrame): Training dataset for LIME initialization.
        user_features (pd.DataFrame): Features of the selected user.

    Returns:
        dict: LIME explanations as JSON.
    """
    user_features = user_features.copy()
    # Normalize user features
    user_features_normalized = user_features.copy()
    user_features_normalized[['Weight', 'Height', 'BMI', 'Age']] = scaler.transform(
        user_features[['Weight', 'Height', 'BMI', 'Age']]
    )

    lime_explainer = LimeTabularExplainer(
        training_data=X_explain.values,
        feature_names=X_explain.columns,
        mode='classification'
    )

    # Explain the prediction for the random user
    lime_exp = lime_explainer.explain_instance(
        user_features_normalized.iloc[0].values,
        lambda x: model.predict({'User_Features': x})
    )

    # Denormalize Gender
    reverse_gender_mapping = {v: k for k, v in mappings['Gender'].items()}
    user_features['Gender'] = user_features['Gender'].map(reverse_gender_mapping)

    # Convert LIME explanations to JSON
    lime_explanations = {
        "feature_importances": [
            {"feature": imp[0], "importance": imp[1]}
            for imp in lime_exp.as_list()
        ],
        "intercept": lime_exp.intercept,
        "feature_values": user_features.iloc[0].to_dict()  # Use unnormalized values
    }
    return lime_explanations


def generate_shap_explanations_for_all(model, X_explain, k=1):
    """
    Generate SHAP explanations for all users in X_explain.

    Args:
        model (tf.keras.Model): The trained model.
        X_explain (pd.DataFrame): The dataset to explain.
        k (int): Number of top classes to explain.

    Returns:
        List[Dict]: SHAP explanations for each user.
    """
    explainer = shap.KernelExplainer(model.predict, X_explain.values)
    shap_explanations = []

    for i, user_features in enumerate(X_explain.values):
        shap_values = explainer.shap_values(np.expand_dims(user_features, axis=0))
        user_explanations = {}

        for class_idx in range(k):  # Explain top K classes
            user_explanations[f"Class {class_idx}"] = shap_values[class_idx].tolist()
        
        shap_explanations.append(user_explanations)
    
    return shap_explanations


def generate_lime_explanations_for_all(model, X_explain, feature_names, k=1):
    """
    Generate LIME explanations for all users in X_explain.

    Args:
        model (tf.keras.Model): The trained model.
        X_explain (pd.DataFrame): The dataset to explain.
        feature_names (List[str]): Names of the features.
        k (int): Number of top classes to explain.

    Returns:
        List[Dict]: LIME explanations for each user.
    """
    explainer = LimeTabularExplainer(
        X_explain.values,
        feature_names=feature_names,
        class_names=[str(i) for i in range(model.output_shape[-1])],
        mode='classification'
    )
    lime_explanations = []

    for i, user_features in enumerate(X_explain.values):
        explanation = explainer.explain_instance(
            user_features,
            model.predict,
            num_features=len(feature_names),
            top_labels=k
        )
        user_explanations = {
            label: explanation.as_list(label) for label in explanation.top_labels
        }
        lime_explanations.append(user_explanations)
    
    return lime_explanations

