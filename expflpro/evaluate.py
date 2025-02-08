from sklearn.metrics import precision_score, recall_score
import time
from memory_profiler import memory_usage
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import jaccard_score

def feature_consistency(feature_importance_lists):
    """
    Measure feature importance consistency using Spearman and Jaccard metrics.
    Args:
        feature_importance_lists (list of lists): Each list contains feature importances from a run.
    Returns:
        dict: Consistency metrics (Spearman, Jaccard).
    """
    n = len(feature_importance_lists)
    spearman_scores = []
    jaccard_scores = []

    for i in range(n):
        for j in range(i + 1, n):
            # Spearman correlation
            spearman, _ = spearmanr(feature_importance_lists[i], feature_importance_lists[j])
            spearman_scores.append(spearman)

            # Jaccard Index for top-k features
            top_k_i = set(sorted(range(len(feature_importance_lists[i])), key=lambda x: -feature_importance_lists[i][x])[:k])
            top_k_j = set(sorted(range(len(feature_importance_lists[j])), key=lambda x: -feature_importance_lists[j][x])[:k])
            jaccard = len(top_k_i.intersection(top_k_j)) / len(top_k_i.union(top_k_j))
            jaccard_scores.append(jaccard)

    return {
        "Spearman": sum(spearman_scores) / len(spearman_scores),
        "Jaccard": sum(jaccard_scores) / len(jaccard_scores)
    }


def measure_computation_overhead(xai_function, *args, **kwargs):
    """
    Measure computation time and memory overhead for an XAI method.
    Args:
        xai_function (callable): Function that generates explanations.
        *args: Arguments for the function.
        **kwargs: Keyword arguments for the function.
    Returns:
        dict: Computation time and memory usage.
    """
    start_time = time.time()
    mem_usage = memory_usage((xai_function, args, kwargs), interval=0.1)
    end_time = time.time()

    return {
        "Time (s)": end_time - start_time,
        "Memory (MB)": max(mem_usage) - min(mem_usage)
    }

def evaluate_injection_detection(true_labels, predicted_labels):
    """
    Evaluate detection rate and FPR for injection attacks.
    Args:
        true_labels (list): True labels (1 for attack, 0 for benign).
        predicted_labels (list): Predicted labels (1 for attack, 0 for benign).
    Returns:
        dict: Detection rate, FPR.
    """
    detection_rate = recall_score(true_labels, predicted_labels, pos_label=1)
    false_positive_rate = sum((predicted_labels == 1) & (true_labels == 0)) / sum(true_labels == 0)
    
    return {
        "Detection Rate": detection_rate,
        "FPR": false_positive_rate
    }

def measure_communication_overhead(data_sent, data_received, privacy_mechanism=None):
    """
    Measure communication overhead for privacy-preserving techniques.
    Args:
        data_sent (float): Total data sent (in MB).
        data_received (float): Total data received (in MB).
        privacy_mechanism (str): Name of the privacy mechanism, if applicable.
    Returns:
        dict: Communication metrics.
    """
    overhead = (data_sent + data_received) / data_sent  # Ratio of data exchanged to original data
    return {
        "Data Sent (MB)": data_sent,
        "Data Received (MB)": data_received,
        "Overhead Ratio": overhead,
        "Privacy Mechanism": privacy_mechanism
    }


def evaluate_system(lime_explanations, shap_explanations, true_labels, predicted_labels, 
                    communication_data, privacy_mechanism=None):
    """
    Comprehensive evaluation for explainability and privacy.
    """
    # Explainability
    lime_consistency = feature_consistency([exp["feature_importances"] for exp in lime_explanations])
    shap_consistency = feature_consistency([np.abs(exp["shap_values"]).mean(axis=0) for exp in shap_explanations])
    
    # Privacy
    privacy_metrics = evaluate_injection_detection(true_labels, predicted_labels)
    comm_overhead = measure_communication_overhead(*communication_data, privacy_mechanism=privacy_mechanism)
    
    return {
        "LIME Consistency": lime_consistency,
        "SHAP Consistency": shap_consistency,
        "Privacy Metrics": privacy_metrics,
        "Communication Overhead": comm_overhead
    }
