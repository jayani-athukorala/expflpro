import numpy as np
import pandas as pd
import surprise
# from recommenders.datasets.python_splitters import python_random_split
from flwr_datasets.partitioner import IidPartitioner, NaturalIdPartitioner, ShardPartitioner
from flwr_datasets import FederatedDataset
from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from surprise import Dataset, Reader
import sys
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense


fds = None  # Cache FederatedDataset

# Enhanced Preprocessing
def preprocess_data(data):
    """
    Preprocess the dataset to handle missing values, normalize features, and encode categorical variables.

    Parameters:
    - data : The input data

    Returns:
    - processed_data : Preprocessed data
    """
    mappings = {}
    # Handle missing values in 'workout_type'
    unique_workouts = data['workout_type'].dropna().unique()
    data['workout_type'] = data['workout_type'].fillna(np.random.choice(unique_workouts))
    # data['workout_type'] = data['workout_type'].fillna('Unknown')
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # Drop date column
    if 'date' in data.columns:
        data.drop(['date'], axis=1, inplace=True)
    # Add `mood_score` column
    mapping = {'Happy': 1, 'Neutral': 0, 'Stressed': -1, 'Tired': -2}
    data['mood_score'] = data['mood'].map(mapping).fillna(0)

    # Numerical and Categorical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    # Preprocessing Pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    processed_data = preprocessor.fit_transform(data)
    processed_columns = numerical_cols + \
        list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols))
    processed_data = pd.DataFrame(processed_data, columns=processed_columns)

    # Add 'user_id' and encoded 'workout_type'
    if 'user_id' in data.columns:
        processed_data['user_id'] = data['user_id'].values
    if 'workout_type' in data.columns:
        label_encoder = LabelEncoder()
        data['workout_type'] = label_encoder.fit_transform(data['workout_type'])
        mappings['workout_type'] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        processed_data['workout_type'] = data['workout_type'].values

    return processed_data, mappings

# Generate user-worout matrix
def generate_interactions_data(data, weights=None):
    """
    Generate user-workout ratings based on engagement metrics and workout frequency

    Parameters:
    - data : The input data containing user interactions with workouts
    - weights : Weights for combining engagement metrics into ratings (for now manual)

    Returns:
    - aggregated : user-workout ratings with user_id, workout_type, and rating
    """
    if weights is None:
        weights = {'steps': 0.04, 'active_minutes': 0.04, 'calories_burned': 0.02, 'workout_frequency': 0.9}

    data['frequency'] = data.groupby(['user_id', 'workout_type'])['workout_type'].transform('count')
    aggregated = data.groupby(['user_id', 'workout_type']).agg({
        'steps': 'mean',
        'active_minutes': 'mean',
        'calories_burned': 'mean',
        'frequency': 'mean'
    }).reset_index()

    scaler = MinMaxScaler()
    aggregated['workout_frequency'] = scaler.fit_transform(aggregated[['frequency']])
    aggregated['combined_score'] = (
        weights['steps'] * aggregated['steps'] +
        weights['active_minutes'] * aggregated['active_minutes'] +
        weights['calories_burned'] * aggregated['calories_burned'] +
        weights['workout_frequency'] * aggregated['workout_frequency']
    )
    aggregated['rating'] = pd.qcut(aggregated['combined_score'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    return aggregated[['user_id', 'workout_type', 'rating']]

# Split data for centralized learning environment
def prepare_collaborative_data(rating_data):
    """
    Prepare collaborative filtering data for training, testing, and explanation.

    Parameters:
        rating_data: Data containing 'user_id', 'workout_type', and 'rating'.

    Returns:
        trainset: Training dataset for the model.
        testset: Test dataset for evaluation.
        explainset: Subset of the testset for explanations.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rating_data[['user_id', 'workout_type', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    _, explainset = sklearn_train_test_split(testset, test_size=0.5, random_state=42)

    # Convert explainset to a pandas DataFrame
    explainset_df = pd.DataFrame(explainset, columns=['user_id', 'workout_type', 'rating'])

    return trainset, testset, explainset_df

# Build user-workout matrix
def build_user_item_matrix(interaction_data, fill_value=0):
    """
    Build a user-workout matrix from the rating data.
    """
    user_item_matrix = interaction_data.pivot(
        index='user_id', columns='workout_type', values='rating'
    ).fillna(fill_value)
    
    return user_item_matrix

# Get user-features and workout-features
def get_features(data):
    """
    Get user and workout features for content/context-based filtering.

    Parameters:
    - data: Preprocessed workout interaction dataset.

    Returns:
    - user_features: DataFrame with user-specific contextual features.
    - workout_features: DataFrame with workout-specific features.
    """
    # User Features: Aggregate user-specific data (e.g., mood, heart rate, steps)
    user_features = data.groupby('user_id').agg({
        'mood_score': 'mean',
        'heart_rate_avg': 'mean',
        'steps': 'mean',
        'active_minutes': 'mean',
        'sleep_hours': 'mean'
    }).reset_index()

    # Workout Features: Aggregate workout-specific data
    workout_features = data.groupby('workout_type').agg({
        # 'weather_conditions': lambda x: pd.Series.mode(x)[0],  # Most common weather condition
        'calories_burned': 'mean',
        'distance_km': 'mean',
        'active_minutes': 'mean',
        'mood_score': 'mean'
    }).reset_index()

    # Set indices for easy access
    user_features.set_index('user_id', inplace=True)
    workout_features.set_index('workout_type', inplace=True)

    return user_features, workout_features

# Load data as partitions
def load_data(partition_id: int, num_partitions: int, dataset_path="data"):
    """
    Load and process federated data for a given partition ID.
    Splits data into trainset, testset, and explainset.
    """
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        # partitioner = NaturalIdPartitioner(partition_by="location")
        fds = FederatedDataset(dataset=dataset_path, partitioners={"train": partitioner})
    
    # Load partition data and preprocess
    partition = fds.load_partition(partition_id, "train")   
    partition_data = pd.DataFrame(partition)
    
    return partition_data

# Split data for federated environment
def prepare_data(interaction_data):
    """
    Prepare collaborative filtering data for training, testing, and explanation.

    Parameters:
        rating_data: Data containing 'user_id', 'workout_type', and 'rating'.

    Returns:
        trainset: Training dataset for the model.
        testset: Test dataset for evaluation.
        explainset: Subset of the testset for explanations.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(interaction_data[['user_id', 'workout_type', 'rating']], reader)

    trainset, test = train_test_split(data, test_size=0.2, random_state=42)
    trainser_len = len(interaction_data) - len(test)
    testset, explainset = sklearn_train_test_split(test, test_size=0.333, random_state=42)

    # Convert explainset to a pandas DataFrame
    explainset_df = pd.DataFrame(explainset, columns=['user_id', 'workout_type', 'rating'])

    return trainset, testset, explainset_df, trainser_len


##############################################################################################################    4
from sklearn.metrics.pairwise import cosine_similarity
# def generate_workout_similarities(data):    

#     print(data.columns)
#     # Extract the relevant one-hot encoded columns dynamically
#     workout_columns = data.loc[:, 'weather_conditions_Clear': 'mood_Tired'].columns
#     print("Selected feature columns for similarity:", workout_columns)

#     # Filter the dataset to include only the relevant columns
#     workout_features = data[workout_columns]

#     # Compute cosine similarity between workout features
#     similarity_matrix = cosine_similarity(workout_features)

#     return similarity_matrix

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# def generate_workout_similarities(data):
#     # Extract relevant columns
#     workout_columns = data.loc[:, 'weather_conditions_Clear': 'mood_Tired'].columns
#     workout_features = csr_matrix(data[workout_columns].values)  # Convert to sparse matrix

#     # Compute cosine similarity using sparse matrix
#     similarity_matrix = cosine_similarity(workout_features, dense_output=False)

#     return similarity_matrix  # Returns a sparse similarity matrix

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# def generate_workout_similarities(data, block_size=1000):
#     # Extract the relevant one-hot encoded columns dynamically
#     workout_columns = data.loc[:, 'weather_conditions_Clear': 'mood_Tired'].columns
#     workout_features = data[workout_columns].values  # Convert to NumPy array for efficient processing

#     n_samples = workout_features.shape[0]
#     similarity_results = []

#     # Compute cosine similarity block by block
#     for i in range(0, n_samples, block_size):
#         for j in range(i, n_samples, block_size):
#             block_i = workout_features[i:i+block_size]
#             block_j = workout_features[j:j+block_size]
            
#             # Compute cosine similarity for the blocks
#             similarity = cosine_similarity(block_i, block_j)
#             similarity_results.append(similarity)

#             # Optionally save or process the similarity matrix here to avoid storing in memory

#     return similarity_results  # Return results in smaller pieces or process directly




