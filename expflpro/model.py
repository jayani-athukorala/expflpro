import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Reduce TensorFlow logging verbosity
# 0 = all messages are logged, 1 = INFO messages are not printed, 
# 2 = INFO and WARNING messages are not printed, 3 = only ERROR messages are printed.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_model(input_shape, num_classes):
    """
    Build and compile a simple feedforward neural network model for a recommendation system.
    
    Parameters:
    - input_shape (int): The number of input features (dimensionality of user feature vector).
    - num_classes (int): The number of output classes (e.g., categories to recommend).

    Returns:
    - model (tensorflow.keras.Model): Compiled Keras model.
    """
    
    # Define input layer
    # 'User_Features' is a placeholder name for the input tensor.
    # input_shape specifies the dimensionality of the input feature vector.
    user_input = Input(shape=(input_shape,), name='User_Features')

    # Add first dense layer with ReLU activation
    # The layer has 64 units and introduces non-linearity.
    x = Dense(64, activation='relu', name='Dense_Layer_1')(user_input)

    # Add dropout to prevent overfitting
    # Dropout randomly sets 30% of the input units to 0 during training.
    x = Dropout(0.3, name='Dropout_1')(x)

    # Add second dense layer with fewer units (32) to reduce dimensionality
    x = Dense(32, activation='relu', name='Dense_Layer_2')(x)

    # Add another dropout layer
    # This layer sets 20% of the input units to 0 during training.
    x = Dropout(0.2, name='Dropout_2')(x)

    # Add output layer with softmax activation
    # The number of units equals the number of output classes.
    # Softmax ensures the outputs represent probabilities summing up to 1.
    output = Dense(num_classes, activation='softmax', name='Output')(x)

    # Create the model
    # Input: user feature vector
    # Output: probabilities for each class
    model = Model(inputs=[user_input], outputs=output, name='Recommendation_Model')

    # Compile the model
    # Optimizer: Adam (adaptive learning rate optimization).
    # Loss: sparse_categorical_crossentropy for multi-class classification.
    # Metric: accuracy to monitor the fraction of correct predictions.
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    return model
