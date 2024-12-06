from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import shap
from lime.lime_tabular import LimeTabularExplainer
from skimage.segmentation import mark_boundaries
import numpy as np
import os

def train(model, train_loader, num_epochs=1):
    # Use CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)  # Logits of shape [batch_size, num_classes]
            
            # Compute loss
            loss = criterion(outputs, y_batch)  # y_batch should be [batch_size]
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader):
    # Use CrossEntropyLoss for evaluation
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Forward pass
            outputs = model(X_batch)  # Logits of shape [batch_size, num_classes]

            # Compute loss
            batch_loss = criterion(outputs, y_batch)
            total_loss += batch_loss.item()

            # Compute predictions
            _, predicted = torch.max(outputs, 1)  # Predicted classes [batch_size]

            # Count correct predictions
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    # Compute accuracy
    accuracy = correct / total
    average_loss = total_loss / len(test_loader)
    return average_loss, accuracy


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_weights(net):
    ndarrays = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return ndarrays


# Funtion to generate lime explanations
def generate_lime_explanations(net, testloader, device, feature_names, class_names, save_dir="flwr_pro/results"):
    """
    Generate LIME explanations for a multi-class classification model.

    Parameters:
    - net: Trained PyTorch model for classification.
    - testloader: DataLoader containing test data.
    - device: Device to run the model (e.g., 'cuda' or 'cpu').
    - feature_names: List of feature names for the dataset.
    - class_names: List of class names for multi-class classification.
    - save_dir: Directory where the LIME explanation file will be saved.

    Returns:
    - LIME explanation for a specific test instance.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    net.to(device)
    net.eval()

    # Prediction function for LIME
    def predict(features):
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = net(features_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probs

    # Fetch a single instance for explanation
    batch = next(iter(testloader))
    features, labels = batch
    features = features.numpy()

    # Initialize LIME explainer
    explainer = LimeTabularExplainer(
        training_data=features,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    # Explain the first instance in the batch
    instance = features[0]
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict,
        num_features=5,
        top_labels=3
    )

    # Save explanation to the specified directory
    explanation_file_path = os.path.join(save_dir, "lime_multiclass_explanation.html")
    explanation.save_to_file(explanation_file_path)
    print(f"LIME explanation saved to {explanation_file_path}")

    return explanation


