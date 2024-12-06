import torch.nn as nn

class FitnessClassifier(nn.Module):
    """Model with support for 10 classifiers."""

    def __init__(self, input_dim: int = 9, num_classes: int = 10):  # Add num_classes parameter
        super(FitnessClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_classes)  # Output layer has num_classes units
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Use sigmoid for binary classification per classifier

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))  # Apply sigmoid to each output
        return x