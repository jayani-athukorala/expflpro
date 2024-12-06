"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from flwr_pro.task import (
    evaluate,
    get_weights,
    set_weights,
    train,
    generate_lime_explanations,
)

from flwr_pro.dataset import load_data
from flwr_pro.model import FitnessClassifier


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate(self.net, self.testloader)

        # Generate LIME explanations
        lime_explanation = generate_lime_explanations(
            net=self.net,
            testloader=self.testloader,
            device="cpu",
            feature_names=["Calories Burn","Dream Weight", "Actual Weight", "Age", "Gender", "Duration", "Heart Rate", "BMI", "Exercise Intensity"],
            class_names=["Exercise 1", "Exercise 2", "Exercise 3", "Exercise 4", "Exercise 5",
                        "Exercise 6", "Exercise 7", "Exercise 8", "Exercise 9", "Exercise 10"]
        )

        return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]

    train_loader, test_loader = load_data(
        partition_id=partition_id, num_partitions=context.node_config["num-partitions"]
    )
    net = FitnessClassifier()
    return FlowerClient(net, train_loader, test_loader).to_client()


app = ClientApp(client_fn=client_fn)
