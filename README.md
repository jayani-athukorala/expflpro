# Explainable Federated Learning Project (expflpro: A Flower / TensorFlow app)

This project implements personalized explainable federated learning for exercise plan recommendation. Below is the structure of the project and instructions on how to set up the environment.

## Project Structure

The repository is organized as follows to support the implementation of personalized explainable federated learning for fitness recommendation:

```
├── expmlpro/                   # Centralized Machine Learning (ML) for Workout Plan Recommendation
├── expflpro/                   # Federated Learning (FL) for Workout Recommendation
├── fl_model/                   # Saved Trained global FL model
├── ml_model/                   # Saved Trained centralized model
├── data/                       # Dataset storage (training and testing data)
├── results/                    # Stores model results, logs, and evaluation metrics
├── README.md                   # Project documentation and guidelines
└── requirements.txt            # Dependencies and package requirements
```

## Setting Up the Environment

### 1. Clone the Repository

```sh
git clone git@github.com:jayani-athukorala/expflpro.git
```
### 2. Install and activate conda environment
```sh
# conda list #View the list of packages available on conda
# conda info --envs #View the list of conda environments
conda create -n fedenv python=3.10
conda activate fedenv

```
### 3. Install Required Packages (Dependancies)

```sh
pip install -e .
pip install -r requirements.txt
# conda env export > environment.yml # Save environment configuration for reproducibility
# conda env create -f environment.yml # Recreate the environment
```

## Run the application

```sh
flwr run . # Run federated learning
cd expmlpro
python centralized.py # Run centrailzed part
python generate_graphs.py # Generate related graphs for results analysis
python generate_evaluations.py # Generate related graphs for explanation evaluations
```


## Deactivate the Environment

```sh
conda deactivate
# conda remove --name fedenv --all # Remove the fedenv conda environment
# conda clean --all # Cleanup unused dependencies
```
