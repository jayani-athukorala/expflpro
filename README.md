# expflpro: A Flower / TensorFlow app

# Explainable Federated Learning Project

This project implements personalized explainable federated learning for fitness data. Below is the structure of the project and instructions on how to set up the environment.

## Project Structure

The repository is organized as follows to support the implementation of personalized explainable federated learning for fitness recommendation:
.
├── expmlpro/                   # Centralized Machine Learning (ML) for Workout Recommendation
│   └── README.md               # Documentation for the ml-pro module
│
├── expflpro/                   # Federated Learning (FL) for Workout Recommendation
│   └── README.md               # Documentation for the fl-pro module
|               
├── data/                       # Dataset
|  
├── results/            
│
├── README.md                   # Overview and documentation of the entire project
└── requirements.txt            # Dependencies for the project

## Setting Up the Environment

### 1. Clone the Repository

```sh
git clone git@github.com:jayani-athukorala/expflpro.git
```
### 2. Install and activate conda environment
```sh
conda create -n fedenv python=3.10
conda activate fedenv

```
### 3. Install Required Packages (Dependancies)

```sh
pip install -e .
pip install -r requirements.txt
conda env export > environment.yml # Save environment configuration for reproducibility
conda env create -f environment.yml # Recreate the environment
```

### 4. Run the application

```sh
flwr run . # Run federated learning
cd expmlpro
python centralized.py # Run centrailzed part
python graph.py # Generate related graphs for results analysis
```


### 5. Deactivate the Environment

```sh
conda deactivate
```
