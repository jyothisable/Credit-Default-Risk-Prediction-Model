# End-to-End ML Credit Default Risk Prediction Model

<img src="notebooks/Designer.jpeg" alt="logo banner" style="width: 800px;"/>

## [LIVE DEMO](https://credit-default-risk-prediction-model.streamlit.app/) -> [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-default-risk-prediction-model.streamlit.app/)

## Introduction

This project aims to develop an End-to-End machine learning model to predict loan default risk for a financial services platform. The model is trained on a dataset of borrower attributes, including financial, demographic, and credit-related features. The goal is to enable data-driven decisions in the loan underwriting process.

### Tech used

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)![Pandas](https://img.shields.io/badge/Pandas-1.x-brightgreen?logo=pandas)![NumPy](https://img.shields.io/badge/NumPy-1.x-orange?logo=numpy)![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blueviolet?logo=plotly)![Plotly](https://img.shields.io/badge/Plotly-5.x-informational?logo=plotly)![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24%2B-lightgrey?logo=scikit-learn)![MLFlow](https://img.shields.io/badge/MLFlow-1.x-blue?logo=mlflow)![Optuna](https://img.shields.io/badge/Optuna-3.x-red?logo=optuna)![FastAPI](https://img.shields.io/badge/FastAPI-0.85%2B-brightgreen?logo=fastapi)![Streamlit](https://img.shields.io/badge/Streamlit-1.x-orange?logo=streamlit)

### Key Features

* Predictive modeling using machine learning algorithms
* Feature engineering and selection with sklearn pipelines
* Model evaluation and hyperparameter tuning with optuna
* Model registry and tracking with MLFlow
* Testing with pytest
* Deployment of API with FastAPI
* Frontend with streamlit App
* Containerization with Docker

## Installation / Environment Setup

### Prerequisites

* Python (>=3.7)
* Poetry (>=1.8) see how to install [here](https://python-poetry.org/docs/)
* Dependencies listed in `pyproject.toml` or `requirements.txt` (not to use `conda.yaml` for dependencies)

### Installation Instructions

1. Clone the repository:

   ```bash
   gh repo clone jyothisable/LoanTap-Credit-Default-Risk-Model
   ```

2. Set up virtual environment and dependencies:
  
   In a terminal, navigate to the project directory and run the following commands:
      ```bash
      poetry install
      poetry shell
      ```

## Project Structure

```ASCI
┣ 📂 data/ - Contains raw and processed data files for the project
┃ ┣ 📄 LCDataDictionary.xlsx - Metadata dictionary for the loan dataset
┃ ┣ 🌍 US_zip_to_cord.csv - CSV mapping U.S. ZIP codes to geographical coordinates
┃ ┣ 📊 loan.csv - Main dataset containing detailed loan information (not available in GitHub because of size limitation, downloaded from [Kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset))
┃ ┣ 📉 loan_reduced.csv - Filtered and reduced version of the main dataset
┃ ┣ 🧪 test_data.csv - Dataset used for evaluating model performance
┃ ┗ 📚 train_data.csv - Dataset used for training machine learning models

┣ 🧩 Prediction_Model/ - Contains source code for model development, training, and predictions
┃ ┣ 🗃️ trained_models/ - Directory for storing trained models and pipelines
┃ ┃ ┣ 🧠 XBG_model_final.pkl - Final trained XGBoost model
┃ ┃ ┣ 📊 fe_eval_model.pkl - Feature engineering evaluation model
┃ ┃ ┣ ⚙️ fe_eval_tuned_model.pkl - Tuned model for evaluating feature engineering
┃ ┃ ┣ 🔄 fe_pipeline_fitted_final.pkl - Fitted feature engineering pipeline
┃ ┃ ┗ 🔙 target_pipeline_fitted.pkl - Fitted target pipeline for reverse transformations after predictions
┃ ┣ 📦 __init__.py - Initialization file for package setup
┃ ┣ 🧩 FE_pipeline.py - Script for feature engineering pipeline configurations
┃ ┣ ⚙️ config.py - Configuration file defining project parameters and settings
┃ ┣ 🧹 data_handling.py - Script for loading, cleaning, and managing datasets and pipelines
┃ ┣ 🧪 evaluation.py - Script for evaluating models and feature engineering pipelines
┃ ┣ 📊 get_features.py - Utility for extracting features from the data
┃ ┣ 📈 plotting.py - Script for generating plots and visualizations
┃ ┣ 🤖 predict.py - Script for running predictions using trained models
┃ ┗ 🏋️ train.py - Main script for training machine learning models

┣ 📒 notebooks/ - Directory for Jupyter notebooks, images, and analysis reports
┃ ┣ 🖼️ Designer.jpeg - Image file for branding or presentation purposes
┃ ┣ 📄 EDA_report.html - HTML report summarizing exploratory data analysis
┃ ┣ 📊 LC.png - Additional image resource related to the project
┃ ┣ 🏷️ loantap_logo.png - Logo image for the project
┃ ┗ 🧪 model_prototyping.ipynb - Jupyter notebook for exploratory data analysis and model prototyping

┣ 🧪 tests/ - Contains unit tests and integration tests to ensure code robustness
┃ ┣ 📦 __init__.py - Initialization file for the tests package
┃ ┣ 🔍 data_tests.py - Tests for data handling and processing functions
┃ ┗ 🧪 test_prediction.py - Tests for the prediction module

┣ 🚫 .dockerignore - Specifies files and directories to ignore when building Docker images
┣ 🚫 .gitignore - File to exclude specific files and directories from Git version control
┣ ⚙️ MLProject - Configuration for running MLflow projects
┣ 🚀 fastapi_app.py - Script to run a FastAPI web application for serving models or APIs
┣ 🎨 streamlit_app.py - Script to run a Streamlit application for visualizing data and making predictions
┣ 🐋 Dockerfile - Instructions to build a Docker container for the project
┣ 🧪 conda.yaml - Environment configuration file for setting up dependencies of MLFlow (not to be used for project setup)
┣ 📜 requirements.txt - List of required Python packages for the project (for use with pip)
┣ ⚙️ pyproject.toml - Configuration file defining project dependencies and settings using Poetry
┣ 🔒 poetry.lock - Dependency lock file for consistent environment setup using Poetry
┣ 📜 LICENSE.md - Legal license information for the project
┣ 📘 README.md - Documentation file with an overview, setup, and usage instructions
```

## Usage

> [!IMPORTANT] 
> Make sure to activate poetry by running `poetry shell` in the root directory before running any commands or add  `poetry run` in beginning of every command below.

### Training

To train the model, run the following command in the root directory:

```bash
python Prediction_Model/train.py # feature engg pipeline
python Prediction_Model/train.py # Traning pipeline
```

### Prediction

To make predictions on test data, run the following command in the root directory:

```bash
python Prediction_Model/predict.py
```

### Testing

To test the model, run the following command in the root directory:

```bash
pytest tests/test_prediction.py
```

### Running Web Apps

#### fastAPI

   ```bash
   python fastapi_app.py 
   ```

   POST to `localhost:8000/predict` with Postman or use `localhost:8000/predict/docs` in browser for documentation / testing

#### Streamlit App

   ```bash
   streamlit run streamlit_app.py # local   ```
   ```

## Usage with docker

### 1. Pulling the Docker Image

To pull the Docker image from Docker Hub, run the following command:

```sh
# Pull the docker image
docker pull jyothisable/credit_risk_streamlit_app
```

### 2. Running the Docker Container

To run the Docker container, use the following command:

```sh
# Run the docker container
docker run -p 8501:8501 credit_risk_streamlit_app # goto http://localhost:8501 in browser
```

## Dataset

Refer to [here](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset) or `data/LCDataDictionary.xlsx`

## Model

The model used for this project is an XGBoost classifier. The hyperparameters used for training are as follows after tuning with optuna:

```python
{
    'max_depth': 9,
    'learning_rate': 0.094,
    'n_estimators': 507,
    'gamma': 0.0062,
    'subsample': 0.962,
    'colsample_bytree': 0.795,
    'lambda': 0.389,
    'alpha': 0.0233,
    'scale_pos_weight': 1.99,
    'min_child_weight': 2,
    'grow_policy': 'lossguide'
}
```

## Results

The model achieved an f1 score of **79.12%** and recall of **85.84%** on the test dataset.

## License

This project is licensed under the MIT License.
