# LoanTap Credit Default Risk Model

## Project Overview

This project aims to develop a machine learning model to predict loan default risk for a financial services platform. The model is trained on a dataset of borrower attributes, including financial, demographic, and credit-related features. The goal is to enable data-driven decisions in the loan underwriting process.

### Key Features

* Predictive modeling using machine learning algorithms
* Feature engineering and selection with sklearn pipelines
* Model evaluation and hyperparameter tuning

## Installation/Environment Setup

### Prerequisites

* Python (>=3.7)
* Poetry (>=1.8)
* Dependencies listed in `pyproject.toml`

### Installation Instructions

1. Clone the repository:

   ```bash
   gh repo clone jyothisable/LoanTap-Credit-Default-Risk-Model
   ```

2. Set up a virtual environment and dependencies:
  
   ```bash
   poetry install
   poetry shell
   ```

## Project Structure

```ASCI
LoanTap-Credit-Default-Risk-Model/
┣ data/ - Contains raw data files for the project
┃ ┣ loantap_data.csv - Main dataset for training and testing
┃ ┗ test_data.csv - Test dataset for evaluating model performance (sampled from above dataset)
┣ loantap_credit_default_risk_model/ - Contains source code for the project
┃ ┣ __init__.py - Initialization file for the package
┃ ┣ trained_models/ - Directory to store trained machine learning models
┃ ┃ ┣ XBG_model.pkl - Main trained model
┃ ┃ ┗ target_pipeline_fitted.pkl - Fitted target pipeline for finding inverse after prediction
┃ ┣ FE_pipeline.py - Feature engineering pipeline configurations
┃ ┣ config.py - Configuration file for project settings
┃ ┣ data_handling.py - Script for data loading, cleaning, and saving data and pipelines
┃ ┣ evaluation.py - Script for evaluating model and pipeline performance
┃ ┣ plotting.py - Script for creating visualizations
┃ ┣ predict.py - Script for making predictions with trained models
┃ ┗ train.py - Script for training machine learning models
┣ notebooks/ - Directory for Jupyter notebooks
┃ ┣ loantap_logo.png - Logo image for the project
┃ ┗ model_prototyping.ipynb - Notebook for EDA, prototyping and testing models
┣ tests/ - Directory for unit tests and integration tests
┃ ┗ data_tests.py - Script for testing data handling functions
┣ .gitignore - File specifying files to ignore in Git version control
┣ README.md - Project README file
┗ pyproject.toml - Project project dependencies file using Poetry
```

## Usage

To train the model, run the following command in the root directory:

```bash
python loantap_credit_default_risk_model/train.py
```

To test the model, run the following command in the root directory:

```bash
python tests/test.py #TODO: add file path
```

To make predictions, run the following command in the root directory:

```bash
python loantap_credit_default_risk_model/predict.py
```

## Dataset

LoanTap has provided a dataset containing various financial and credit-related features for loan applicants. Below is a summary of the dataset:

| Column               | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| loan_amnt            | The loan amount applied for by the borrower                        |
| term                 | Loan term in months (36 or 60)                                     |
| int_rate             | Interest rate on the loan                                          |
| installment          | Monthly payment owed if the loan originates                        |
| grade                | LoanTap assigned grade                                             |
| sub_grade            | LoanTap assigned subgrade                                          |
| emp_title            | Job title supplied by the borrower                                 |
| emp_length           | Employment length in years (0-10)                                  |
| home_ownership       | Home ownership status                                              |
| annual_inc           | Self-reported annual income                                        |
| verification_status  | Income verification status (verified/not verified)                 |
| issue_d              | Date the loan was funded                                           |
| loan_status          | Target variable (current loan status: default or not)              |
| purpose              | Purpose of the loan                                                |
| dti                  | Debt-to-income ratio                                               |
| earliest_cr_line     | Month the borrower’s earliest credit line was opened               |
| open_acc             | Number of open credit lines                                        |
| pub_rec              | Number of derogatory public records                                |
| revol_bal            | Total revolving credit balance                                     |
| revol_util           | Revolving line utilization rate                                    |
| total_acc            | Total number of credit lines                                       |
| initial_list_status  | The initial listing status of the loan. Possible values are – W, F |
| pub_rec              | Number of derogatory public records                                |
| application_type     | Individual or joint application                                    |
| mort_acc             | Number of mortgage accounts                                        |
| pub_rec_bankruptcies | Number of public record bankruptcies                               |
| address              | Address of the individual                                          |

Dataset available at path: `data/loantap_data.csv`

## Model

The model used for this project is an XGBoost classifier. The hyperparameters used for training are as follows:

* learning_rate: 0.15
* n_estimators: 300
* max_depth: 5
* subsample: 0.95
* colsample_bytree: 0.95
* lambda: 0.1
* tree_method: hist
* eval_metric: aucpr

## Results

The model achieved an f1 score of 66% on the test set.

## License

This project is licensed under the MIT License.

## Future Work

* Implementing MLflow for experimental tracking and docker for deployment
* Exploring other machine learning algorithms to improve model performance.