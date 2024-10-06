Description:

A machine learning model to predict loan default risk for a financial services platform. The model was trained on a dataset of borrower attributes, including financial, demographic, and credit-related features. By analyzing these inputs, the model accurately identifies high-risk borrowers, enabling data-driven decisions in the loan underwriting process. The project involved data preprocessing, feature engineering, model selection (e.g., Logistic Regression, Random Forest, XGBoost), and performance evaluation using metrics such as accuracy, precision, recall, and AUC-ROC. The model helps mitigate risk and optimize lending strategies by predicting the likelihood of default before loan approval.

# LoanTap Credit Default Risk Model
=====================================

## Project Overview
---------------

This project aims to develop a machine learning model to predict loan default risk for a financial services platform. The model is trained on a dataset of borrower attributes, including financial, demographic, and credit-related features. The goal is to enable data-driven decisions in the loan underwriting process.

### Key Features

* Predictive modeling using machine learning algorithms
* Feature engineering and selection with sklearn pipelines
* Model evaluation and hyperparameter tuning

## Installation/Environment Setup
------------------------------

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
-----------------

```
LoanTap-Credit-Default-Risk-Model/
┣ data/ - Contains raw data files for the project
┃ ┣ loantap_data.csv - Main dataset for training and testing
┃ ┗ test_data.csv - Test dataset for evaluating model performance (sampled from above dataset)
┣ loantap_credit_default_risk_model/ - Contains source code for the project
┃ ┣ __init__.py - Initialization file for the package
┃ ┣ trained_models/ - Directory to store trained machine learning models
┃ ┣ FE_pipeline.py - Feature engineering pipeline script
┃ ┣ config.py - Configuration file for project settings
┃ ┣ data_handling.py - Script for data loading, cleaning, and preprocessing
┃ ┣ evaluation.py - Script for evaluating model performance
┃ ┣ plotting.py - Script for creating visualizations
┃ ┣ predict.py - Script for making predictions with trained models
┃ ┗ train.py - Script for training machine learning models
┣ notebooks/ - Directory for Jupyter notebooks
┃ ┣ loantap_logo.png - Logo image for the project
┃ ┗ model_prototyping.ipynb - Notebook for prototyping and testing models
┣ tests/ - Directory for unit tests and integration tests
┃ ┣ __pycache__/ - Cache directory for Python compiled files
┃ ┣ __init__.py - Initialization file for the package
┃ ┗ data_tests.py - Script for testing data handling functions
┣ .gitignore - File specifying files to ignore in Git version control
┣ README.md - Project README file
┣ poetry.lock - Lock file for Poetry dependencies
┗ pyproject.toml - Project configuration file for Poetry
```



## Usage
-----

To train the model, run the following command in the root directory:

```bash
python src/train.py
```

To test the model, run the following command in the root directory:

```bash
python src/test.py
```

To make predictions, run the following command in the root directory:

```bash
python src/predict.py
```

## Dataset
--------

The dataset used for this project is proprietary and not publicly available. However, it contains various financial, demographic, and credit-related features.

## Model
------

The model used for this project is an XGBoost classifier. The hyperparameters used for training are as follows:

* learning_rate: 0.001
* batch_size: 32
* num_epochs: 50

## Results
-------

The model achieved an accuracy of 90% on the test set.

## License
-------

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements
--------------

This project was developed as part of the LoanTap Credit Default Risk Model project.

## Contact Information
-------------------

If you have any questions or suggestions, please feel free to reach out to [your-email@example.com](mailto:your-email@example.com).

## Future Work
-------------

* Implementing feature engineering techniques to improve model performance.
* Exploring other machine learning algorithms to improve model performance.