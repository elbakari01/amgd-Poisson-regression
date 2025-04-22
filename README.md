

### Adaptive Momentum Gradient Descent (AMGD) for Penalized Poisson Regression

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)

This repository introduces **Adaptive Momentum Gradient Descent (AMGD)**, a novel optimization algorithm designed to optimize penalized Poisson regression models in high-dimensional settings. AMGD integrates adaptive learning rates, momentum updates, and adaptive soft-thresholding to address critical limitations of existing methods such as AdaGrad (rapid learning rate decay) and Adam (gradient instability in sparse data). The algorithm demonstrates significant performance improvements over competing methods and is particularly effective for modeling ecological health data.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Dataset Description](#dataset-description)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Pipeline Workflow](#pipeline-workflow)
7. [Optimization Algorithms](#optimization-algorithms)
8. [Statistical Analysis](#statistical-analysis)
9. [Visualization](#visualization)
10. [Directory Structure](#directory-structure)
11. [Contributing](#contributing)
12. [License](#license)
13. [Acknowledgments](#acknowledgments)

---

## Overview

**AMGD** is a first-order optimization algorithm specifically tailored for regularized Poisson regression problems. It combines three key components:
- **Adaptive Learning Rates**: Prevents premature convergence by dynamically adjusting step sizes.
- **Momentum Updates**: Accelerates convergence and reduces oscillations during optimization.
- **Adaptive Soft-Thresholding**: Enhances sparsity and feature selection, addressing multicollinearity and uneven feature importance.

The algorithm is validated on a large-scale ecological health dataset, demonstrating superior performance compared to popular optimization algorithms like Adam, AdaGrad, and GLMNet. AMGD achieves:
- **2.7% reduction in Mean Absolute Error (MAE)** compared to Adam.
- **56.6% reduction in MAE** compared to AdaGrad.
- Faster convergence and better fit to the statistical properties of count data.

---

## Key Features

- **AMGD Algorithm**: A novel optimization method for penalized Poisson regression with L1 and ElasticNet regularization.
- **High-Dimensional Data Handling**: Effectively handles sparse, high-dimensional, and correlated count data.
- **Cross-Validation**: Performs k-fold cross-validation to identify optimal hyperparameters.
- **Feature Selection**: Provides robust feature selection through adaptive soft-thresholding.
- **Comprehensive Visualization**: Generates plots for convergence rates, coefficient paths, feature importance, and metric comparisons.
- **Statistical Validation**: Includes bootstrap analysis and paired t-tests to assess the significance of performance differences.

---

## Dataset Description

The dataset used in this project is sourced from Kaggle and is designed for the classification of ecological health in urban environments. It consists of hourly data generated from January 1, 2018, to December 31, 2024, simulating various environmental parameters that affect ecological health.


### Features
- **Timestamp**: Date and time of observations (hourly intervals).
- **PM2.5**: Concentration of particulate matter (PM2.5) in the air.
- **Temperature**: Ambient temperature (mean: 20°C, std: 5°C).
- **Humidity**: Relative humidity (range: 30%–100%).
- **Soil_Moisture**: Soil moisture content (%), modeled using a beta distribution.
- **Biodiversity_Index**: Measure of biodiversity, simulated using a Poisson distribution.
- **Nutrient_Level**: Nutrient level in the soil (values: 0, 50, 100, 150, 200).
- **Water_Quality**: Categorical measure of water quality (0–3).
- **Air_Quality_Index**: Overall air quality index (mean: 100, std: 20).
- **Pollution_Level**: Categorical variable indicating pollution levels (Low, Moderate, High).
- **Soil_pH**: pH level of the soil (range: 5.0–8.5).
- **Dissolved_Oxygen**: Concentration of dissolved oxygen (mg/L).
- **Chemical_Oxygen_Demand**: Chemical oxygen demand (COD) in mg/L.
- **Biochemical_Oxygen_Demand**: Biochemical oxygen demand (BOD) in mg/L.
- **Total_Dissolved_Solids**: Concentration of total dissolved solids (TDS) in mg/L.
- **Ecological_Health_Label**: Categorical label representing ecological health classification ('Ecologically Healthy', 'Ecologically Stable', 'Ecologically Degraded', 'Ecologically Critical').

### Use Cases
- Training machine learning models for ecological health classification.
- Research on the impacts of environmental parameters on urban ecology.
- Development of predictive analytics tools for urban planners and environmental managers.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`

### Steps to Install

1. Clone the repository:
   ```bash
   git clone https://github.com/elbakari01/amgd-Poisson-regression.git
   cd amgd-Poisson-regression
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset (`ecological_health_dataset.csv`) is available in the root directory.

---

## Usage

To run the pipeline, execute the following command:

```bash
python src/main.py
```

### Customization Options

- **Random Seed**: Set the random seed for reproducibility using the `random_state` argument.
- **Number of Folds**: Adjust the number of folds for cross-validation using the `k_folds` argument.
- **Metric to Optimize**: Choose the evaluation metric (`MAE`, `RMSE`, `Mean_Deviance`, etc.) for selecting the best model.

Example:

```python
best_params, test_metrics, model_results = run_poisson_regression_pipeline(
    filepath="data/ecological_health_dataset.csv",
    random_state=42,
    k_folds=5,
    metric_to_optimize='MAE'
)
```

---

## Pipeline Workflow

The pipeline consists of the following steps:

1. **Data Preprocessing**:
   - Load and preprocess the ecological health dataset.
   - Handle missing values, encode categorical variables, and standardize numeric features.

2. **Data Splitting**:
   - Split the data into training (70%), validation (15%), and test (15%) sets.

3. **Hyperparameter Tuning**:
   - Perform k-fold cross-validation on the validation set to find optimal hyperparameters (e.g., regularization strength).

4. **Model Training**:
   - Train models using the best parameters identified during cross-validation.

5. **Convergence Analysis**:
   - Compare the convergence rates of different optimization algorithms.

6. **Test Evaluation**:
   - Evaluate all trained models on the test set using metrics like MAE, RMSE, Mean Deviance, and Sparsity.

7. **Visualization**:
   - Generate plots for coefficient paths, feature importance, and metric comparisons.

8. **Statistical Analysis**:
   - Perform statistical significance tests to compare the performance of optimizers.

---

## Optimization Algorithms

The pipeline supports the following optimization algorithms:

1. **AMGD (Adaptive Momentum Gradient Descent)**:
   - Combines momentum with adaptive learning rates for faster convergence.
   - Supports L1 and ElasticNet regularization.
   - Uses adaptive soft-thresholding

2. **Adam**:
   - A popular optimization algorithm that uses adaptive learning rates and momentum.

3. **AdaGrad**:
   - An optimizer that adapts the learning rate based on the historical gradient information.

4. **GLMnet**:
   - Implements coordinate descent for generalized linear models with L1 and ElasticNet penalties.

---

## Statistical Analysis

The pipeline includes statistical significance analysis to evaluate the performance of different optimizers. Key components include:

- **Bootstrap Confidence Intervals**: Estimates the variability of performance metrics across multiple runs.
- **Paired t-Tests**: Compares the performance of optimizers using paired t-tests.
- **Effect Size**: Measures the magnitude of differences between optimizers.

---

## Visualization

The pipeline generates the following visualizations:

1. **Algorithm Comparison Plots**:
   - Barplots comparing MAE, RMSE, Mean Deviance, and runtime across optimizers.

2. **Convergence Rates**:
   - Line plots showing the convergence behavior of different optimizers.

3. **Coefficient Paths**:
   - Plots showing how coefficients evolve with varying regularization strengths.

4. **Feature Importance**:
   - Bar charts highlighting the most important features selected by each optimizer.

5. **Metric Distributions**:
   - Violin plots showing the distribution of metrics across multiple runs.

---

## Directory Structure

```
amgd-Poisson-regression/
├── data/                     # Dataset files
│   └── ecological_health_dataset.csv
├── notebooks/                # Jupyter notebooks for exploratory analysis
├── src/                      # Source code
│   ├── main.py               # Main pipeline script
│   ├── preprocess.py         # Data preprocessing functions
│   ├── optimizers.py         # Optimization algorithms
│   ├── evaluation.py         # Model evaluation functions
│   ├── cross_validation.py   # Cross-validation logic
│   └── plots.py              # Visualization functions
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── LICENSE                   # License file
└── README.md                 # Project documentation
```

---

## Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

For major changes, please open an issue first to discuss the proposed changes.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.


---
## Acknowledgments

- Inspired by research on Poisson regression and ecological modeling.
- Built using Python and popular libraries such as `scikit-learn`, `numpy`, and `matplotlib`.
- Thanks to the open-source community for their contributions and support.

## Citation

If you use this project or data set in your research or work, please cite them as follows:

```bibtex
@misc{bakari2025amgdpoisson,
  author = {Ibrahim Bakari},
  title = {Adaptive Momentum Gradient Descent},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/elbakari01/amgd-Poisson-regression}
}

@misc{kaggle_ecological_health_dataset,
  author = {Laoshi},
  title = {Ecological Health Dataset},
  year = {2014},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/datasetengineer/ecological-health-dataset}
}



