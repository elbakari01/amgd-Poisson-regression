
### Adaptive momentum gradient descent 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import special
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  
from matplotlib.ticker import PercentFormatter
import time




RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# matplotlib style
plt.style.use('ggplot')

# Clipping function
def clip(x, threshold=None):
    if threshold is None:
        return x
    return np.clip(x, -threshold, threshold)

#Poisson log-likelihood function 
def poisson_log_likelihood(beta, X, y):
    """
     negative Poisson log-likelihood
    """
    linear_pred = X @ beta
    linear_pred = np.clip(linear_pred, -20, 20)
    mu = np.exp(linear_pred)
    
    log_likelihood = np.sum(y * linear_pred - mu - special.gammaln(y + 1))
    
    return -log_likelihood  # Negative because we want to minimize the function

# Evaluation metrics function 
def evaluate_model(beta, X, y, target_name='Target'):
    """
    Evaluate model performance for a single target
    """
    linear_pred = X @ beta
    linear_pred = np.clip(linear_pred, -20, 20)
    y_pred = np.exp(linear_pred)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y - y_pred))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    # Mean Poisson Deviance
    eps = 1e-10  # To avoid log(0)
    deviance = 2 * np.sum(y * np.log((y + eps) / (y_pred + eps)) - (y - y_pred))
    mean_deviance = deviance / len(y)
    
    results = {
        'MAE': mae,
        'RMSE': rmse,
        'Mean Deviance': mean_deviance,
        'Non-zero coeffs': np.sum(np.abs(beta) > 1e-6),
        'Sparsity': 1.0 - (np.sum(np.abs(beta) > 1e-6) / len(beta))
    }
    
    return results