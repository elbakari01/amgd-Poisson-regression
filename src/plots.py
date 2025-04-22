
import matplotlib.pyplot as plt

def create_algorithm_comparison_plots(cv_results_df):
    """
    Create barplots comparing the performance of AMGD, AdaGrad, Adam, and GLMnet algorithms
    across different metrics with L1 and ElasticNet regularization only.
    
    Parameters:
    -----------
    cv_results_df : pandas.DataFrame
        DataFrame with cross-validation results
    
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    # Create a list to store the figures
    figs = []
    
    # Metrics to compare
    metrics = ['MAE', 'RMSE', 'Mean Deviance', 'Runtime']
    
    #Best result for each optimizer and metric combination
    best_results = []
    
    for optimizer in ['AMGD', 'AdaGrad', 'Adam', 'GLMnet']:
        optimizer_df = cv_results_df[cv_results_df['Optimizer'] == optimizer]
        
        for metric in metrics:
            if metric in ['MAE', 'RMSE', 'Mean Deviance']:
                # For these metrics, lower is better
                best_idx = optimizer_df[metric].idxmin()
            else:  # Runtime
                # For runtime, lower is better
                best_idx = optimizer_df['Runtime'].idxmin()
            
            best_results.append({
                'Optimizer': optimizer,
                'Metric': metric,
                'Value': optimizer_df.loc[best_idx, metric],
                'Regularization': optimizer_df.loc[best_idx, 'Regularization'],
                'Lambda': optimizer_df.loc[best_idx, 'Lambda']
            })
    
    # Converting to DataFrame 
    best_results_df = pd.DataFrame(best_results)
    
    # Creating a barplot for each metric
    for metric in metrics:
        metric_df = best_results_df[best_results_df['Metric'] == metric]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Colors for each optimizer
        colors = {'AMGD': '#3498db', 'AdaGrad': '#2ecc71', 'Adam': '#e74c3c', 'GLMnet': '#9b59b6'}
        bar_colors = [colors[opt] for opt in metric_df['Optimizer']]
        
        # Creating barplot
        bars = ax.bar(metric_df['Optimizer'], metric_df['Value'], color=bar_colors)
        
        # Adding value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Adding regularization and lambda information below bars
        for i, (_, row) in enumerate(metric_df.iterrows()):
            ax.text(i, 0, f"{row['Regularization']}\nλ={row['Lambda']:.4f}", 
                   ha='center', va='bottom', fontsize=8, color='black',
                   transform=ax.get_xaxis_transform())
        
        # Set title and labels
        ax.set_title(f'Best {metric} Comparison Across Optimizers (L1/ElasticNet)', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_xlabel('Optimizer', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        
        plt.ylim(0, metric_df['Value'].max() * 1.15)
        plt.tight_layout()
        
        figs.append(fig)
    
    return figs

# Function to compare convergence rates
def compare_convergence_rates(X_train, y_train, best_params):
    """
    Compare convergence rates of optimization algorithms
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target values
    best_params : dict
        Dictionary with best parameters for each optimizer
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Plot comparing convergence rates of all optimizers
    """
    print("Comparing convergence rates of optimization algorithms...")
    
    # Include all optimizers
    optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']
    optimizer_functions = {
        'AMGD': amgd, 
        'Adam': adam, 
        'AdaGrad': adagrad, 
        'GLMnet': glmnet
    }
    colors = {
        'AMGD': '#3498db',  # Blue
        'Adam': '#e74c3c',  # Red
        'AdaGrad': '#2ecc71',  # Green
        'GLMnet': '#9b59b6'  # Purple
    }
    linestyles = {
        'AMGD': '-', 
        'Adam': '--', 
        'AdaGrad': '-.', 
        'GLMnet': ':'
    }
    
    # Store loss histories
    all_loss_histories = {}
    
    for optimizer_name in optimizers:
        # Get best parameters for MAE (or RMSE)
        params = best_params[f'{optimizer_name}_MAE']
        reg_type = params['Regularization']
        lambda_val = params['Lambda']
        
        # Setup base parameters
        if optimizer_name == "AMGD":
            base_params = {
                "alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "T": 20.0, 
                "tol": 1e-6, "max_iter": 1000, "eta": 0.0001, "epsilon": 1e-8
            }
        elif optimizer_name == "Adam":
            base_params = {
                "alpha": 0.01, "beta1": 0.9, "beta2": 0.999, 
                "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8
            }
        elif optimizer_name == 'GLMnet':
            base_params = {
                "alpha": 0.01, "tol": 1e-6, "max_iter": 1000, 
                "epsilon": 1e-8, "is_pre_scaled": True
            }
        else:  # AdaGrad
            base_params = {
                "alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8
            }
        
        # Configure regularization parameters
        if reg_type == "L1":
            base_params["lambda1"] = lambda_val
            base_params["lambda2"] = 0.0
            base_params["penalty"] = "l1"
        else:  # ElasticNet
            base_params["lambda1"] = lambda_val / 2
            base_params["lambda2"] = lambda_val / 2
            base_params["penalty"] = "elasticnet"
        
        # Run optimizer and track loss history
        _, loss_history, _, _ = optimizer_functions[optimizer_name](
            X_train, y_train, **base_params, verbose=False, return_iters=False
        )
        
        # Only store non-empty loss histories
        if len(loss_history) > 0:
            all_loss_histories[optimizer_name] = loss_history
        else:
            print(f"Warning: {optimizer_name} returned an empty loss history. Skipping in convergence plot.")
    
    # Check if we have any valid loss histories to plot
    if not all_loss_histories:
        print("Error: No valid loss histories to plot. Check that at least one optimizer returns non-empty loss history.")
        # Return an empty figure 
        return plt.figure(figsize=(15, 10))
    
    # Create comprehensive convergence plot
    plt.figure(figsize=(15, 10))
    
    # Main convergence plot (log scale)
    plt.subplot(2, 1, 1)
    for optimizer_name, loss_history in all_loss_histories.items():
        # Calculate percentage of max iterations (normalization)
        iterations = np.linspace(0, 100, len(loss_history))
        
        # Plot with log scale for y-axis
        plt.semilogy(
            iterations, 
            loss_history, 
            label=optimizer_name, 
            color=colors[optimizer_name], 
            linestyle=linestyles[optimizer_name], 
            linewidth=2
        )
    
    plt.title('Convergence Rate Comparison (Log Scale)', fontsize=14)
    plt.xlabel('Percentage of Max Iterations (%)', fontsize=13)
    plt.ylabel('Loss (log scale)', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Optimizer', loc='best')
    
    # Normalized convergence plot (if we have multiple valid histories)
    if len(all_loss_histories) > 1:
        plt.subplot(2, 1, 2)
        max_lengths = max(len(loss_history) for loss_history in all_loss_histories.values())
        
        for optimizer_name, loss_history in all_loss_histories.items():
            # Skip empty loss histories
            if len(loss_history) == 0:
                continue
                
            # Normalize loss history to same length
            if len(loss_history) < max_lengths:
                # Interpolate to match max length
                try:
                    x_new = np.linspace(0, 1, max_lengths)
                    x_old = np.linspace(0, 1, len(loss_history))
                    normalized_loss = np.interp(x_new, x_old, loss_history)
                except ValueError as e:
                    print(f"Error interpolating {optimizer_name} loss history: {e}")
                    continue
            else:
                normalized_loss = loss_history
            
            plt.plot(
                np.linspace(0, 100, len(normalized_loss)), 
                normalized_loss, 
                label=optimizer_name, 
                color=colors[optimizer_name], 
                linestyle=linestyles[optimizer_name], 
                linewidth=2
            )
        
        plt.title('Normalized Convergence Rate Comparison', fontsize=14)
        plt.xlabel('Percentage of Max Iterations (%)', fontsize=13)
        plt.ylabel('Normalized Loss', fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Optimizer', loc='best')
    
    plt.tight_layout()
    
    return plt.gcf()



def plot_coefficient_paths_for_ecological_data():
    """
    Plot coefficient paths for different optimizers using the ecological dataset
    """
    print("Analyzing coefficient paths for ecological dataset...")
    
    # 1. Load and preprocess the data
    X, y, feature_names = preprocess_ecological_dataset("ecological_health_dataset.csv")
    
    # 2. Split data into train, validation, and test sets (70/15/15)
    # First split: 85% train+val, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Second split: 70% train, 15% validation (82.35% of train_val is train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]:.1%})")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]:.1%})")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 3. Configure lambda values for the regularization path
    lambda_values = np.logspace(-3, 1, 10)  # From 0.001 to 10
    
    # 4. Select only the top most important features for readability
    # Runing a basic model to identify important features
    params = {
        "alpha": 0.01, 
        "beta1": 0.9, 
        "beta2": 0.999, 
        "lambda1": 0.1,
        "lambda2": 0.0,
        "penalty": "l1",
        "T": 20.0, 
        "tol": 1e-6, 
        "max_iter": 200,  # Reduced for quicker execution
        "eta": 0.0001, 
        "epsilon": 1e-8,
        "verbose": False
    }
    
    initial_beta, _, _, _ = amgd(X_train, y_train, **params)
    
    # Finding top 10 features by coefficient magnitude
    importance = np.abs(initial_beta)
    top_indices = np.argsort(importance)[-17:]  # Top 10 features
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # 5. Creating figure for the coefficient paths
    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharex=True)
    fig.suptitle('Coefficient Paths for Biodiversity Prediction: L1/ElasticNet Regularization', fontsize=16)
    
    # Configure plot settings
    optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet'] 
    penalty_types = ['l1', 'elasticnet']
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_indices)))
    
    # 6. Plot coefficient paths for each optimizer and regularization type
    for i, optimizer_name in enumerate(optimizers):
        for j, penalty in enumerate(penalty_types):
            ax = axes[i, j]
            
            # Storage for coefficient values at each lambda
            coef_paths = []
            
            # Running optimization for each lambda value
            for lambda_val in lambda_values:
                if optimizer_name == 'AMGD':
                    params = {
                        "alpha": 0.01, 
                        "beta1": 0.9, 
                        "beta2": 0.999, 
                        "T": 20.0, 
                        "tol": 1e-6, 
                        "max_iter": 200,  
                        "eta": 0.0001, 
                        "epsilon": 1e-8,
                        "lambda1": lambda_val if penalty == 'l1' else lambda_val/2,
                        "lambda2": 0.0 if penalty == 'l1' else lambda_val/2,
                        "penalty": penalty,
                        "verbose": False
                    }
                    beta, _, _, _ = amgd(X_train, y_train, **params)
                    
                elif optimizer_name == 'Adam':
                    params = {
                        "alpha": 0.01, 
                        "beta1": 0.9, 
                        "beta2": 0.999, 
                        "tol": 1e-6, 
                        "max_iter": 200,  
                        "epsilon": 1e-8,
                        "lambda1": lambda_val if penalty == 'l1' else lambda_val/2,
                        "lambda2": 0.0 if penalty == 'l1' else lambda_val/2,
                        "penalty": penalty,
                        "verbose": False
                    }
                    beta, _, _, _ = adam(X_train, y_train, **params)
                    
                elif optimizer_name == 'AdaGrad':
                    params = {
                        "alpha": 0.01, 
                        "tol": 1e-6, 
                        "max_iter": 200,  
                        "epsilon": 1e-8,
                        "lambda1": lambda_val if penalty == 'l1' else lambda_val/2,
                        "lambda2": 0.0 if penalty == 'l1' else lambda_val/2,
                        "penalty": penalty,
                        "verbose": False
                    }
                    beta, _, _, _ = adagrad(X_train, y_train, **params)
                
                else:  # GLMnet
                    params = {
                        "alpha": 0.01,
                        "lambda1": lambda_val if penalty == 'l1' else lambda_val/2,
                        "lambda2": 0.0 if penalty == 'l1' else lambda_val/2,
                        "penalty": penalty,
                        "tol": 1e-6,
                        "max_iter": 200,
                        "epsilon": 1e-8,
                        "is_pre_scaled": False,
                        "verbose": False
                    }
                    try:
                        # unpacking 5 values using only the first (beta)
                        beta, _, _, _, _ = glmnet(X_train, y_train, **params, return_iters=True)
                    except ValueError:
                        # If only 4 values returned
                        beta, _, _, _ = glmnet(X_train, y_train, **params, return_iters=False)
                
                # Extracting coefficients for the top features only
                selected_coeffs = [beta[idx] for idx in top_indices]
                coef_paths.append(selected_coeffs)
            
            # Converting to numpy array for easier manipulation
            coef_paths = np.array(coef_paths)
            
            # Plotting coefficient paths for top features
            for idx, feature_idx in enumerate(range(len(top_indices))):
                ax.plot(lambda_values, coef_paths[:, idx], 
                        color=colors[idx], 
                        label=top_feature_names[idx],
                        linewidth=2)
            
            # labels and title
            ax.set_xscale('log')
            ax.set_xlabel('Regularization Strength (λ)' if i == 3 else '')  
            ax.set_ylabel('Coefficient Value' if j == 0 else '')
            ax.set_title(f'{optimizer_name} - {penalty.capitalize()} Regularization')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # legend 
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.08),
               title='Features', ncol=5, frameon=True)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    plt.savefig('ecological_coefficient_paths.png', dpi=300, bbox_inches='tight')
    plt.show()

    #plot_coefficient_evolution_for_ecological_data(X_train, y_train, top_indices, top_feature_names)


def plot_training_and_test_metrics(model_results, test_metrics, metric_to_plot='MAE'):
    """
    Plot training and test metrics across optimizers.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing all optimizer models and their results
    test_metrics : dict
        Dictionary containing test metrics for all optimizer models
    metric_to_plot : str
        Metric to plot ('MAE', 'RMSE', 'Mean Deviance')
    
    Returns:
    --------
    None (displays plots)
    """
    
    
    plt.figure(figsize=(15, 10))
    
    # 1. Plot loss histories (convergence) for all optimizers
    plt.subplot(2, 2, 1)
    
    colors = {'AMGD': '#3498db', 'Adam': '#e74c3c', 'AdaGrad': '#2ecc71', 'GLMnet': '#9b59b6'}
    
    for optimizer_name, results in model_results.items():
        loss_history = results['loss_history']
        iterations = np.arange(1, len(loss_history) + 1)
        plt.semilogy(iterations, loss_history, label=f"{optimizer_name}", color=colors[optimizer_name], linewidth=2)
    
    plt.xlabel('Iterations', fontsize=13)
    plt.ylabel('Loss (log scale)', fontsize=13)
    plt.title('Training Loss Convergence', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # 2. Plot training metrics for all optimizers
    plt.subplot(2, 2, 2)
    optimizer_names = list(model_results.keys())
    train_metrics = [model_results[opt]['train_metrics'][metric_to_plot] for opt in optimizer_names]
    
    # Bar chart for training metrics
    bars = plt.bar(optimizer_names, train_metrics, color=[colors[opt] for opt in optimizer_names])

    #Value label
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylabel(f'Training {metric_to_plot}', fontsize=12)
    plt.title(f'Training {metric_to_plot} by Optimizer', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Plotting test metrics for all optimizers
    plt.subplot(2, 2, 3)
    test_metric_values = [test_metrics[opt][metric_to_plot] for opt in optimizer_names]
    
    # Bar chart for test metrics
    bars = plt.bar(optimizer_names, test_metric_values, color=[colors[opt] for opt in optimizer_names])
    
    # Value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylabel(f'Test {metric_to_plot}', fontsize=12)
    plt.title(f'Test {metric_to_plot} by Optimizer', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Plotting train vs test comparison
    plt.subplot(2, 2, 4)
    x = np.arange(len(optimizer_names))
    width = 0.35
    
    #Grouped bar chart
    bars1 = plt.bar(x - width/2, train_metrics, width, label=f'Training {metric_to_plot}', alpha=0.7)
    bars2 = plt.bar(x + width/2, test_metric_values, width, label=f'Test {metric_to_plot}', alpha=0.7)
    
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel(metric_to_plot, fontsize=12)
    plt.title(f'Training vs Test {metric_to_plot} Comparison', fontsize=14)
    plt.xticks(x, optimizer_names)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # figure for the feature importance comparison
    plt.figure(figsize=(16, 6))
    
    # Subplots for each optimizer's feature importance 
    for i, optimizer_name in enumerate(optimizer_names, 1):
        plt.subplot(1, len(optimizer_names), i)
        
        #Beta coefficients for this optimizer
        beta = model_results[optimizer_name]['beta']
        
        # Calculating feature importance based on absolute coefficient values
        importance = np.abs(beta)
        indices = np.argsort(importance)[::-1]
        
        #Top N features
        top_n = min(10, len(importance))
        
        # Get feature names 
        feature_indices = indices[:top_n]
        feature_labels = [f"Feature {idx}" for idx in feature_indices]  # Replace with actual feature names if available
        
        # Plot importance
        plt.barh(range(top_n), importance[feature_indices], align='center', color=colors[optimizer_name])
        plt.yticks(range(top_n), feature_labels)
        plt.xlabel('Coefficient Magnitude')
        plt.title(f'{optimizer_name} Feature Importance')
        
    plt.tight_layout()
    plt.show()

    #Non-zero features evolution plot
    plt.figure(figsize=(12, 6))
    
    for optimizer_name, results in model_results.items():
        nonzero_history = results['nonzero_history']
        iterations = np.arange(1, len(nonzero_history) + 1)
        plt.plot(iterations, nonzero_history, label=f"{optimizer_name}", color=colors[optimizer_name], linewidth=2)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Number of Non-Zero Coefficients', fontsize=12)
    plt.title('Sparsity Evolution During Training', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_optimizer_comparison(model_results, test_metrics, metrics_to_compare=None):
    """
    Plot a comprehensive comparison of all optimizer performances.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing all optimizer models and their results
    test_metrics : dict
        Dictionary containing test metrics for all optimizer models
    metrics_to_compare : list, optional
        List of metrics to compare ('MAE', 'RMSE', 'Mean Deviance', 'Sparsity')
    
    Returns:
    --------
    None (displays plots)
    """
    
    
    if metrics_to_compare is None:
        metrics_to_compare = ['MAE', 'RMSE', 'Mean Deviance', 'Sparsity']
    
    optimizer_names = list(model_results.keys())
    
    # Preparing data for radar chart
    data = []
    for optimizer in optimizer_names:
        row = []
        for metric in metrics_to_compare:
            if metric in test_metrics[optimizer]:
                # For all metrics except Sparsity, lower is better
                if metric != 'Sparsity':
                    row.append(test_metrics[optimizer][metric])
                else:
                    # For Sparsity, higher is better (inverted)
                    row.append(1 - test_metrics[optimizer][metric])
        data.append(row)
    
    # Converting to numpy array
    data = np.array(data)
    
    # Normalize the data between 0 and 1 for radar chart
    data_normalized = np.zeros_like(data, dtype=float)
    for i in range(len(metrics_to_compare)):
        if metrics_to_compare[i] != 'Sparsity':
            # For error metrics, smaller is better
            data_normalized[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]) + 1e-10)
        else:
            # For sparsity===> inverted, so smaller is better
            data_normalized[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]) + 1e-10)
    
    # Number of variables
    N = len(metrics_to_compare)
    
    # figure for the radar chart
    plt.figure(figsize=(10, 10))
    
    # Plotting the radar chart
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    ax = plt.subplot(111, polar=True)
    
    # Adding variable labels
    plt.xticks(angles[:-1], metrics_to_compare, size=12)
    
    # y-labels (percentages)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # colors
    colors = {'AMGD': '#3498db', 'Adam': '#e74c3c', 'AdaGrad': '#2ecc71', 'GLMnet': '#9b59b6'}
    for i, optimizer in enumerate(optimizer_names):
        values = data_normalized[i].tolist()
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=optimizer, color=colors[optimizer])
        ax.fill(angles, values, alpha=0.1, color=colors[optimizer])
    
    #Legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Optimizer Performance Comparison\n(Closer to center is better)", size=15)
    
    plt.tight_layout()
    plt.show()
    
    # summary table for runtime comparison
    runtimes = [model_results[opt]['runtime'] for opt in optimizer_names]
    iterations = [len(model_results[opt]['loss_history']) for opt in optimizer_names]
    
    plt.figure(figsize=(12, 6))
    
    # Runtime comparison
    plt.subplot(1, 2, 1)
    
    bar_colors = [colors[opt] for opt in optimizer_names]
    bars = plt.bar(optimizer_names, runtimes, color=bar_colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Total Runtime Comparison', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Iterations comparison
    plt.subplot(1, 2, 2)
    bars = plt.bar(optimizer_names, iterations, color=bar_colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height}', ha='center', va='bottom', fontsize=10)
    plt.ylabel('Number of Iterations', fontsize=12)
    plt.title('Convergence Iterations Comparison', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
