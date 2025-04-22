
# Main function to run pipeline 
def run_poisson_regression_pipeline(filepath="ecological_health_dataset.csv", 
                                   random_state=42, k_folds=5,
                                   lambda_values=None, 
                                   metric_to_optimize='MAE',
                                   plot_coefficients=True,
                                   plot_metrics=True,
                                   optimizers=None):
    """
    Run the Poisson regression pipeline with L1 and ElasticNet regularization only:
    1. Load and preprocess data
    2. Split data into train, validation, and test sets
    3. Find optimal parameters using k-fold cross-validation on validation set
    4. Create comparison barplots for the algorithms
    5. Train model on training set using best parameters
    6. Compare convergence rates of the algorithms
    7. Evaluate model on test set
    8. Plot coefficient paths and evolution (if enabled)
    9. Plot comprehensive metric visualizations (if enabled)
    
    Parameters:
    -----------
    filepath : str
        Path to the dataset CSV file
    random_state : int
        Random seed for reproducibility
    k_folds : int
        Number of folds for cross-validation
    lambda_values : list, optional
        List of lambda values to try
    metric_to_optimize : str
        Metric to use for selecting best parameters ('MAE', 'RMSE', 'Mean_Deviance', 'Runtime')
    plot_coefficients : bool
        Whether to generate coefficient path and evolution plots
    plot_metrics : bool
        Whether to generate comprehensive metric visualizations
    optimizers : list, optional
        List of optimizers to use ('AMGD', 'Adam', 'AdaGrad', 'GLMnet'). If None, all optimizers will be used.
    
    Returns:
    --------
    best_params : dict
        Best parameters found through cross-validation
    test_metrics : dict
        Evaluation metrics on test set
    model_results : dict
        Results for all models including coefficients and performance
    """
    print("=" * 80)
    print("POISSON REGRESSION PIPELINE (L1 AND ELASTICNET)")
    print("=" * 80)
    
    # Set default optimizers if not provided
    if optimizers is None:
        optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']
    
    # Validating optimizers
    valid_optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']
    for opt in optimizers:
        if opt not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {opt}. Valid options are: {valid_optimizers}")
    
    # 1. Load and preprocess data
    print("\nStep 1: Loading and preprocessing data")
    X, y, feature_names = preprocess_ecological_dataset(filepath)
    
    # 2. Splitting data into train, validation, and test sets (70/15/15)
    print("\nStep 2: Splitting data into train, validation, and test sets (70/15/15)")
    # First split: 85% train+val, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state
    )
    
    # Second split: 70% train, 15% validation (82.35% of train_val is train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]:.1%})")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]:.1%})")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 3. Find optimal parameters using k-fold cross-validation on validation set
    print(f"\nStep 3: Finding optimal parameters using {k_folds}-fold cross-validation on validation set")
    print(f"Regularization with L1 and ElasticNet using optimizers: {optimizers}")
    
    if lambda_values is None:
        lambda_values = np.logspace(-4, np.log10(20), 20)  

        #lambda_values = np.logspace(-4, 1, 6)  # [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        print(f"Using lambda values: {lambda_values}")
    
    # optimizers_to_use
    best_params, cv_results_df = k_fold_cross_validation(
        X_val, y_val, k=k_folds, lambda_values=lambda_values, seed=random_state,
        optimizers_to_use=optimizers  # Fixed: optimizers -> optimizers_to_use
    )
    
    # Print best parameters
    print("\nBest parameters found through cross-validation:")
    for metric, params in best_params.items():
        if metric.startswith('Overall_Best_'):
            metric_name = metric.replace('Overall_Best_', '')
            print(f"Best for {metric_name}: {params['Optimizer']} with {params['Regularization']} (λ={params['Lambda']:.6f}), Value: {params['Metric_Value']:.6f}")
    
    # 4.  comparison barplots for the algorithms
    print("\nStep 4: Creating algorithm performance comparison plots")
    comparison_plots = create_algorithm_comparison_plots(cv_results_df)
    
    # Display comparison plots
    for fig in comparison_plots:
        plt.figure(fig.number)
        plt.show()
    
    # 5. Train model on training set using best parameters
    print(f"\nStep 5: Training model on training set using best parameters for {metric_to_optimize}")
    model_results = train_all_optimizers(
        X_train, y_train, best_params, metric=metric_to_optimize
    )
    
    # Selecting the best optimizer based on the chosen metric
    best_optimizer = best_params[f'Overall_Best_{metric_to_optimize}']['Optimizer']
    print(f"Using {best_optimizer} model for test evaluation (best for {metric_to_optimize})")
    
    # 6. Comparing convergence rates of the algorithms
    print("\nStep 6: Comparing convergence rates of optimization algorithms")
    # Check if compare_convergence_rates accepts an optimizers parameter
    try:
        convergence_plot = compare_convergence_rates(
            X_train, y_train, best_params
        )
    except TypeError:
        # If the function doesn't accept an optimizers parameter, catch the error
        print("Note: Using all optimizers for convergence rate comparison")
        convergence_plot = compare_convergence_rates(
            X_train, y_train, best_params
        )
    
    plt.figure(convergence_plot.number)
    plt.show()
    
    # 7. Evaluating all models on test set
    print("\nStep 7: Evaluating all models on test set")
    
    test_metrics = {}
    
    # Evaluating each optimizer with its best parameters
    for optimizer_name, results in model_results.items():
        # Skip optimizers that weren't requested
        if optimizer_name not in optimizers:
            continue
            
        beta = results['beta']
        metrics = evaluate_model(beta, X_test, y_test, target_name=optimizer_name)
        test_metrics[optimizer_name] = metrics
        
        print(f"\nTest metrics for {optimizer_name}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
                
    # Creating a comparison table
    comparison_metrics = ['MAE', 'RMSE', 'Mean Deviance', 'Sparsity']
    comparison_data = []
    
    for optimizer, metrics in test_metrics.items():
        row = [optimizer]
        for metric in comparison_metrics:
            row.append(metrics[metric])
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data, columns=['Optimizer'] + comparison_metrics)
    print("\nModel comparison on test set:")
    print(comparison_df)
    
    # Highlight the best model according to the chosen metric
    best_idx = comparison_df[metric_to_optimize if metric_to_optimize in comparison_metrics else 'MAE'].idxmin()
    best_test_optimizer = comparison_df.iloc[best_idx]['Optimizer']
    print(f"\nBest model on test set for {metric_to_optimize}: {best_test_optimizer}")
    
    # Visualizing feature importance for the best model
    best_beta = model_results[best_test_optimizer]['beta']
    
    top_n = 15
    importance = np.abs(best_beta)
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances ({best_test_optimizer} model)')
    plt.bar(range(min(top_n, len(feature_names))), 
           importance[indices[:top_n]], 
           align='center')
    plt.xticks(range(min(top_n, len(feature_names))), 
               [feature_names[i] for i in indices[:top_n]], 
               rotation=90)
    plt.tight_layout()
    plt.show()
    
    # 8. Plot coefficient paths and evolution 
    if plot_coefficients:
        print("\nStep 8: Plotting coefficient paths and evolution")
        # Get top indices for plotting
        top_indices = indices[:17]  # Top 17 features
        top_feature_names = [feature_names[i] for i in top_indices]
        
        # Plot coefficient paths
        print("Plotting coefficient paths...")
        try:
            # try with the optimizers parameter
            plot_coefficient_paths_for_ecological_data()
        except Exception as e:
            print(f"Note: Could not use optimizers parameter for plotting. Using all optimizers. Error: {e}")
            plot_coefficient_paths_for_ecological_data()
    
    # 9. Plot comprehensive metric visualizations 
    if plot_metrics:
        print("\nStep 9: Creating comprehensive metric visualizations")
        
        # Add feature names to model_results for plotting
        for optimizer_name in model_results:
            model_results[optimizer_name]['feature_names'] = feature_names
        
        # Plot training and test metrics
        print("Plotting training and test metrics comparison...")
        plot_training_and_test_metrics(model_results, test_metrics, metric_to_plot=metric_to_optimize)
        
        # Plot radar chart and performance comparison
        print("Plotting optimizer performance comparison...")
        plot_optimizer_comparison(model_results, test_metrics)

      # 8. Statistical Significance Analysis
        print("\nStep 8: Performing Statistical Significance Analysis")
        significance_results = add_to_pipeline(
        X, y, 
        best_params, 
        feature_names, 
        model_results
        )
    
    print("\nPipeline completed successfully!")
    
    # Return best parameters, test metrics for all models, and all model results
    return best_params, test_metrics, model_results, significance_results



# execute the pipeline
if __name__ == "__main__":
    
    #lambda_values = np.logspace(-4, 1, 50)  # Creates 50 values between 10^-4 and 10^1
    lambda_values = np.logspace(-4, np.log10(20), 50)  
    
    try:
        best_params, test_metrics, model_results, significance_results = run_poisson_regression_pipeline(
        filepath="ecological_health_dataset.csv",
        random_state=42,
        k_folds=5,
        lambda_values=lambda_values,
        metric_to_optimize='MAE'
                                )
        
        
        print("\nAnalysis complete. Summary of test metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found.")
        print("Please ensure the dataset file is in the correct location.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
