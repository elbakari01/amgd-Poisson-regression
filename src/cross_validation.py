    
## Cross validation

def k_fold_cross_validation(X_val, y_val, k=5, lambda_values=None, seed=42, optimizers_to_use=None):
    """
    Perform k-fold cross-validation to find optimal parameters for selected optimizers
    and regularization types
    
    Parameters:
    -----------
    X_val : numpy.ndarray
        Validation feature matrix
    y_val : numpy.ndarray
        Validation target values
    k : int
        Number of folds for cross-validation
    lambda_values : list
        List of lambda values to try
    seed : int
        Random seed for reproducibility
    optimizers_to_use : list or None
        List of optimizer names to evaluate. If None, use all available optimizers.
    
    Returns:
    --------
    best_params : dict
        Dictionary with best parameters for each optimizer and metric
    cv_results : pd.DataFrame
        DataFrame with all cross-validation results
    """
    if lambda_values is None:
        #lambda_values = list(np.logspace(-4, 1, 50))
        lambda_values = np.logspace(-4, np.log10(20), 50)

    # Defining all available optimizers with their base configurations
    all_optimizers = {
        "AMGD": {
            "func": amgd,
            "base_params": {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "T": 20.0, 
                          "tol": 1e-6, "max_iter": 1000, "eta": 0.0001, "epsilon": 1e-8}
        },
        "Adam": {
            "func": adam,
            "base_params": {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, 
                          "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        },
        "AdaGrad": {
            "func": adagrad,
            "base_params": {"alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        },
        "GLMnet": {
            "func": glmnet,
            "base_params": {"alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        }
    }
    
    # Filter optimizers
    if optimizers_to_use is not None:
        optimizers = {name: config for name, config in all_optimizers.items() 
                     if name in optimizers_to_use}
    else:
        optimizers = all_optimizers
    
    # Regularization types
    regularizations = ["L1", "ElasticNet"]
    
    # Set up k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    # Store results
    cv_results = []
    
    # Loop through all combinations
    for optimizer_name, optimizer_info in optimizers.items():
        for reg_type in regularizations:
            for lambda_val in lambda_values:
                print(f"Evaluating {optimizer_name} with {reg_type} regularization, lambda={lambda_val}")
                
                # Prepare parameters based on regularization type
                params = optimizer_info["base_params"].copy()
                
                if reg_type == "L1":
                    params["lambda1"] = lambda_val
                    params["lambda2"] = 0.0
                    params["penalty"] = "l1"
                elif reg_type == "ElasticNet":
                    params["lambda1"] = lambda_val / 2  # Distribute lambda between L1 and L2
                    params["lambda2"] = lambda_val / 2
                    params["penalty"] = "elasticnet"
                
                # Metrics for this combination across all folds
                fold_maes = []
                fold_rmses = []
                fold_deviances = []
                fold_runtimes = []
                fold_sparsities = []
                
                # Run k-fold cross-validation
                for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_val)):
                    X_fold_train, X_fold_test = X_val[train_idx], X_val[test_idx]
                    y_fold_train, y_fold_test = y_val[train_idx], y_val[test_idx]
                    
                    # Train the model on this fold - handle different return value counts
                    try:
                        # gett all 5 return values
                        beta, loss_history, runtime, nonzero_history, _ = optimizer_info["func"](
                            X_fold_train, y_fold_train, **params, verbose=False, return_iters=False
                        )
                    except ValueError:
                        
                        beta, loss_history, runtime, nonzero_history = optimizer_info["func"](
                            X_fold_train, y_fold_train, **params, verbose=False, return_iters=False
                        )
                    
                    # Evaluating on the hold-out fold
                    metrics = evaluate_model(beta, X_fold_test, y_fold_test)
                    
                    # Store metrics
                    fold_maes.append(metrics['MAE'])
                    fold_rmses.append(metrics['RMSE'])
                    fold_deviances.append(metrics['Mean Deviance'])
                    fold_runtimes.append(runtime)
                    fold_sparsities.append(metrics['Sparsity'])
                
                # Calculating average metrics across folds
                avg_mae = np.mean(fold_maes)
                avg_rmse = np.mean(fold_rmses)
                avg_deviance = np.mean(fold_deviances)
                avg_runtime = np.mean(fold_runtimes)
                avg_sparsity = np.mean(fold_sparsities)
                
                # Calculating standard deviations
                std_mae = np.std(fold_maes)
                std_rmse = np.std(fold_rmses)
                std_deviance = np.std(fold_deviances)
                
                # Storing the results
                result = {
                    "Optimizer": optimizer_name,
                    "Regularization": reg_type,
                    "Lambda": lambda_val,
                    "MAE": avg_mae,
                    "MAE_std": std_mae,
                    "RMSE": avg_rmse,
                    "RMSE_std": std_rmse,
                    "Mean Deviance": avg_deviance,
                    "Mean Deviance_std": std_deviance,
                    "Runtime": avg_runtime,
                    "Sparsity": avg_sparsity
                }
                
                cv_results.append(result)
    
    # Converting results to DataFrame
    cv_results_df = pd.DataFrame(cv_results)
    
    # Finding best parameters for each metric
    best_params = {}
    
    for optimizer_name in optimizers.keys():
        optimizer_results = cv_results_df[cv_results_df['Optimizer'] == optimizer_name]
        
        # Finding best parameters for MAE
        best_mae_idx = optimizer_results['MAE'].idxmin()
        best_params[f"{optimizer_name}_MAE"] = {
            "Optimizer": optimizer_name,
            "Regularization": optimizer_results.loc[best_mae_idx, 'Regularization'],
            "Lambda": optimizer_results.loc[best_mae_idx, 'Lambda'],
            "Metric_Value": optimizer_results.loc[best_mae_idx, 'MAE']
        }
        
        # Finding best parameters for RMSE
        best_rmse_idx = optimizer_results['RMSE'].idxmin()
        best_params[f"{optimizer_name}_RMSE"] = {
            "Optimizer": optimizer_name,
            "Regularization": optimizer_results.loc[best_rmse_idx, 'Regularization'],
            "Lambda": optimizer_results.loc[best_rmse_idx, 'Lambda'],
            "Metric_Value": optimizer_results.loc[best_rmse_idx, 'RMSE']
        }
        
        # Finding best parameters for Mean Deviance
        best_dev_idx = optimizer_results['Mean Deviance'].idxmin()
        best_params[f"{optimizer_name}_Mean_Deviance"] = {
            "Optimizer": optimizer_name,
            "Regularization": optimizer_results.loc[best_dev_idx, 'Regularization'],
            "Lambda": optimizer_results.loc[best_dev_idx, 'Lambda'],
            "Metric_Value": optimizer_results.loc[best_dev_idx, 'Mean Deviance']
        }
        
        # Finding best parameters for Runtime (fastest)
        best_runtime_idx = optimizer_results['Runtime'].idxmin()
        best_params[f"{optimizer_name}_Runtime"] = {
            "Optimizer": optimizer_name,
            "Regularization": optimizer_results.loc[best_runtime_idx, 'Regularization'],
            "Lambda": optimizer_results.loc[best_runtime_idx, 'Lambda'],
            "Metric_Value": optimizer_results.loc[best_runtime_idx, 'Runtime']
        }
        
        # Finding best parameters for Sparsity
        best_sparsity_idx = optimizer_results['Sparsity'].idxmax()
        best_params[f"{optimizer_name}_Sparsity"] = {
            "Optimizer": optimizer_name,
            "Regularization": optimizer_results.loc[best_sparsity_idx, 'Regularization'],
            "Lambda": optimizer_results.loc[best_sparsity_idx, 'Lambda'],
            "Metric_Value": optimizer_results.loc[best_sparsity_idx, 'Sparsity']
        }
    
    # Finding the overall best parameters
    mae_results = [params for key, params in best_params.items() if key.endswith('_MAE')]
    best_mae_params = min(mae_results, key=lambda x: x['Metric_Value'])
    best_params['Overall_Best_MAE'] = best_mae_params
    
    rmse_results = [params for key, params in best_params.items() if key.endswith('_RMSE')]
    best_rmse_params = min(rmse_results, key=lambda x: x['Metric_Value'])
    best_params['Overall_Best_RMSE'] = best_rmse_params
    
    deviance_results = [params for key, params in best_params.items() if key.endswith('_Mean_Deviance')]
    best_deviance_params = min(deviance_results, key=lambda x: x['Metric_Value'])
    best_params['Overall_Best_Mean_Deviance'] = best_deviance_params
    
    runtime_results = [params for key, params in best_params.items() if key.endswith('_Runtime')]
    best_runtime_params = min(runtime_results, key=lambda x: x['Metric_Value'])
    best_params['Overall_Best_Runtime'] = best_runtime_params
    
    sparsity_results = [params for key, params in best_params.items() if key.endswith('_Sparsity')]
    best_sparsity_params = max(sparsity_results, key=lambda x: x['Metric_Value'])
    best_params['Overall_Best_Sparsity'] = best_sparsity_params
    
    # Printing a summary of best lambda values for each optimizer and metric
    print("\nSummary of best lambda values from cross-validation:")
    print("{:<15} {:<15} {:<15} {:<15} {:<15}".format("Optimizer", "MAE Lambda", "RMSE Lambda", "Deviance Lambda", "Sparsity Lambda"))
    print("-" * 75)
    
    for optimizer in optimizers.keys():
        mae_lambda = best_params[f"{optimizer}_MAE"]["Lambda"]
        rmse_lambda = best_params[f"{optimizer}_RMSE"]["Lambda"]
        dev_lambda = best_params[f"{optimizer}_Mean_Deviance"]["Lambda"]
        spar_lambda = best_params[f"{optimizer}_Sparsity"]["Lambda"]
        
        print("{:<15} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            optimizer, mae_lambda, rmse_lambda, dev_lambda, spar_lambda))
    
    print("\nOverall best parameters:")
    for metric in ["MAE", "RMSE", "Mean_Deviance", "Runtime", "Sparsity"]:
        params = best_params[f"Overall_Best_{metric}"]
        print(f"Best for {metric}: {params['Optimizer']} with {params['Regularization']} (λ={params['Lambda']:.6f})")
    
    return best_params, cv_results_df
