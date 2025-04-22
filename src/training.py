#Function to train all optimizers

def train_all_optimizers(X_train, y_train, best_params, metric='MAE'):
    """
    Train all optimizer models using their best parameters from cross-validation
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target values
    best_params : dict
        Dictionary with best parameters for each optimizer and metric
    metric : str
        Metric to use for selecting the best parameters ('MAE', 'RMSE', 'Mean_Deviance', 'Runtime', 'Sparsity')
    
    Returns:
    --------
    model_results : dict
        Dictionary with trained models, their coefficients, and performance metrics
    """
    optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']
    model_results = {}
    
    for optimizer_name in optimizers:
        # Get best parameters for this optimizer and metric
        params = best_params[f"{optimizer_name}_{metric}"]
        
        reg_type = params['Regularization']
        lambda_val = params['Lambda']
        
        print(f"\nTraining {optimizer_name} with best parameters for {metric}:")
        print(f"  Regularization: {reg_type}")
        print(f"  Lambda: {lambda_val}")
        
        # Setting uptimizer function and parameters
        if optimizer_name == "AMGD":
            optimizer_func = amgd
            base_params = {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "T": 20.0, 
                          "tol": 1e-6, "max_iter": 1000, "eta": 0.0001, "epsilon": 1e-8}
        elif optimizer_name == "Adam":
            optimizer_func = adam
            base_params = {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, 
                          "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        elif optimizer_name == "AdaGrad":
            optimizer_func = adagrad
            base_params = {"alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        else:  # GLMnet
            optimizer_func = glmnet
            base_params = {"alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        
        # Configure regularization parameters
        if reg_type == "L1":
            base_params["lambda1"] = lambda_val
            base_params["lambda2"] = 0.0
            base_params["penalty"] = "l1"
        else:  # ElasticNet
            base_params["lambda1"] = lambda_val / 2
            base_params["lambda2"] = lambda_val / 2
            base_params["penalty"] = "elasticnet"
        
        # Train the model with tracking of per-iteration values
        try:
            # Try for GLMnet which returns 6 values when return_iters=True
            if optimizer_name == "GLMnet":
                beta, loss_history, runtime, nonzero_history, beta_history, lr_history = optimizer_func(
                    X_train, y_train, **base_params, verbose=True, return_iters=True
                )
            else:
                # For AMGD, Adam, and AdaGrad which return 5 values
                beta, loss_history, runtime, nonzero_history, beta_history = optimizer_func(
                    X_train, y_train, **base_params, verbose=True, return_iters=True
                )
                lr_history = None  # These optimizers don't return learning rate
        except ValueError as e:
            print(f"Error with {optimizer_name}: {e}")
            print("Trying alternative return value handling...")
            
            # Fallback approach
            result = optimizer_func(X_train, y_train, **base_params, verbose=True, return_iters=True)
            
            # Extracting values based on the length of the result tuple
            if len(result) == 6:  # GLMnet with lr_history
                beta, loss_history, runtime, nonzero_history, beta_history, lr_history = result
            elif len(result) == 5:  # AMGD, Adam, AdaGrad
                beta, loss_history, runtime, nonzero_history, beta_history = result
                lr_history = None
            else:
                # Handle any other unexpected return value count
                beta = result[0]
                loss_history = result[1] if len(result) > 1 else []
                runtime = result[2] if len(result) > 2 else 0
                nonzero_history = result[3] if len(result) > 3 else []
                beta_history = result[4] if len(result) > 4 else []
                lr_history = result[5] if len(result) > 5 else None
        
        # Evaluating model on training set
        train_metrics = evaluate_model(beta, X_train, y_train)
        
        # Storing results
        model_results[optimizer_name] = {
            'beta': beta,
            'loss_history': loss_history,
            'runtime': runtime,
            'nonzero_history': nonzero_history,
            'beta_history': beta_history,
            'lr_history': lr_history,
            'train_metrics': train_metrics,
            'params': base_params
        }
        
        print(f"  Training complete in {runtime:.2f} seconds")
        print(f"  Training MAE: {train_metrics['MAE']:.4f}")
        print(f"  Training RMSE: {train_metrics['RMSE']:.4f}")
        print(f"  Training Mean Deviance: {train_metrics['Mean Deviance']:.4f}")
        print(f"  Sparsity: {train_metrics['Sparsity']:.4f}")
    
    return model_results
