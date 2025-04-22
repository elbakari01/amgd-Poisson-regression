import numpy as np

#AMGD implementation 
def amgd(X, y, alpha=0.001, beta1=0.8, beta2=0.999, 
         lambda1=0.1, lambda2=0.0, penalty='l1',
         T=20.0, tol=1e-6, max_iter=1000, eta=0.0001, epsilon=1e-8, 
         verbose=False, return_iters=False):
    """
    Adaptive Momentum Gradient Descent (AMGD) for single-target Poisson regression
    with L1 or Elastic Net regularization
    """
    n_samples, n_features = X.shape
    
    # Initializing coefficient vector
    beta = np.random.normal(0, 0.1, n_features)
    
    # Initializing momentum variables
    m = np.zeros(n_features)
    v = np.zeros(n_features)
    
    prev_loss = float('inf')
    loss_history = []
    start_time = time.time()
    
    # Tracking non-zero coefficients for debugging
    nonzero_history = []
    
    # Tracking values at each iteration 
    if return_iters:
        beta_history = []
    
    for t in range(1, max_iter + 1):
        alpha_t = alpha / (1 + eta * t)
        
        # Computing predictions and gradient
        linear_pred = X @ beta
        linear_pred = np.clip(linear_pred, -20, 20)
        mu = np.exp(linear_pred)
        
        # Gradient of negative log-likelihood
        grad_ll = X.T @ (mu - y)
        
        # Adding regularization gradient
        if penalty == 'l1':
            # Pure L1: no gradient term (handled in soft thresholding step)
            grad = grad_ll
        elif penalty == 'elasticnet':
            # Elastic Net: add gradient of L2 component
            grad = grad_ll + lambda2 * beta
        else:
            raise ValueError(f"Unknown penalty: {penalty}")
        
        grad = clip(grad, T)
        
        # Momentum updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Parameter update
        beta = beta - alpha_t * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Apply appropriate regularization
        if penalty == 'l1' or penalty == 'elasticnet':
            # Adaptive soft-thresholding for L1 component
            denom = np.abs(beta) + 0.1
            beta = np.sign(beta) * np.maximum(np.abs(beta) - alpha_t * lambda1 / denom, 0)


        
        # Compute loss
        ll = poisson_log_likelihood(beta, X, y)
        
        # Add regularization component to loss
        reg_pen = 0
        if penalty == 'l1':
            reg_pen = lambda1 * np.sum(np.abs(beta))
        elif penalty == 'elasticnet':
            reg_pen = lambda1 * np.sum(np.abs(beta)) + (lambda2 / 2) * np.sum(beta**2)
        
        total_loss = ll + reg_pen
        loss_history.append(total_loss)
        
        # Tracking non-zero coefficients
        non_zeros = np.sum(np.abs(beta) > 1e-6)
        nonzero_history.append(non_zeros)
        
        # Tracking beta values 
        if return_iters:
            beta_history.append(beta.copy())
        
        if verbose and t % 100 == 0:
            print(f"Iteration {t}, Loss: {total_loss:.4f}, Log-likelihood: {ll:.4f}, Penalty: {reg_pen:.4f}")
            print(f"Non-zero coefficients: {non_zeros}/{n_features}, Sparsity: {1-non_zeros/n_features:.4f}")
        
        # Checking convergence
        if abs(prev_loss - total_loss) < tol:
            if verbose:
                print(f"Converged at iteration {t}")
            break
            
        prev_loss = total_loss
    
    runtime = time.time() - start_time
    
    if return_iters:
        return beta, loss_history, runtime, nonzero_history, beta_history
    else:
        return beta, loss_history, runtime, nonzero_history

# AdaGrad implementation with L1 or Elastic Net regularization
def adagrad(X, y, alpha=0.01, lambda1=0.1, lambda2=0.0, penalty='l1',
            tol=1e-6, max_iter=1000, epsilon=1e-8, verbose=False, return_iters=False):
    """
    AdaGrad optimizer for single-target Poisson regression
    with L1 or Elastic Net regularization
    """
    n_samples, n_features = X.shape
    
    # Initialize coefficient vector
    beta = np.random.normal(0, 0.1, n_features)
    
    # Initializing accumulator for squared gradients
    G = np.zeros(n_features)
    
    prev_loss = float('inf')
    loss_history = []
    start_time = time.time()
    
    # Tracking non-zero coefficients
    nonzero_history = []
    
    # Tracking values at each iteration 
    if return_iters:
        beta_history = []
    
    for t in range(1, max_iter + 1):
        # Computing predictions and gradient
        linear_pred = X @ beta
        linear_pred = np.clip(linear_pred, -20, 20)
        mu = np.exp(linear_pred)
        
        # Gradient of negative log-likelihood
        grad_ll = X.T @ (mu - y)
        
        # Add regularization gradient
        if penalty == 'l1':
            # Pure L1: add subgradient of L1 penalty
            grad = grad_ll + lambda1 * np.sign(beta)
        elif penalty == 'elasticnet':
            # Elastic Net: add combined gradient
            grad = grad_ll + lambda1 * np.sign(beta) + lambda2 * beta
        else:
            raise ValueError(f"Unknown penalty: {penalty}")
        
        # Update accumulator
        G += grad ** 2
        
        # Parameter update with AdaGrad scaling
        beta = beta - alpha * grad / (np.sqrt(G) + epsilon)
        
        # Apply proximal operator for L1 regularization 
        if penalty == 'l1' or penalty == 'elasticnet':
            beta = np.sign(beta) * np.maximum(np.abs(beta) - lambda1 * alpha / (np.sqrt(G) + epsilon), 0)
        
        # Compute loss
        ll = poisson_log_likelihood(beta, X, y)
        
        # Add regularization component to loss
        reg_pen = 0
        if penalty == 'l1':
            reg_pen = lambda1 * np.sum(np.abs(beta))
        elif penalty == 'elasticnet':
            reg_pen = lambda1 * np.sum(np.abs(beta)) + (lambda2 / 2) * np.sum(beta**2)
        
        total_loss = ll + reg_pen
        loss_history.append(total_loss)
        
        # Tracking non-zero coefficients
        non_zeros = np.sum(np.abs(beta) > 1e-6)
        nonzero_history.append(non_zeros)
        
        # Tracking beta values 
        if return_iters:
            beta_history.append(beta.copy())
        
        if verbose and t % 100 == 0:
            print(f"Iteration {t}, Loss: {total_loss:.4f}, Log-likelihood: {ll:.4f}, Penalty: {reg_pen:.4f}")
            print(f"Non-zero coefficients: {non_zeros}/{n_features}, Sparsity: {1-non_zeros/n_features:.4f}")
        
        # Checking convergence
        if abs(prev_loss - total_loss) < tol:
            if verbose:
                print(f"Converged at iteration {t}")
            break
            
        prev_loss = total_loss
    
    runtime = time.time() - start_time
    
    if return_iters:
        return beta, loss_history, runtime, nonzero_history, beta_history
    else:
        return beta, loss_history, runtime, nonzero_history



# Adam implementation with L1 or Elastic Net regularization
def adam(X, y, alpha=0.001, beta1=0.9, beta2=0.999, 
         lambda1=0.1, lambda2=0.0, penalty='l1',
         tol=1e-6, max_iter=1000, epsilon=1e-8, verbose=False, return_iters=False):
    """
    Adam optimizer for single-target Poisson regression
    with L1 or Elastic Net regularization
    """
    n_samples, n_features = X.shape
    
    # Initializing coefficient vector
    beta = np.random.normal(0, 0.1, n_features)
    
    # Initialize moment estimates
    m = np.zeros(n_features)  # First moment estimate
    v = np.zeros(n_features)  # Second moment estimate
    
    prev_loss = float('inf')
    loss_history = []
    start_time = time.time()
    
    # Tracking non-zero coefficients
    nonzero_history = []
    
    # Track values at each iteration 
    if return_iters:
        beta_history = []
    
    for t in range(1, max_iter + 1):
        # Compute predictions and gradient
        linear_pred = X @ beta
        linear_pred = np.clip(linear_pred, -20, 20)
        mu = np.exp(linear_pred)
        
        # Gradient of negative log-likelihood
        grad_ll = X.T @ (mu - y)
        
        # Add regularization gradient
        if penalty == 'l1':
            # Pure L1:  Subgradient for non-zero elements
            grad = grad_ll + lambda1 * np.sign(beta) * (np.abs(beta) > 0)
        elif penalty == 'elasticnet':
            # Elastic Net:  Combined gradient
            grad = grad_ll + lambda1 * np.sign(beta) * (np.abs(beta) > 0) + lambda2 * beta
        else:
            raise ValueError(f"Unknown penalty: {penalty}")
        
        # Updating biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        # Updating biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Computing bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)
        # Computing bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)
        
        # Updating parameters
        beta = beta - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Apply proximal operator for L1 regularization 
        if penalty == 'l1' or penalty == 'elasticnet':
            beta = np.sign(beta) * np.maximum(np.abs(beta) - lambda1 * alpha, 0)
        
        # Compute loss
        ll = poisson_log_likelihood(beta, X, y)
        
        # Adding regularization component to loss
        reg_pen = 0
        if penalty == 'l1':
            reg_pen = lambda1 * np.sum(np.abs(beta))
        elif penalty == 'elasticnet':
            reg_pen = lambda1 * np.sum(np.abs(beta)) + (lambda2 / 2) * np.sum(beta**2)
        
        total_loss = ll + reg_pen
        loss_history.append(total_loss)
        
        # Tracking non-zero coefficients
        non_zeros = np.sum(np.abs(beta) > 1e-6)
        nonzero_history.append(non_zeros)
        
        # Tracking beta values
        if return_iters:
            beta_history.append(beta.copy())
        
        if verbose and t % 100 == 0:
            print(f"Iteration {t}, Loss: {total_loss:.4f}, Log-likelihood: {ll:.4f}, Penalty: {reg_pen:.4f}")
            print(f"Non-zero coefficients: {non_zeros}/{n_features}, Sparsity: {1-non_zeros/n_features:.4f}")
        
        # Checking convergence
        if abs(prev_loss - total_loss) < tol:
            if verbose:
                print(f"Converged at iteration {t}")
            break
            
        prev_loss = total_loss
    
    runtime = time.time() - start_time
    
    if return_iters:
        return beta, loss_history, runtime, nonzero_history, beta_history
    else:
        return beta, loss_history, runtime, nonzero_history

##GLM implementation

def glmnet(
    X, y,
    alpha=1.0,
    lambda1=1.0,
    lambda2=1.0,
    penalty='elasticnet',
    tol=1e-4,
    max_iter=1000,
    fit_intercept=False,
    verbose=False,
    epsilon=1e-8, 
    return_iters=False,
    is_pre_scaled=False,
    lr_schedule='inverse_time',  # Options: 'constant', 'inverse_time', 'exponential', 'step'
    initial_lr=0.1,
    decay_rate=0.01,
    step_size=100,  # For step decay
    step_factor=0.5  # For step decay
):
    """
   Glmnet
    """
    # Standardizing features
    if not is_pre_scaled:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Adding intercept column
    if fit_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])
    
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)  # Initializing coefficients
    
    # Computing regularization parameters
    total_penalty = lambda1 + lambda2
    if penalty == 'l1':
        l1_ratio = 1.0
    elif penalty == 'l2':
        l1_ratio = 0.0
    elif penalty == 'elasticnet':
        l1_ratio = lambda1 / total_penalty if total_penalty > 0 else 0.0
    else:  # 'none'
        l1_ratio = 0.0
        total_penalty = 0.0
    
    # Tracking variables
    loss_history = []
    beta_history = []
    nonzero_history = []
    lr_history = []
    start_time = time.time()
    
    # Function to calculate learning rate based on the selected schedule
    def get_learning_rate(iteration):
        if lr_schedule == 'constant':
            return initial_lr
        elif lr_schedule == 'inverse_time':
            return initial_lr / (1.0 + decay_rate * iteration)
        elif lr_schedule == 'exponential':
            return initial_lr * np.exp(-decay_rate * iteration)
        elif lr_schedule == 'step':
            return initial_lr * (step_factor ** (iteration // step_size))
        else:
            return initial_lr  # Default to constant 
    
    # Main optimization loop
    for iteration in range(max_iter):
        # Storing current coefficients for convergence check
        old_beta = beta.copy()
        
        # Compute linear predictor and expected values 
        eta = X @ beta
        # Add clipping to prevent exponential overflow
        eta = np.clip(eta, -20, 20)  # Safe range for exp()
        mu = np.exp(eta) + epsilon  # Add epsilon for numerical stability
        
        # Compute gradient 
        residual = y - mu
        # Checking for and handle NaN/Inf values
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        gradient = -X.T @ residual / n_samples
        
        # Adding regularization gradients
        if total_penalty > 0:
            # L2 penalty component with safeguards
            l2_grad = total_penalty * (1 - l1_ratio) * beta
            # L1 penalty component (subgradient)
            l1_grad = total_penalty * l1_ratio * np.sign(beta)
            # Don't regularize intercept if present
            if fit_intercept:
                l2_grad[0] = 0
                l1_grad[0] = 0
            gradient += l2_grad + l1_grad
        
        # Adding gradient clipping to prevent extreme updates
        max_grad_norm = 10.0
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > max_grad_norm:
            gradient = gradient * (max_grad_norm / grad_norm)
        
        # Calculating current learning rate using the selected schedule
        learning_rate = get_learning_rate(iteration)
        lr_history.append(learning_rate)
        
        # Updating coefficients with the current learning rate
        beta -= learning_rate * gradient
        
        # Computing loss with safety checks
        log_likelihood = np.sum(y * eta - mu)
        reg_penalty = 0
        if total_penalty > 0:
            # Clip beta to prevent overflow in beta**2
            beta_clipped = np.clip(beta, -20, 20)
            l2_penalty = 0.5 * total_penalty * (1 - l1_ratio) * np.sum(beta_clipped**2)
            l1_penalty = total_penalty * l1_ratio * np.sum(np.abs(beta))
            reg_penalty = l2_penalty + l1_penalty
        loss = -log_likelihood + reg_penalty
        
        # Check for invalid loss
        if not np.isfinite(loss):
            if verbose:
                print(f"Non-finite loss detected at iteration {iteration}, resetting to previous beta")
            beta = old_beta
            break
            
        loss_history.append(loss)
        
        # Storing iteration information
        if return_iters:
            beta_history.append(beta.copy())
            nonzero_history.append(np.sum(np.abs(beta) > 1e-6))
        
        # Checking convergence
        beta_change = np.linalg.norm(beta - old_beta)
        if beta_change < tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations")
            break
        
        # Print progress if verbose
        if verbose and iteration % 100 == 0:
            print(f"Iter {iteration}: Loss = {loss:.4f}, LR = {learning_rate:.6f}, "
                  f"Non-zero = {np.sum(np.abs(beta) > 1e-6)}")
    
    runtime = time.time() - start_time
    
    if verbose:
        print(f"Training completed in {runtime:.4f} seconds")
        print(f"Final loss: {loss_history[-1]:.4f}")
        print(f"Final learning rate: {lr_history[-1]:.6f}")
        print(f"Non-zero coefficients: {np.sum(np.abs(beta) > 1e-6)}")
    
    # Tracking non-zero coefficients for the final model
    if not nonzero_history:
        nonzero_history = [np.sum(np.abs(beta) > 1e-6)]
    
    # For consistency with other optimizers
    if return_iters:
        if not beta_history:
            beta_history = [beta.copy()]
        # Add learning rate history to the return tuple
        return beta, loss_history, runtime, nonzero_history, beta_history, lr_history
    else:
        return beta, loss_history, runtime, nonzero_history


def evaluate_lr_schedules(X, y, schedules=True, alphas=None, max_iter=500, verbose=True):
    """
    Compare different learning rate schedules for GLMnet optimization.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target variable (count data).
    schedules : list or None
        List of learning rate schedules to evaluate.
    alphas : list or None
        List of regularization strengths to evaluate.
    max_iter : int
        Maximum number of iterations for each run.
    verbose : bool
        Whether to print results.
        
    Returns:
    --------
    dict
        Results for each schedule and alpha.
    """
    if schedules is None:
        schedules = ['constant', 'inverse_time', 'exponential', 'step']
    
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0]
    
    results = {}
    
    for schedule in schedules:
        schedule_results = {}
        
        for alpha in alphas:
            if verbose:
                print(f"\nEvaluating {schedule} schedule with alpha={alpha}")
            
            # Run GLMnet with the current schedule and alpha
            beta, losses, runtime, nonzeros, betas, lrs = glmnet(
                X, y,
                alpha=alpha,
                penalty='elasticnet',
                max_iter=max_iter,
                verbose=verbose,
                lr_schedule=schedule,
                return_iters=True
            )
            
            # Store results
            schedule_results[alpha] = {
                'beta': beta,
                'final_loss': losses[-1],
                'iterations': len(losses),
                'runtime': runtime,
                'nonzero_coefs': nonzeros[-1],
                'loss_history': losses,
                'lr_history': lrs
            }
            
            if verbose:
                print(f"  Final loss: {losses[-1]:.6f}")
                print(f"  Iterations: {len(losses)}")
                print(f"  Runtime: {runtime:.4f} seconds")
                print(f"  Non-zero coefficients: {nonzeros[-1]}")
        
        results[schedule] = schedule_results
    
    return results
